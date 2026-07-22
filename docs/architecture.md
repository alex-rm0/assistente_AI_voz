# Arquitetura do AssistenteIA

Este documento descreve a arquitetura técnica atual.

O modelo cognitivo que orienta estes componentes está em [COGNITIVE_MODEL.md](COGNITIVE_MODEL.md).
A arquitetura cognitiva dos módulos de pensamento está em [COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md).

O AssistenteIA esta separado em quatro areas principais:

## Conversacao

`assistant/conversation.py` contem o `AssistantEngine`.

Responsabilidades:
- receber mensagens do utilizador;
- aplicar a politica de seguranca;
- identificar contextos automaticos atraves do Context Manager;
- consultar a camada de delegacao antes do Agent Loop;
- pedir ao LLM para decidir se deve usar uma ferramenta;
- executar ferramentas atraves do registry;
- guardar apenas historico de conversa apropriado;
- enviar conversa normal para o Ollama.

O `AssistantEngine` não deve tentar usar todas as capacidades em todas as
mensagens. A Executive Function decide quando usar Sistema 1 ou Sistema 2 e que
módulos cognitivos participam.

## Cognição

`assistant/cognition/` contem os módulos que implementam o modelo cognitivo:

- `intent_engine.py`: identifica o objetivo provável do pedido.
- `executive_function.py`: escolhe que módulos devem participar.
- `context_builder.py`: recolhe apenas contexto relevante.
- `reflection_engine.py`: transforma contexto em incertezas, insights e perguntas.
- `reasoning_engine.py`: cria raciocínio interno e plano, sem responder diretamente.
- `preference_builder.py`: avalia preferências antes de recomendações.

Estes módulos produzem conhecimento, estratégia e hipóteses. Não devem produzir
linguagem final para o utilizador.

## Delegacao

`assistant/delegation.py` contem o `DelegationManager`.

Responsabilidades:
- decidir se o pedido deve ser resolvido localmente;
- preparar contexto para ChatGPT quando o pedido for mais amplo ou exploratorio;
- preparar contexto para Codex quando o pedido envolver codigo, testes, Git ou alteracoes ao projeto;
- preparar contexto para uma ferramenta externa quando o pedido depender de outro programa;
- explicar ao utilizador a estrategia escolhida.

A delegacao nao executa comandos, nao abre programas e nao altera ficheiros. Apenas prepara um prompt/contexto para o destino adequado.

## Ferramentas

`assistant/tool_registry.py` define:
- `Tool`;
- `ToolRegistry`;
- `tool_registry`, o registry global.

Cada ferramenta tem:
- `name`;
- `description`;
- `function`;
- `permissions`;
- `remember_result`.

As ferramentas sao registadas automaticamente com decoradores em `assistant/tools.py`.

Ferramentas iniciais:
- `list_workspace_files`;
- `read_workspace_file`;
- `create_workspace_file`.
- `get_active_window`;
- `get_active_application`;
- `get_open_windows`;
- `get_recent_activity`.
- `get_last_context_snapshot`.
- `get_current_activity_summary`.

Todas as ferramentas de ficheiros usam `WorkspaceGuard` para garantir que so trabalham dentro da pasta `workspace`.
A leitura de documentos Word e PDF esta separada em `assistant/document_reader.py`.
As ferramentas de estado do computador leem apenas dados ja observados pelo Context Observer.
Se nao existir informacao observada suficiente, devolvem uma mensagem clara para o utilizador aguardar ou mudar de janela.
Se uma ferramenta de monitorizacao nao estiver registada, o Agent Loop responde que a ferramenta ainda nao esta ligada ao agente, sem chamar o LLM.

## Contextos

`assistant/context_manager.py` define o `ContextManager`.

O utilizador nao escolhe contextos manualmente. Em cada mensagem, o Context
Manager identifica um ou varios contextos relevantes e atribui um peso a cada um.

Contextos iniciais:
- `PERSONAL_CONTEXT`;
- `WORK_CONTEXT`;
- `TECH_CONTEXT`;
- `PRODUCTIVITY_CONTEXT`;
- `TRAVEL_CONTEXT`;
- `SOCIAL_CONTEXT`.

Cada contexto tem:
- descricao;
- memoria associada;
- peso de relevancia;
- razao de ativacao para debug.

O Agent Loop recebe os contextos ativos antes de gerar resposta ou decidir usar ferramentas.

## Context Observer

`assistant/context_observer.py` observa passivamente o computador quando o estado
de presenca permite.

Observa:
- janela ativa;
- janelas abertas;
- processos ativos;
- sessoes VSCode;
- repositorios Git no projeto;
- pasta/projeto inferido no VSCode;
- ficheiros recentemente modificados;
- ficheiros recentes do Windows.

O observer nao deve guardar todos os eventos como memoria. Em vez disso, agrega
sessoes de atividade e cria resumos curtos, por exemplo:

```text
Entre as 09:00 e as 11:00 o Alexandre trabalhou no projeto AssistenteIA usando VSCode e Git.
```

Estes resumos sao guardados na memoria permanente e na timeline.

`assistant/context_interpreter.py` traduz snapshots brutos do observer em contexto
humano util. Agrupa aplicacoes por categoria, remove ruido tecnico por defeito e
e usado pela ferramenta `get_last_context_snapshot`.

O interpretador nunca substitui o snapshot observado. A ferramenta
`get_last_context_snapshot` devolve primeiro o resumo interpretado e depois um
bloco `Snapshot bruto` com a informacao observada disponivel. Ferramentas como
`get_open_windows` e `get_active_window` continuam a devolver dados diretos do
Context Observer.

`assistant/context_reasoning.py` recebe o snapshot, contextos ativos, memoria
relevante e tarefas pendentes. A sua funcao e produzir conclusoes suportadas por
evidencias: atividade principal, projeto principal, aplicacoes relevantes,
objetivos possiveis e sugestoes opcionais. Se nao houver evidencias, nao inventa.

## Memoria

`assistant/memory.py` guarda historico simples em JSON dentro de `data/history.json`.

Responsabilidades:
- carregar historico;
- guardar historico;
- limpar historico.

A memoria nao deve guardar conteudo lido de ficheiros.

`assistant/long_term_memory.py` guarda memoria permanente em SQLite dentro de
`data/long_term_memory.sqlite`.

Tipos de memoria permanente:
- preferencias;
- projetos;
- contexto recorrente.

Comandos suportados:
- `lembra-te que...`;
- `esquece...`;
- `o que sabes sobre...`.

A memoria permanente e separada do historico da conversa. Quando o Ollama
suporta embeddings para o modelo configurado, a pesquisa usa similaridade
semantica. Se os embeddings nao estiverem disponiveis, a app usa uma pesquisa
textual simples como fallback.

## Response Composer

`assistant/response_composer.py` contem o `ResponseComposer`.

O Personal Model, o Session Manager e a memoria de longo prazo nao respondem
diretamente ao utilizador. Estes modulos apenas fornecem factos em linguagem
simples (sem categorias, confianca, timestamps ou estrutura tecnica). O
`ResponseComposer` e o unico ponto que decide o que interessa, o que omitir,
o tom e a estrutura da resposta final:

- se o pedido nao exige detalhe tecnico, os factos sao entregues ao LLM
  atraves de `ComposerRequest`, que gera a resposta em linguagem natural;
- se o utilizador pedir explicitamente detalhes ("mostra os detalhes",
  "mostra a confianca", "mostra as evidencias", "mostra os factos"), a
  resposta tecnica pre-formatada e devolvida sem chamar o LLM;
- se nao houver factos disponiveis, ou se o LLM falhar, e devolvido um
  `fallback` deterministico.

### Regra de memoria como evidencia

As memorias nunca devem aparecer literalmente na resposta final.

Cada memoria passa por tres etapas:

1. Interpretacao.
2. Contextualizacao.
3. Integracao na conversa.

O texto original da memoria deve ser tratado apenas como evidencia. Nunca deve
ser usado como resposta pronta.

Se uma resposta puder ser construida apenas copiando uma memoria, o Response
Composer falhou.

Quando existirem muitas memorias, o objetivo nao e mostrar "o que esta guardado".
O objetivo e extrair significado do conjunto: padroes, preferencias, recorrencias,
mudancas, tensoes, objetivos e sinais de contexto. A sensacao de continuidade
deve vir da interpretacao dessas evidencias, nao da sua listagem.

O Response Composer tambem aplica [VOICE_AND_CONVERSATION.md](VOICE_AND_CONVERSATION.md):
falar pouco, evitar monologos, respeitar o ritmo da conversa e nunca expor
raciocinio interno tecnico por defeito.

Comandos que ja usam o Response Composer:
- `o que sabes sobre...` / `o que sabes sobre mim` (Personal Model);
- `onde ficamos` / `em que estavamos a trabalhar`;
- `resume a ultima sessao`;
- `o que fizemos hoje`, `qual e o proximo passo`, `o que mudou desde a ultima vez`.

## Interface

`ui/main_window.py` contem a janela PySide6.

Responsabilidades:
- mostrar historico;
- recolher mensagens;
- chamar o motor de conversacao;
- limpar conversa atraves do botao "Limpar conversa".

A interface nao conhece detalhes de ferramentas, Ollama ou armazenamento.

## Fluxo

1. O utilizador escreve uma mensagem.
2. A UI chama `AssistantEngine.respond()`.
3. O motor identifica os contextos automaticos relevantes.
4. O motor aplica `check_user_request()`.
5. O motor consulta a delegacao.
6. Se o pedido for delegado, o motor devolve a estrategia e o prompt preparado.
7. Se o pedido ficar local, o motor envia ao LLM o system prompt com contextos e ferramentas disponiveis.
8. O LLM devolve uma decisao em JSON.
9. Se houver ferramenta, o registry executa-a.
10. Se nao houver ferramenta, a mensagem segue para conversa normal com Ollama.

## Adicionar Ferramentas Futuras

Para adicionar uma ferramenta:

1. Criar uma funcao em `assistant/tools.py` ou num modulo importado no arranque.
2. Decorar a funcao com `@tool_registry.register(...)`.
3. Definir `name`, `description`, `permissions` e `remember_result`.

O nucleo da aplicacao nao precisa de ser alterado.
