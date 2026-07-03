# Auditoria do Estado Atual do AssistenteIA

Data da auditoria: 2026-06-07

## Escala usada

- **Implementado e funcional**: existe codigo ligado ao fluxo principal, com comportamento util e testes ou verificacao razoavel.
- **Parcialmente implementado**: existe codigo e alguma integracao, mas ainda ha limitacoes relevantes, heuristicas frageis ou falta de cobertura.
- **Apenas estrutura/base**: existe modulo, classe, tabela ou configuracao, mas ainda nao entrega a capacidade completa esperada.
- **Nao implementado**: nao existe implementacao real no projeto atual.

## Resumo executivo

O AssistenteIA ja tem uma base funcional de aplicacao desktop local com PySide6, Ollama, Tool Registry, Agent Loop limitado, memoria local, memoria permanente SQLite, estados de presenca, contextos automaticos, observacao passiva simples e delegacao por heuristicas.

O projeto ainda nao e um agente autonomo completo. Muitas capacidades existem numa primeira versao funcional, mas dependem de regras deterministicas e palavras-chave. A parte mais madura e o acesso seguro a ficheiros da workspace. A parte menos madura e RAG/contexto documental avancado, monitorizacao real de ficheiros abertos e decisao inteligente robusta.

## Estado por componente

| Componente | Estado | Comentario curto |
|---|---|---|
| Agent Loop | Parcialmente implementado | Usa ferramentas, aceita tarefas compostas simples e tem limite de 5 passos, mas o planeamento e maioritariamente heuristico. |
| Presence Manager | Implementado e funcional | Estados e flags existem, UI permite mudar estado, motor respeita resposta/memoria/ferramentas. |
| Memory Manager | Parcialmente implementado | Historico JSON e memoria SQLite funcionam, mas consolidacao/curadoria de memoria ainda e simples. |
| Context Manager | Parcialmente implementado | Contextos automaticos existem e entram no prompt, mas sao detetados por palavras-chave. |
| Context Observer | Parcialmente implementado | Observa janela ativa, app ativa, recentes e tempo, mas so em modo passivo/foco e com heuristicas. |
| Timeline | Parcialmente implementado | Eventos, datas relativas, projetos e pessoas existem, mas parsing temporal e semantico e limitado. |
| Task Manager | Parcialmente implementado | Cria e consulta tarefas em SQLite, mas nao ha notificacoes nem gestao completa de estados. |
| Tool Registry | Implementado e funcional | Registo automatico por decorador, Tool class, permissoes e descricoes existem. |
| RAG | Apenas estrutura/base | Ha leitura de documentos e memoria pesquisavel, mas nao ha pipeline RAG documental/indexacao/chunking. |
| Delegacao | Parcialmente implementado | Decide local/ChatGPT/Codex/ferramenta externa por heuristicas e prepara prompt, mas nao executa handoff real. |
| Monitorizacao de aplicacoes | Parcialmente implementado | Captura processo ativo no Windows via `ctypes`, integrada no Context Observer. |
| Detecao da janela ativa | Implementado e funcional | Usa `GetForegroundWindow` e `GetWindowTextW`; funciona em Windows. |
| Detecao de ficheiros abertos | Apenas estrutura/base | Lista atalhos/itens recentes do Windows, nao ficheiros atualmente abertos. |
| Detecao de projetos ativos | Parcialmente implementado | Infere por titulo da janela e nome do projeto; nao deteta projetos reais de IDE de forma robusta. |
| Persistencia SQLite | Implementado e funcional | Memoria permanente, timeline, tarefas e context observer usam SQLite local. |
| Pesquisa semantica | Parcialmente implementado | Usa embeddings do Ollama quando disponiveis, com fallback textual; nao ha vector store dedicado. |

## Analise detalhada

### Agent Loop

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/agent.py`
- `assistant/conversation.py`
- `tests/test_agent.py`

O que esta implementado:

- `Agent.run()` recebe mensagem e `AgentContext`.
- O agente pode responder diretamente via LLM.
- O agente pode escolher ferramentas via plano deterministico ou `llm.choose_tool()`.
- Existe limite de passos: `MAX_AGENT_STEPS = 5`.
- Existe lista de observacoes intermedias (`AgentStep`).
- Consegue executar tarefas compostas simples:
  - listar ficheiros e resumir o primeiro;
  - ler ficheiro e criar nota;
  - procurar documentos relevantes;
  - analisar ficheiros existentes.
- Criacao de ficheiros exige confirmacao antes de executar.
- `DEBUG_AGENT` mostra passos internos quando ativo.

Limitacoes:

- O planeamento real ainda e sobretudo baseado em regras deterministicas e regex.
- O LLM so escolhe uma ferramenta inicial; o encadeamento complexo e controlado por codigo.
- Nao existe ciclo ReAct livre nem validacao semantica robusta de planos.
- A memoria e o contexto entram no prompt, mas nao ha raciocinio persistente estruturado sobre planos.

Classificacao final: **Parcialmente implementado**.

### Presence Manager

**Estado: Implementado e funcional**

Ficheiros principais:

- `assistant/presence_manager.py`
- `app.py`
- `ui/main_window.py`
- `tests/test_presence_manager.py`
- `tests/test_presence_integration.py`

O que esta implementado:

- Enum com estados:
  - `ACTIVE_CONVERSATION`
  - `PASSIVE_MONITORING`
  - `FOCUS_MODE`
  - `PRIVATE_MODE`
  - `OFFLINE`
- Flags comportamentais:
  - `can_respond`
  - `can_use_tools`
  - `can_make_suggestions`
  - `can_ask_confirmation`
  - `can_store_memory`
  - `can_observe_activity`
  - `can_interrupt`
- UI tem seletor de estado de presenca.
- `AssistantEngine.respond()` respeita estados que impedem resposta.
- `PRIVATE_MODE` impede gravacao de memoria.
- `PASSIVE_MONITORING` e `FOCUS_MODE` permitem observacao passiva via timer.

Limitacoes:

- `FOCUS_MODE` ainda nao implementa logica real de "interromper apenas em situacoes importantes".
- `PASSIVE_MONITORING` nao inicia conversa, mas tambem nao tem logica rica de atualizacao de memoria.

Classificacao final: **Implementado e funcional**, com comportamento ainda simples.

### Memory Manager

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/memory.py`
- `assistant/long_term_memory.py`
- `assistant/conversation.py`
- `tests/test_long_term_memory.py`

O que esta implementado:

- Historico recente da conversa em JSON (`data/history.json`).
- Memoria permanente SQLite (`data/long_term_memory.sqlite`).
- Categorias:
  - `perfil_utilizador`
  - `projetos`
  - `conversas`
  - `preferencias`
  - `tarefas`
  - `relacoes`
- Comandos:
  - `lembra-te que...`
  - `lembra que...`
  - `nao te esquecas que...`
  - `guarda isto...`
  - `esquece...`
  - `o que sabes sobre...`
- Classificacao simples de memorias por conteudo.
- Pesquisa textual e semantica quando embeddings estao disponiveis.

Limitacoes:

- Nao ha processo de consolidacao de memoria.
- Nao ha deduplicacao robusta.
- Nao ha criterios sofisticados para decidir o que deve ser memoria permanente.
- Ainda existem nomes antigos no codigo, como `_try_profile_memory`, apesar de funcionarem como memoria de utilizador.

Classificacao final: **Parcialmente implementado**.

### Context Manager

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/context_manager.py`
- `assistant/conversation.py`
- `assistant/agent.py`
- `tests/test_context_manager.py`
- `tests/test_context_manager_integration.py`

O que esta implementado:

- Conceito de contexto substitui perfis manuais.
- Contextos existentes:
  - `PERSONAL_CONTEXT`
  - `WORK_CONTEXT`
  - `TECH_CONTEXT`
  - `PRODUCTIVITY_CONTEXT`
  - `TRAVEL_CONTEXT`
  - `SOCIAL_CONTEXT`
- Cada contexto tem:
  - descricao;
  - categoria de memoria associada;
  - peso de relevancia;
  - razao de ativacao.
- O motor identifica contextos antes do Agent Loop.
- O system prompt recebe os contextos ativos.
- O painel de debug pode mostrar contextos ativos quando `DEBUG_AGENT=true`.

Limitacoes:

- A deteccao e baseada em palavras-chave, nao em classificacao semantica pelo LLM.
- O peso e calculado por regras fixas simples.
- A memoria associada ao contexto ainda nao filtra consultas por categoria de forma forte.

Classificacao final: **Parcialmente implementado**.

### Context Observer

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/context_observer.py`
- `app.py`
- `assistant/conversation.py`
- `tests/test_context_observer.py`
- `tests/test_context_observer_integration.py`

O que esta implementado:

- `ContextObserver.observe_once()` cria snapshots.
- Guarda snapshots em SQLite.
- Guarda:
  - aplicacao ativa;
  - janela ativa;
  - ficheiros recentes;
  - projeto inferido;
  - data/hora observada.
- Acumula tempo por atividade em tabela `activity_time`.
- Integracao com `app.py` atraves de `QTimer`.
- So observa quando o Presence Manager permite (`PASSIVE_MONITORING` ou `FOCUS_MODE`).
- O ultimo snapshot entra no contexto passado ao agente.

Limitacoes:

- Observacao e periodica e simples.
- Nao observa quando esta em `ACTIVE_CONVERSATION`.
- Ficheiros recentes nao sao ficheiros abertos.
- Projeto ativo e inferido por titulo de janela.
- Nao ha UI para ver historico de observacao ou tempo acumulado.

Classificacao final: **Parcialmente implementado**.

### Timeline

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/long_term_memory.py`
- `assistant/conversation.py`
- `tests/test_timeline.py`
- `tests/test_timeline_integration.py`

O que esta implementado:

- Tabela SQLite `timeline_events`.
- Registo de eventos com:
  - data;
  - conteudo;
  - projeto;
  - pessoas.
- Inferencia simples de datas relativas:
  - ontem;
  - anteontem;
  - semana passada;
  - ha X dias/semanas/meses.
- Consultas:
  - eventos de uma data;
  - periodo;
  - contexto recente de trabalho;
  - inicio de projeto.
- Integracao com comandos no `AssistantEngine`.

Limitacoes:

- Parsing temporal e limitado.
- Nao ha calendario real.
- Nao ha agrupamento ou resumo temporal avancado.
- Associacao a projetos/pessoas e heuristica.

Classificacao final: **Parcialmente implementado**.

### Task Manager

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/long_term_memory.py`
- `assistant/conversation.py`
- `tests/test_tasks.py`
- `tests/test_tasks_integration.py`

O que esta implementado:

- Tabela SQLite `tasks`.
- Criacao de tarefas.
- Datas simples:
  - hoje;
  - amanha;
  - depois de amanha;
  - proxima semana;
  - ha X dias/semanas/meses para inferencia auxiliar.
- Associacao simples a projeto.
- Consulta de tarefas para hoje.
- Consulta de tarefas pendentes.
- Tarefas tambem sao guardadas como memoria de categoria `tarefas`.

Limitacoes:

- Nao ha notificacoes automaticas.
- Nao ha UI dedicada.
- Nao ha comandos para concluir, adiar, editar ou apagar tarefas.
- Nao ha scheduler real de lembretes.

Classificacao final: **Parcialmente implementado**.

### Tool Registry

**Estado: Implementado e funcional**

Ficheiros principais:

- `assistant/tool_registry.py`
- `assistant/tools.py`
- `tests/test_tools.py`

O que esta implementado:

- Classe `Tool`.
- Classe `ToolRegistry`.
- Decorador `register`.
- Registo automatico ao importar `assistant.tools`.
- Cada ferramenta tem:
  - nome;
  - descricao;
  - funcao associada;
  - permissoes;
  - flag `remember_result`.
- Descricao das ferramentas e passada ao LLM/Agent.
- Ferramentas implementadas:
  - `list_workspace_files`;
  - `read_workspace_file`;
  - `create_workspace_file`.

Limitacoes:

- Permissoes sao metadata; a seguranca real ainda vive nas funcoes e no `WorkspaceGuard`.
- Nao ha mecanismo generico de enforcement de permissoes antes de `tool.run`.

Classificacao final: **Implementado e funcional**.

### RAG

**Estado: Apenas estrutura/base**

Ficheiros relacionados:

- `assistant/tools.py`
- `assistant/document_reader.py`
- `assistant/long_term_memory.py`
- `assistant/agent.py`

O que existe:

- Leitura segura de `.txt`, `.md`, `.docx` e `.pdf`.
- Resumo de ficheiros pequenos usando Ollama.
- Pesquisa em memoria permanente.
- Embeddings guardados para memorias/tarefas quando o embedder devolve vetores.

O que ainda falta para ser RAG real:

- Indexacao de documentos.
- Chunking.
- Embeddings por chunk documental.
- Tabela/vector store para documentos.
- Recuperacao top-k de excertos.
- Citacoes ou referencias a partes do documento.
- Atualizacao de indice quando ficheiros mudam.

Classificacao final: **Apenas estrutura/base**.

### Delegacao

**Estado: Parcialmente implementado**

Ficheiros principais:

- `assistant/delegation.py`
- `assistant/conversation.py`
- `tests/test_delegation.py`
- `tests/test_delegation_integration.py`

O que esta implementado:

- `DelegationManager`.
- Alvos:
  - local;
  - ChatGPT;
  - Codex;
  - ferramenta externa.
- Detecao por heuristicas/palavras-chave.
- Preparacao de prompt com:
  - destino sugerido;
  - contextos ativos;
  - pedido original;
  - contexto relevante;
  - objetivo.
- Integrado antes do Agent Loop.

Limitacoes:

- Nao abre ChatGPT, Codex ou ferramentas externas.
- Nao cria handoff real.
- Pode delegar cedo demais em alguns pedidos por causa das palavras-chave.
- Estrategia nao e decidida semanticamente por LLM.

Classificacao final: **Parcialmente implementado**.

### Monitorizacao de aplicacoes

**Estado: Parcialmente implementado**

Ficheiro principal:

- `assistant/context_observer.py`

O que esta implementado:

- Em Windows, usa `GetForegroundWindow`, `GetWindowThreadProcessId`, `OpenProcess` e `QueryFullProcessImageNameW`.
- Obtem nome do processo ativo.
- Guarda app ativa em SQLite.
- Acumula tempo por app/janela/projeto.

Limitacoes:

- Apenas observa a app em primeiro plano.
- Nao monitoriza lista de apps abertas.
- Nao deteta mudancas em tempo real fora do timer.
- Nao existe painel de atividade.

Classificacao final: **Parcialmente implementado**.

### Detecao da janela ativa

**Estado: Implementado e funcional**

Ficheiro principal:

- `assistant/context_observer.py`

O que esta implementado:

- Captura titulo da janela ativa com `GetWindowTextW`.
- Guarda em SQLite.
- Usa no contexto observado passado ao agente.

Limitacoes:

- Funciona apenas em Windows.
- Depende do titulo da janela estar disponivel.
- Nao faz normalizacao semantica da janela.

Classificacao final: **Implementado e funcional** para a versao Windows atual.

### Detecao de ficheiros abertos

**Estado: Apenas estrutura/base**

Ficheiro principal:

- `assistant/context_observer.py`

O que esta implementado:

- `_recent_file_names()` le a pasta `APPDATA/Microsoft/Windows/Recent`.
- Guarda nomes recentes no snapshot.

O que nao esta implementado:

- Detecao real de ficheiros atualmente abertos.
- Ligacao a handles do sistema.
- Integracao com Office/VS Code/Explorer para documentos abertos.
- Resolucao robusta de atalhos `.lnk` para caminhos reais.

Classificacao final: **Apenas estrutura/base**.

### Detecao de projetos ativos

**Estado: Parcialmente implementado**

Ficheiro principal:

- `assistant/context_observer.py`

O que esta implementado:

- `_infer_project()` tenta inferir projeto pelo titulo da janela.
- Reconhece nome do root do projeto.
- Tenta separar nomes em titulos de VS Code, Cursor e PyCharm.

Limitacoes:

- Heuristico.
- Nao usa APIs das IDEs.
- Nao confirma caminho do projeto aberto.
- Pode confundir titulo de ficheiro com projeto.

Classificacao final: **Parcialmente implementado**.

### Persistencia SQLite

**Estado: Implementado e funcional**

Ficheiros principais:

- `assistant/long_term_memory.py`
- `assistant/context_observer.py`

Bases/tabelas:

- `data/long_term_memory.sqlite`
  - `memories`
  - `timeline_events`
  - `tasks`
- `data/context_observer.sqlite`
  - `context_snapshots`
  - `activity_time`

O que esta implementado:

- Criacao automatica de tabelas.
- Indices basicos.
- Validacao para manter DB dentro da pasta `data`.
- Persistencia entre execucoes.

Limitacoes:

- Nao ha migracoes versionadas.
- Nao ha compactacao/limpeza.
- Nao ha backup/exportacao.

Classificacao final: **Implementado e funcional**.

### Pesquisa semantica

**Estado: Parcialmente implementado**

Ficheiro principal:

- `assistant/long_term_memory.py`

O que esta implementado:

- Interface `Embedder`.
- `OllamaClient.embed()` chama `/api/embeddings`.
- Memorias e tarefas podem guardar embedding em JSON.
- `search()` calcula similaridade coseno quando existem embeddings.
- Fallback textual quando embeddings nao existem ou nao devolvem resultados.

Limitacoes:

- Embeddings dependem do modelo configurado no Ollama.
- Nao ha modelo de embeddings dedicado configuravel.
- Vetores sao guardados como JSON em SQLite, sem indice vetorial.
- Nao ha pesquisa semantica documental, apenas memoria/tarefas.
- Se embeddings falham, a aplicacao fica em pesquisa textual simples.

Classificacao final: **Parcialmente implementado**.

## Observacoes transversais

### Segurança de workspace

**Estado geral: Implementado e funcional**

O acesso a ficheiros e protegido por:

- `WorkspaceGuard`;
- bloqueio de caminhos absolutos;
- bloqueio de `..`;
- extensoes permitidas;
- nao sobrescrever ficheiros existentes;
- testes de path traversal.

Esta e uma das partes mais solidas do projeto.

### Interface

**Estado geral: Parcialmente implementado**

Existe UI funcional com:

- conversa;
- input;
- botao Enviar;
- botao Limpar conversa;
- seletor de presenca;
- painel debug de contextos opcional.

Ainda falta:

- painel dedicado de memoria;
- painel de tarefas;
- painel de timeline;
- painel de atividade observada;
- UI para confirmacoes mais rica do que responder `sim`/`nao`.

### Qualidade do codigo

Pontos positivos:

- Separacao razoavel por modulos.
- Ferramentas isoladas.
- Segurança de workspace centralizada.
- Testes para varios modulos.
- Documentacao criada em `README.md`, `docs/architecture.md` e `docs/VISION.md`.

Riscos atuais:

- Existem nomes legados ligados a perfis (`active_profile_name`, `set_profile`, `_try_profile_memory`), embora os perfis manuais ja nao estejam na UI.
- Ha alguns textos com problemas de encoding/mojibake em strings antigas.
- Algumas decisoes importantes ainda dependem de heuristicas por palavras-chave.
- A integracao com Ollama em runtime nao foi validada nesta auditoria.

## Conclusao

O AssistenteIA esta num estado de prototipo funcional avancado, nao apenas numa estrutura vazia. Ja existe uma aplicacao desktop com memoria, ferramentas seguras, contexto automatico, presenca, observacao passiva e delegacao basica.

No entanto, as capacidades mais ambiciosas ainda estao em versao inicial:

- agente verdadeiramente autonomo;
- RAG documental;
- monitorizacao real de ficheiros abertos;
- gestao completa de tarefas/lembretes;
- inferencia rica de contexto;
- pesquisa semantica robusta.

Prioridade recomendada para proximas fases:

1. Estabilizar arquitetura e remover nomes legados de perfis.
2. Criar RAG documental real para a workspace.
3. Melhorar o Context Manager com classificacao semantica.
4. Criar UI para memoria, tarefas, timeline e atividade.
5. Adicionar migracoes SQLite versionadas.
6. Melhorar testes end-to-end com Ollama mockado.
