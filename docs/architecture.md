# Arquitetura do AssistenteIA

O AssistenteIA esta separado em quatro areas principais:

## Conversacao

`assistant/conversation.py` contem o `AssistantEngine`.

Responsabilidades:
- receber mensagens do utilizador;
- aplicar a politica de seguranca;
- pedir ao LLM para decidir se deve usar uma ferramenta;
- executar ferramentas atraves do registry;
- guardar apenas historico de conversa apropriado;
- enviar conversa normal para o Ollama.

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

Todas as ferramentas de ficheiros usam `WorkspaceGuard` para garantir que so trabalham dentro da pasta `workspace`.

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
3. O motor aplica `check_user_request()`.
4. O motor envia ao LLM a descricao das ferramentas disponiveis.
5. O LLM devolve uma decisao em JSON.
6. Se houver ferramenta, o registry executa-a.
7. Se nao houver ferramenta, a mensagem segue para conversa normal com Ollama.

## Adicionar Ferramentas Futuras

Para adicionar uma ferramenta:

1. Criar uma funcao em `assistant/tools.py` ou num modulo importado no arranque.
2. Decorar a funcao com `@tool_registry.register(...)`.
3. Definir `name`, `description`, `permissions` e `remember_result`.

O nucleo da aplicacao nao precisa de ser alterado.
