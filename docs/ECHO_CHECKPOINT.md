# Echo Checkpoint

Data: 2026-07-22  
Commit auditado: `3f6cca7`  
Estado Git no início da auditoria: limpo (`git status --short` sem alterações)

Este checkpoint descreve o estado técnico real do Echo antes de adicionar novos providers ou funcionalidades. A análise foi feita a partir do código presente no repositório, não a partir de intenções anteriores.

## Actualização Da Fase 0

Actualizado em 2026-07-22:

- `.venv` antiga estava realmente quebrada: apontava para Python da Microsoft Store em `WindowsApps`.
- A `.venv` antiga foi preservada como `.venv.broken-msstore-20260722`.
- Foi instalado Python oficial 3.11.9 no perfil do utilizador.
- Foi criada uma nova `.venv` funcional.
- O comando global `python` continua fora do PATH nesta sessão PowerShell; a execução reproduzível usa `.\.venv\Scripts\python.exe` ou a `.venv` activada.
- `requirements.txt` foi instalado.
- `pytest` passou: `497 passed`.
- A resolução do modelo Ollama foi corrigida em `app.resolve_ollama_model()`.
- Sem override, o runtime passa a usar `settings["ollama"]["model"]`, actualmente `gemma3:12b`.
- A telemetry passa a expor `model_source`.
- Não foi criada baseline oficial nova porque existem alterações de código/documentação ainda por commitar.

## 1. Visão Atual Do Echo

O Echo já deixou de ser apenas uma janela de chat com Ollama. O repositório contém uma arquitetura local com:

- duas interfaces desktop: UI clássica PySide6 e protótipo Echo OS com QWebEngineView/QWebChannel;
- `AssistantEngine` como orquestrador principal;
- caminhos determinísticos para conversa social, comandos rápidos, segurança, pesquisa, memória, tarefas e estado do computador;
- camada cognitiva com intent engine, executive function, context builder, reflection engine e reasoning engine;
- memória persistente em SQLite;
- Personal Model separado;
- Session Manager e Session Reflection;
- Tool Registry com ferramentas de workspace, tarefas, Context Observer e desktop actions;
- evals com casos fixos/gerados, relatórios e comparação;
- provider abstraction preparada para evals, mas não ainda para o runtime principal.

O projeto está numa fase de fundação avançada, mas ainda não deve ser tratado como produto estável. Há protótipos úteis, dívida técnica de encoding/memória e funcionalidades preparadas que ainda não têm ferramenta real ligada.

## 2. Arquitetura De Alto Nível

Entrypoints:

- `app.py`: entrypoint principal da app desktop. Aceita `--ui classic` e `--ui echo-os`.
- `prototype_web_ui/run_prototype.py`: entrypoint do protótipo Echo OS.
- `evals/run_evals.py`: entrypoint da suite de evals.

Camadas principais:

- UI: `ui/main_window.py`, `prototype_web_ui/window.py`, `prototype_web_ui/controller.py`, `prototype_web_ui/web/*`.
- Orquestração: `assistant/conversation.py`.
- Cognição: `assistant/cognition/*`, `assistant/planner.py`, `assistant/response_composer.py`.
- Ferramentas: `assistant/tool_registry.py`, `assistant/tools.py`, `assistant/agent.py`, `assistant/desktop_actions.py`.
- Memória: `assistant/memory.py`, `assistant/long_term_memory.py`, `assistant/personal_model.py`, `assistant/session_manager.py`.
- Contexto: `assistant/context_observer.py`, `assistant/context_interpreter.py`, `assistant/context_reasoning.py`.
- Modelo: `assistant/llm.py`, `assistant/model_provider.py`.
- Voz: `assistant/voice_input.py`, `assistant/audio_device.py`, `assistant/voice_critic.py`.
- Evals: `evals/*`.

## 3. Fluxo De Uma Mensagem

Fluxo real observado em `AssistantEngine.respond()`:

1. `respond()` envolve o turno inteiro em guarda de exceções.
2. `_respond_inner()` inicia telemetry e bookkeeping de sessão.
3. Extração passiva de memória pode ocorrer, excepto em caminhos explicitamente excluídos.
4. Presence commands são tratados antes de bloqueios de estado.
5. Se o estado de presença não permite resposta, devolve resposta silenciosa/local.
6. Fast router tenta comandos rápidos e confirmações pendentes.
7. Topic shift e research request são tratados de forma determinística.
8. Perguntas sobre estado do sistema passam por ferramentas do Context Observer.
9. Conversa social simples usa caminhos curtos sem LLM.
10. Segurança bloqueia pedidos perigosos.
11. Escrita/consulta/inventário de memória são tratados deterministicamente quando possível.
12. Conhecimento geral pode ir ao Response Composer.
13. Conversational refinement pode responder a casos conversacionais específicos.
14. Intent Engine e Executive Function escolhem estratégia.
15. Cognitive loop constrói contexto/reflexão/raciocínio quando necessário.
16. Agent pode planear/usar ferramentas.
17. `_complete_turn()` guarda histórico, sessão, telemetry, grounding e eventos UI.

Risco atual: `AssistantEngine` concentra demasiadas responsabilidades e tem muitos caminhos `_try_*`. Está funcional, mas já pede decomposição.

## 4. UI E Estados

### UI Clássica

Ficheiro principal: `ui/main_window.py`  
Estado: IMPLEMENTADO

Funcional:

- janela PySide6;
- histórico de conversa;
- input com Enter/Ctrl+Enter/Esc;
- botão Enviar;
- botão Limpar conversa;
- modos de presença;
- painel de tarefas expansível;
- botão Mic e botão Testar microfone;
- workers Qt para chat, voz e teste de microfone.

Limitações:

- UI ainda é utilitária, não representa a filosofia final da esfera/presença;
- debug e textos técnicos podem aparecer no modo normal se flags estiverem activas;
- voz depende de Whisper, sounddevice, ffmpeg e microfone local.

### Echo OS / QWebEngineView / QWebChannel

Ficheiros:

- `prototype_web_ui/window.py`
- `prototype_web_ui/controller.py`
- `prototype_web_ui/web/index.html`
- `prototype_web_ui/web/echo_ui.js`
- `prototype_web_ui/web/echo_entity.js`
- `prototype_web_ui/web/styles.css`

Estado: PARCIAL / PROTÓTIPO FUNCIONAL

Funcional:

- QWebEngineView carrega UI local;
- QWebChannel regista `echoController`;
- sinais `requestStarted`, `responseReady`, `stateChanged`, `uiEvent`, `requestFinished`;
- processamento em worker thread;
- resposta volta ao controller na thread principal;
- estados `thinking`, `speaking`, `idle`;
- workspace visual de pesquisa básico;
- CLEAR chama `engine.clear_conversation()`.

Limitações:

- ainda existem textos corrompidos/mojibake em partes da UI/logs;
- workspace de pesquisa depende de eventos, mas não há pesquisa web real ligada;
- boot, first run, workspaces de código/reunião/calendário e UI adaptativa completa não existem.

## 5. Conversa E Routing

Ficheiro principal: `assistant/conversation.py`  
Estado: IMPLEMENTADO, mas complexo

Componentes relevantes:

- `AssistantEngine.respond()`;
- `_try_fast_route()`;
- `_try_research_request()`;
- `_try_system_state_tool_query()`;
- `_try_pure_social_turn()`;
- `_try_memory_write_command()`;
- `_try_memory_recall_question()`;
- `_try_general_knowledge_query()`;
- `_run_cognitive_loop()`;
- `_complete_turn()`;
- `get_last_turn_telemetry()`.

Routing determinístico existente:

- conversa social simples;
- comandos rápidos de URL/sites/pesquisa rápida;
- confirmações pendentes;
- segurança;
- research intent;
- estado de janelas/aplicações;
- memória explícita;
- tarefas;
- presença;
- linguagem;
- briefing/session continuity.

Limitações:

- `conversation.py` é grande e acumula regras de routing, memory policy, telemetry e composição;
- alguns caminhos ainda dependem de heurísticas sensíveis a fraseado;
- risco de caminhos demasiado cedo ou demasiado tarde no pipeline;
- ainda há dívida de separação entre routing, operação activa, memória e resposta final.

## 6. Memória E Personal Model

### Conversation Memory

Ficheiro: `assistant/memory.py`  
Estado: IMPLEMENTADO

Guarda histórico curto em JSON (`data/history.json`) e suporta limpeza.

### Long Term Memory

Ficheiro: `assistant/long_term_memory.py`  
Estado: IMPLEMENTADO / PARCIAL

Tabelas observadas em `data/long_term_memory.sqlite`:

- `memories`: 76 registos;
- `timeline_events`: 75 registos;
- `tasks`: 1 registo;
- `preferences`: 9 registos;
- `structured_facts`: 2 registos.

Funcional:

- memórias por categoria;
- timeline;
- tarefas;
- preferências;
- structured facts;
- embeddings quando o embedder devolve vector;
- fallback por pesquisa textual.

Limitações:

- embedding local depende de `/api/embeddings`;
- dados antigos podem conter respostas erradas ou mojibake;
- `memories` genéricas podem contaminar prompts se usadas sem curadoria;
- há risco de duplicação entre `memories`, `structured_facts`, `timeline_events`, `session_summaries` e Personal Model.

### Personal Model

Ficheiro: `assistant/personal_model.py`  
Estado: IMPLEMENTADO / INICIAL

Tabela observada em `data/personal_model.sqlite`:

- `personal_model_entries`: 1 registo.

Funcional:

- categorias;
- confiança;
- evidência;
- origem;
- estado;
- search/list/delete/decrease confidence;
- respostas formatadas de forma mais natural.

Limitações:

- ainda pouco populado;
- não existe ainda motor robusto de inferência longitudinal;
- Preference Builder existe, mas não está maduro como modelo de pessoa.

### Session Manager

Ficheiro: `assistant/session_manager.py`  
Estado: IMPLEMENTADO / PARCIAL

Tabela observada em `data/session_manager.sqlite`:

- `session_summaries`: 66 registos.

Funcional:

- start/end session;
- fim por inactividade;
- resumo de sessão;
- ferramentas usadas;
- tarefas alteradas;
- decisões;
- próximo passo;
- respostas a “onde ficámos?”.

Limitações:

- pode haver resumos antigos demasiado técnicos;
- precisa de política de curadoria/limpeza;
- session summaries devem ser usados como contexto, não como texto copiado.

## 7. Context Observer

Ficheiros:

- `assistant/context_observer.py`
- `assistant/context_interpreter.py`
- `assistant/context_reasoning.py`

Estado: IMPLEMENTADO / PARCIAL

Base observada: `data/context_observer.sqlite`

- `context_snapshots`: 5;
- `activity_time`: 1;
- `context_summaries`: 0.

Funcional:

- snapshots;
- janela activa;
- janelas abertas via Win32;
- processos;
- VS Code sessions;
- Git repositories;
- ficheiros modificados;
- resumo interpretado;
- raciocínio sobre actividade actual;
- flush de summary.

Limitações:

- poucos snapshots reais na base actual;
- summaries ainda a zero;
- dependente de Windows APIs;
- qualidade depende de periodicidade e de permissões;
- risco de ruído técnico se o interpreter não filtrar bem.

## 8. Ferramentas E Ações

Ficheiros:

- `assistant/tool_registry.py`
- `assistant/tools.py`
- `assistant/agent.py`
- `assistant/desktop_actions.py`
- `assistant/workspace.py`
- `assistant/security.py`

Estado: IMPLEMENTADO / PARCIAL

Ferramentas registadas:

- `list_workspace_files`
- `read_workspace_file`
- `create_workspace_file`
- `get_presence_state`
- `get_active_window`
- `get_active_application`
- `get_open_windows`
- `get_recent_activity`
- `get_last_context_snapshot`
- `get_current_activity_summary`
- `get_raw_context_snapshot`
- `list_pending_tasks`
- `complete_task`
- `cancel_task`
- `postpone_task`
- `open_application`
- `open_url`
- `open_folder`
- `open_file`
- `open_project`

Funcional:

- workspace guard;
- path traversal blocking;
- leitura `.txt`, `.md`, `.docx`, `.pdf`;
- criação `.txt` sem overwrite;
- tarefas com SQLite;
- desktop actions com confirmação;
- Context Observer tools.

Limitações:

- desktop actions devem continuar limitadas a aliases/configuração;
- sem execução arbitrária de shell;
- browser control real não está integrado;
- pesquisa web real ainda não está implementada.

## 9. Providers De Modelo

Ficheiros:

- `assistant/llm.py`
- `assistant/model_provider.py`
- `evals/harness.py`

Estado: PARCIAL / PREPARADO

Runtime principal:

- usa `OllamaClient` directamente em `app.py`;
- endpoints: `/api/chat` e `/api/embeddings`;
- telemetria de call source/tokens;
- debug de payload;
- `summarize_text()`;
- `choose_tool()`.

Provider abstraction:

- `ModelProvider` protocol;
- `OllamaProvider`;
- `ProviderBackedLLM`;
- usado pelos evals.

Limitação corrigida na Fase 0:

- `app.py` passou a resolver o modelo por prioridade explícita: `--model`, `ECHO_MODEL_NAME`, `OLLAMA_MODEL`, `settings.json`, default.
- `settings["ollama"]["model"]` é agora usado quando não existe override.
- `model_source` é mostrado no arranque e incluído na telemetry.

Providers externos:

- Anthropic: NÃO IMPLEMENTADO;
- OpenAI: NÃO IMPLEMENTADO;
- Ruflo: não é provider de modelo no runtime.

## 10. Evals E Observabilidade

Ficheiros:

- `evals/schemas.py`
- `evals/harness.py`
- `evals/run_evals.py`
- `evals/assertions.py`
- `evals/human_review.py`
- `evals/results_store.py`
- `evals/comparisons.py`
- `evals/report.py`

Estado: IMPLEMENTADO / BOM PARA BASELINE LOCAL

Inventário:

- 51 casos JSON;
- 43 fixos;
- 8 gerados;
- categorias: `academic_status`, `conversation`, `general_knowledge`, `language`, `memory`, `real_conversation`, `search`;
- 69 ficheiros de teste unitário;
- 483 funções `test_*`.

Última baseline guardada:

- `evals/results/latest/metadata.json`;
- data: 2026-07-19;
- provider: `ollama`;
- modelo: `llama3.1:8b`;
- suite: fixed + generated;
- resultado: 51/51;
- latência média: 239 ms;
- `git_dirty=true`, portanto não é prova limpa do estado actual.

Observabilidade:

- `DEBUG_PERFORMANCE`;
- `DEBUG_OLLAMA_PAYLOAD`;
- `DEBUG_AGENT`;
- `DEBUG_CONTEXT`;
- `get_last_turn_telemetry()`;
- evals consomem telemetry estruturada, não parsing de terminal.

Limitações:

- `pytest` está funcional na `.venv` recriada;
- evals locais dependem de Ollama se executados contra modelo real;
- revisão humana existe como sinalização, não scoring preenchido;
- não há ainda comparação multi-provider real.

## 11. Workspaces Adaptativos

Estado: PREPARADO / PROTÓTIPO

Implementado:

- `UIEventAdapter`;
- eventos de pesquisa: `research_started`, `research_results_ready`, `research_unavailable`, `research_failed`, `research_completed`;
- frontend tem `researchWorkspace`.

Limitações:

- só há workspace visual de pesquisa;
- sem pesquisa web real;
- sem workspace de código, reuniões, calendário ou documentos;
- a interface ainda não é adaptativa no sentido completo definido em `docs/UI_PHILOSOPHY.md`.

## 12. Voz

Ficheiros:

- `assistant/voice_input.py`
- `assistant/audio_device.py`
- `assistant/voice_critic.py`
- `ui/main_window.py`

Estado: PARCIAL

Funcional:

- verificação de runtime de voz;
- teste de microfone;
- escolha/resolução de input device;
- gravação;
- debug audio em `data/debug/last_voice_input.wav`;
- transcrição local via Whisper;
- `language="pt"`;
- initial prompt PT;
- indicador visual na UI clássica.

Não implementado:

- wake word;
- TTS;
- ElevenLabs;
- conversa contínua;
- integração de voz na Echo OS.

Limitações:

- depende de `openai-whisper`, `ffmpeg`, `sounddevice`, `numpy`;
- qualidade depende do microfone e do modelo Whisper local;
- não deve avançar para presença contínua antes de estabilizar memória/contexto.

## 13. Pesquisa

Estado: PARCIAL

Existente:

- comandos rápidos de pesquisa Google/YouTube abrem URLs com confirmação;
- `RESEARCH_REQUEST` detecta pedidos como “pesquisa sobre Picasso”;
- eventos UI de research existem;
- workspace visual de pesquisa existe;
- se não há ferramenta real, resposta honesta.

Não existe:

- ferramenta real de web search;
- parsing/grounding de fontes;
- citações/fontes;
- cache de resultados;
- browser control ligado ao workspace de pesquisa.

Conclusão: pesquisa real ainda é NO-GO até existir ferramenta com grounding.

## 14. Segurança E Grounding

Estado: IMPLEMENTADO / EM EVOLUÇÃO

Mecanismos:

- `assistant/security.py`;
- bloqueios a shell/terminal/powershell/cmd/delete/move/outside workspace;
- `WorkspaceGuard`;
- validação de path traversal;
- fast router com validação de URL;
- desktop actions com confirmação;
- deteção de claims falsas de ferramenta/memória;
- telemetry de `tools_used`, `grounding_sources`, `unsupported_*_claim`.

Limitações:

- segurança está distribuída por `security.py`, `fast_router.py`, `agent.py`, `tools.py` e `conversation.py`;
- deve haver uma política central cada vez mais explícita antes de aumentar desktop actions/browser control;
- não há sandbox OS-level para ações externas.

## 15. Limitações Conhecidas

- `.venv` foi recriada e está funcional, mas `Activate.ps1` pode exigir ajuste temporário da Execution Policy.
- A baseline oficial nova ainda não foi criada porque há alterações por commitar.
- Documentação/resultados antigos mostram mojibake.
- Algumas respostas de baseline ainda têm português do Brasil (“em um grafo”).
- `AssistantEngine` é demasiado grande.
- Pesquisa real não está ligada.
- Echo OS é protótipo, não UI final.
- Context summaries estão vazios na base observada.
- Personal Model tem apenas 1 entrada.
- Session summaries podem conter resumos técnicos antigos.
- Provider abstraction ainda não alimenta o runtime principal.
- Voz não tem TTS/wake word/conversa contínua.

## 16. Dívida Técnica

### P0

- Criar baseline oficial em commit limpo depois de rever e commitar esta Fase 0.
- Auditar/corrigir mojibake em dados persistidos antes de usar memória como contexto produtivo.

### P1

- Limpar memória antiga contaminada.
- Garantir que respostas antigas erradas não entram como factos.
- Reduzir tamanho/complexidade de `assistant/conversation.py`.
- Centralizar política de segurança/grounding.
- Tornar pesquisa real explicitamente grounded antes de mostrar cartões.

### P2

- Separar routing determinístico em módulos menores.
- Consolidar Session Reflection, briefing e Personal Assistant Layer.
- Rever duplicação entre memória, timeline, session summaries e Personal Model.
- Documentar contratos de `UIEventAdapter`.

### P3

- Melhorar Echo OS visual.
- Criar workspaces adaptativos adicionais.
- Expandir evals de naturalidade com revisão humana.
- Preparar adapters para providers externos.

## 17. Próximos Passos Recomendados

1. Rever e commitar as alterações da Fase 0.
2. Criar baseline limpa com working tree limpa.
3. Fazer limpeza/curadoria da memória persistente.
4. Adicionar provider abstraction ao runtime, começando por manter Ollama através da mesma interface.
5. Implementar pesquisa real com grounding, só depois expandir workspace visual.
6. Escolher um workflow diferenciador e estabilizá-lo de ponta a ponta: retomar automaticamente um projeto do ponto em que ficou.

## 18. Papel Possível Do Ruflo

Conclusão: Ruflo pode ser ferramenta externa experimental, não dependência do runtime.

### A. Não Usar No Runtime Agora

- Não substituir `AssistantEngine`.
- Não substituir router.
- Não substituir memória.
- Não colocar Ruflo entre UI e Echo.
- Não introduzir swarms no fluxo normal.
- Não tornar Node uma dependência obrigatória.

### B. Possível Uso Externo

- Gerar candidatos de evals.
- Rever cobertura de testes.
- Comparar alterações entre branches.
- Analisar falhas de evals.
- Coordenar agentes de desenvolvimento.
- Produzir relatórios técnicos.

### C. Código A Reutilizar

Nenhum código deve ser copiado sem auditoria de licença. Candidatos só fazem sentido se forem pequenos, isolados e não duplicarem o Echo.

| Componente | Benefício | Sobreposição | Custo | Risco | Decisão |
|---|---|---|---|---|---|
| Geração de casos/testes | Pode ajudar evals | Baixa | Médio | Dependência externa | Explorar fora do runtime |
| Análise de falhas | Relatórios úteis | Média | Médio | Ruído/duplicação | Usar experimentalmente |
| Coordenação multi-agente | Útil para desenvolvimento | Alta no runtime | Alto | Complexidade | Não usar no runtime |

## 19. Componentes Que Não Devem Ser Substituídos

- `AssistantEngine` como núcleo actual.
- `ToolRegistry`.
- `LongTermMemory`/`PersonalModel` sem plano de migração.
- `ContextObserver`.
- `ResponseComposer`.
- `PresenceManager`.
- UI clássica enquanto Echo OS não estiver madura.
- Evals/harness existentes.

## 20. Decisões Arquiteturais Atuais

- Echo é local-first.
- Ollama é provider operacional actual.
- Providers externos devem entrar por abstraction, não por caminhos paralelos.
- Memória persistente deve ser curada, não despejada no prompt.
- Ferramentas reais exigem grounding.
- A UI adaptativa deve responder a eventos semânticos, não a texto inventado.
- Ruflo fica fora do runtime até prova em experiências isoladas.

## 21. Go / No-Go

| Decisão | Go / No-Go | Justificação | Pré-condições |
|---|---|---|---|
| Adicionar AnthropicProvider | Go condicionado | Abstraction existe nos evals, mas runtime ainda usa `OllamaClient` directamente. | Corrigir modelo em `app.py`; adaptar runtime a `ModelProvider`; garantir sem chamadas pagas por omissão. |
| Pesquisa real | No-Go agora | UI/events existem, mas falta ferramenta grounded. | Definir provider de search, fontes, citações, limites e testes. |
| Voz avançada | No-Go agora | STT existe, mas sem TTS/wake word/conversa contínua. | Estabilizar microfone/Whisper e criar testes manuais claros. |
| Ruflo no runtime | No-Go | Aumentaria complexidade e dependência Node. | Só experimentar fora do runtime. |
| Ruflo externo para evals/dev | Go experimental | Pode ajudar a gerar/rever casos sem afectar utilizador. | Isolar em scripts/experiências. |
| Próximo melhor passo | Go | Corrigir baseline/model config/memória antes de providers. | Ambiente Python funcional e baseline limpa. |

## 22. Memory Cleanup Required Before Production

Sem apagar ou migrar dados nesta tarefa, deve ser preparado um plano para:

- identificar memórias com mojibake;
- separar factos observados de respostas antigas;
- remover ou desactivar resumos técnicos antigos;
- deduplicar `memories`, `structured_facts`, `timeline_events` e `session_summaries`;
- rever confidence/source/status;
- impedir que texto bruto antigo seja usado literalmente pelo Response Composer;
- migrar dados relevantes para Personal Model com evidência e confiança;
- marcar dados incertos como hipótese, não facto.

## 23. Ficheiros E Testes Existentes

Resumo observado:

- 69 ficheiros `tests/test_*.py`;
- 483 funções `test_*`;
- 51 eval cases JSON;
- última baseline guardada: 51/51 em `ollama/llama3.1:8b`.

Validação executada neste checkpoint:

- `compileall`: passou;
- `pytest`: passou com `497 passed`;
- smoke `real_conversation`: passou com `10/10` em `ollama/gemma3:12b`;
- `evals --include-generated --mark-baseline`: adiado até existir commit autorizado e working tree limpa;
- `.venv`: recriada com Python 3.11.9 real.

## 24. Limitações Desta Auditoria

- Não foram feitas chamadas externas.
- Não foi integrado Anthropic, Ruflo, pesquisa web nova nem voz nova.
- Não foram apagados nem migrados dados em `data/`.
- A auditoria depende do estado local do repositório em 2026-07-22.
- A execução normal de `pytest` já funciona na `.venv` recriada.
