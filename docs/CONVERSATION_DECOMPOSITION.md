# Decomposição do conversation.py

Data: 2026-11-12. Referência: `docs/ECHO_NORTH_STAR.md` §10.1. Só análise: nenhum código foi alterado.

Os números de linha referem-se a `assistant/conversation.py` no commit `b2a5d55` (15 366 linhas).
Os intervalos, tamanhos e dependências foram obtidos com `ast` sobre o ficheiro; a classificação por
responsabilidade, o risco e a ordem são julgamento meu. Os intervalos de funções de módulo incluem
as constantes (tabelas de frases, regex) que ficam entre elas.

## 0. Forma do ficheiro

| Bloco | Linhas | Conteúdo |
|---|---|---|
| Imports | 1–178 | 40 módulos de `assistant.*` |
| Dataclasses de estado | 180–381 | `PendingWritingTask`, `ActiveTaskContext`, `ActivityInterpretation`, `DocumentIntent`, `ActiveDocumentContext`, `WorkSession` |
| `AssistantEngine` | 419–9380 | 223 métodos, ≈8 960 linhas |
| Funções de módulo | 9383–15366 | 271 funções (≈4 280 linhas de código) mais constantes e 5 dataclasses pequenas |

Três factos que condicionam tudo o resto:

- `_respond_inner` (822–1729, 908 linhas) é uma cadeia de 58 tentativas `_try_*` por ordem fixa.
  A ordem é comportamento.
- Todas as respostas saem por `_complete_turn` (6821–6867), que tem 68 pontos de chamada.
- Dois objetos são tocados por quase tudo: o dicionário `_turn_trace` (65 métodos) e o
  `_work_session` (50 métodos).

## 1. Responsabilidades

### 1.1 Dentro de `AssistantEngine`

| # | Responsabilidade | Linhas | ≈Tamanho | Funções principais |
|---|---|---|---|---|
| A | Construção e configuração | 422–666 | 245 | `__init__` (188 linhas, 36 parâmetros), propriedades de compatibilidade do `WorkSession`, `set_profile` |
| B | Pedido em curso: cancelamento e progresso | 711–752 | 42 | `begin_request`, `cancel_current_request`, `is_cancel_requested`, `set_progress_listener`, `_emit_progress` |
| C | Entrada do turno e erros | 754–820 | 67 | `respond`, `_handle_unexpected_error`, `_handle_provider_configuration_error` |
| D | Cadeia de routing | 822–1729 | 908 | `_respond_inner` |
| E | Idioma | 1731–1812, 9126–9195 | 150 | `_load_response_language_preference`, `_set_response_language`, `_try_language_preference_command`, `_try_translate_previous_response` |
| F | Sessão e histórico | 1824–1856, 7932–8017, 9029–9124 | 250 | `clear_conversation`, `_reset_transient_context`, `_recent_conversation_history`, `_remember_pair`, `close_session`, `_finish_session`, `_remember_session_summary`, `_sync_session_with_presence`, `_try_briefing_question` |
| G | Tarefas "to-do" (memória longa) e painel | 1858–1885, 8188–8197, 8904–8963 | 100 | `pending_tasks_summary`, `_try_task_command`, `_run_task_tool` |
| H | Atalhos determinísticos e sociais | 668–709, 1905–2027, 6584–6627, 6712–6784 | 330 | `_try_system_datetime_command`, `_try_memory_inventory`, `_try_general_knowledge_query`, `_try_fast_route`, `_try_pure_social_turn`, `_try_conversational_refinement` |
| I | Escrita (email, rascunhos) | 2029–2194, 5383–5471 | 255 | `_try_writing_workflow`, `_compose_pending_writing_draft`, `_ensure_pending_writing_artifact`, `_try_writing_draft_followup` |
| J | Eventos para a UI | 2205–2252 | 45 | `consume_ui_events`, `_queue_ui_event`, `_queue_semantic_object_sync` |
| K | Pesquisa e operações sobre conteúdo | 2254–2637 | 385 | `_try_research_request` (232), `_try_content_operation`, `_try_open_research_source` |
| L | Estado do sistema e contexto observado | 2639–2741 | 105 | `_try_system_state_tool_query`, `_try_system_context_status`, `_try_application_activity_query` |
| M | Comandos de workspace, atenção, capacidade | 2743–3382, 4034–4505 | 1 110 | `_try_compound_workspace_command`, `_try_workspace_command` (217), `_try_attention_command`, `_try_pending_workspace_action`, `report_workspace_capacity` |
| N | Tasks de atividade | 3391–4017 | 630 | `_try_active_task_turn`, `_process_task_turn` (194), `_execute_active_task` (143), `_update_task_artifact_ui`, `_pause_active_task`, `_resume_task_from_stack` |
| O | Representação e comparações | 4507–5290 | 785 | `_try_representation_command` (177), `_try_mixed_representation_intent`, `_ensure_activity_for_explicit_alternatives` |
| P | Exportação | 5292–5381 | 90 | `_try_artifact_export`, `_export_document_content` |
| Q | Documentos | 5473–6565 | 1 095 | `_try_document_task`, `_try_active_document_followup` (540), `_run_document_task_with_file`, `_record_document_trace` |
| R | Estado de voz e TTS | 1794–1803, 6567–6582, 6629–6710 | 110 | `_test_microphone_now`, `_try_voice_question`, `_tts_status_summary`, `_try_tts_command`, `_coordinate_tts_voice_for_language` |
| S | Presence | 659–666, 6786–6819, 7076–7270, 9008–9027 | 260 | `set_presence_state`, `_try_presence_mode_command`, `_compute_presence_decision_shadow`, `_presence_question_discipline_instruction`, `_try_vague_action_clarification` |
| T | Finalização da resposta | 6821–7074 | 254 | `_complete_turn`, `_finalize_response` (206) |
| U | Trace e telemetria do turno | 7272–7930 | 660 | `_begin_turn_trace`, `_run_shadow_interpreter`, `_end_turn_trace` (502), `_print_turn_trace`, `get_last_turn_telemetry` |
| V | Cognição e contexto para o agente | 8019–8247 | 230 | `_run_cognitive_loop`, `_context_for_agent`, `_gate_memory_relevance`, `_observed_context_summary` |
| W | Memória e modelo pessoal | 8249–8902 | 655 | `_maybe_extract_structured_memory`, `_try_memory_write_command`, `_try_memory_recall_question`, `_try_project_history_recall`, `_answer_from_personal_model` |
| X | Perguntas sobre contexto, perfil e a própria conversa | 8965–9006, 9197–9371 | 220 | `_try_context_question`, `_try_profile_memory`, `_try_conversation_memory_question`, `_system_prompt_with_tools` |

### 1.2 Funções de módulo (9383–15366)

| Linhas | Serve | Exemplos |
|---|---|---|
| 9383–9665 | H, T | `is_pure_social_turn`, `_deterministic_response_cleanup`, `_remove_extra_questions`, `_block_invented_echo_first_person` |
| 9667–9886 | U | `_log_language_audit`, `_execution_path_for_turn`, `_model_routing_telemetry`, `_llm_call_details` |
| 9916–10003 | H | `_parse_system_datetime_question`, `_format_pt_date` |
| 10027–11232 | Q, I | `_parse_document_task_request`, `_workspace_search`, `_validate_rewrite_draft`, `_document_rewrite_prompt` |
| 11238–11822 | M, O | `_entity_workspace_command`, `_workspace_command_actions`, `resolve_semantic_collection_targets` |
| 11822–12011 | Q | `_active_document_followup_action`, `_build_active_document_context` |
| 12070–12235 | T | `_detect_unsupported_tool_claim`, `_detect_ungrounded_memory_claim`, `_detect_ungrounded_activity_claim`, `_detect_ungrounded_workspace_operation_claim` |
| 12258–12990 | vários | ≈70 detetores `_asks_*` / `_looks_like_*` consumidos por F, G, H, W, X |
| 12992–13469 | K, L, I, W | `_extract_research_query`, `_is_topic_shift_request`, `_application_activity_from_snapshot` |
| 13489–13512 | V | `_memory_relevance_score` |
| 13608–14363 | N | `_interpret_activity`, `_detect_new_task_intent`, `_detect_task_control_action`, `_extract_generic_activity_fields` |
| 14383–14628 | M | `resolve_workspace_reference`, `resolve_semantic_targets` |
| 14653–15164 | N, I | `_extract_compose_email_fields`, `_compose_email_task_prompt`, `_validate_task_generation`, `_update_pending_writing_task` |
| 15164–15366 | W, H | `_project_recall_context`, `_intent_instruction_for_user_message` |

## 2. As seis responsabilidades que bloqueiam a F1

"Entradas" = quem chama este código. "Saídas" = o que este código chama. "Estado" = atributos do
engine em que toca (E = escreve, L = só lê).

### 2.1 Ciclo de vida da sessão (F)

- **Código**: cabeça de `_respond_inner` (826–845: fim por inatividade de 2 h, novo segmento,
  `start_session`), `close_session`, `_finish_session`, `_remember_session_summary`,
  `_sync_session_with_presence` (7978–8017), `clear_conversation`, `_reset_transient_context`,
  `_try_topic_shift`, `startup_greeting`, `session_summary_has_content` (módulo).
- **Entradas**: `app.py` (`close_session` ×2, `clear_conversation`, `clear_history`,
  `startup_greeting`, `history`); `__init__` (inicia sessão se a Presence estiver ativa);
  `_respond_inner`; S (`set_presence_state`, `_try_presence_mode_command`);
  `_recent_conversation_history` é chamado por 11 métodos de outras responsabilidades.
- **Saídas**: `SessionManager` (`start_session`, `end_if_inactive`, `end_session`),
  `LongTermMemory.remember_context_summary`, `ConversationMemory` (`load`, `clear`, gravação do par),
  `ContextObserver` (passado ao `SessionManager`), `_queue_ui_event` (J).
- **Estado**: E `_conversation_segment_start`. `_reset_transient_context` escreve 14 atributos que
  pertencem a outras responsabilidades (`_active_operation_*` de K, `_active_memory_*` de W,
  `_pending_*`, `active_contexts`/`last_cognitive_*` de V) e quatro campos do `_work_session`.
  L `presence`, `session_manager`, `memory`, `long_term_memory`, `context_observer`, `_turn_trace`.
- **Nota**: o estado da sessão de trabalho (`WorkSession`) vive só em memória e morre com o
  processo; o `SessionManager` persiste apenas o resumo.

### 2.2 Orquestração de tasks (N, com I)

- **Código**: N (3391–4017) e a retoma (`_try_resume_activity_reference`, `_find_task_by_id`,
  3252–3324); escrita I; funções de módulo 13608–14363 e 14653–15164 (≈42 funções usadas).
  No total ≈935 linhas de métodos e ≈1 300 de funções de módulo.
- **Entradas**: `_respond_inner` (`_try_writing_draft_followup`, `_try_writing_workflow`,
  `_try_task_intelligence_query`, `_try_active_task_turn`, `_try_pending_writing_workflow`);
  M, O e K chamam `_find_task_by_id` (6 métodos), `_touch_task` (4), `_update_task_artifact_ui` (3).
- **Saídas**: `llm` e `response_composer` (geração da task), `task_intelligence`
  (`classify_task_query`, `answer_task_query`, `update_task_knowledge`, `task_dialogue_response`),
  `representation`/`attention` (`register_object`, `set_bulk_attention`), J (`_queue_ui_event`,
  `_queue_semantic_object_sync`), O (`_materialize_comparison_options_from_context`,
  `_maybe_prepare_task_surface_capacity_action`), Q (`_create_workspace_txt`).
- **Estado**: E `_active_task_id_counter`, `_task_execution_requested`, `_last_fast_tools_used`;
  E através das propriedades `_work_session.active_task`, `.paused_task_stack`, `.writing_context`,
  e ainda `.objects`, `.attention`, `.export_focus`. L `_turn_trace`, `memory`, `presence`,
  `now_provider`, `workspace_path`, `response_language`.
- **Nota**: há duas coisas chamadas "task". Esta é a task de atividade (`ActiveTaskContext`, em
  memória). As tarefas "to-do" (G) vivem na memória longa e passam por `assistant.tools`.

### 2.3 Orquestração de ferramentas e ações (K, L, parte de H e V)

- **Código**: chamada ao `Agent` (1672–1729), `_try_fast_route` (1995–2027), L (2639–2741),
  K (2254–2637), `_try_delegation` (1814–1822), G (`_run_task_tool`), contexto para o agente
  (8134–8247), `_remember_tools_used`, `_record_tool_intent_check`, `_record_agent_debug_trace`,
  `_system_prompt_with_tools`, `_tools_summary`. ≈730 linhas de métodos, 23 funções de módulo.
- **Entradas**: só `_respond_inner`. Nenhuma outra responsabilidade chama estes métodos, exceto
  `_context_for_agent` e `_system_prompt_for_agent`.
- **Saídas**: `Agent.run` / `has_pending_confirmation`, `ToolRegistry`, `fast_router`,
  `DelegationManager`, `task_research` (`run_task_research`, `build_research_request`),
  `addressable_content`, `assistant.tools` (tarefas), `ContextObserver`, `SessionManager`,
  `LongTermMemory`, `PersonalModel`; internamente J, N (`_update_task_artifact_ui`),
  F (`clear_conversation`), R (`_test_microphone_now`), V.
- **Estado**: E `_active_operation_type/_topic/_id/_ttl`, `_last_fast_tools_used`. L `agent`,
  `tools`, `quick_sites`, `desktop_config`, `delegation`, `presence` (permissões:
  `can_use_tools`, `can_ask_confirmation`), `active_contexts`, `_pending_document_task`,
  `_work_session`, `_turn_trace`.
- **Nota**: a confirmação pendente de uma ação vive dentro do `Agent`, não no engine. O engine só
  lê `has_pending_confirmation()` para marcar a resposta.

### 2.4 Presence (S)

- **Código**: 659–666, 6786–6819, 7076–7270, 9008–9027. ≈260 linhas, uma função de módulo.
- **Entradas**: `app.py` (`set_presence_state`, `presence_state`); `_respond_inner`
  (`_try_presence_mode_command`, `_presence_silent_response`, `_try_vague_action_clarification`,
  `_try_presence_question`, `_presence_question_discipline_instruction`); T chama
  `_compute_presence_decision_shadow`.
- **Saídas**: `PresenceManager`, `presence_decision.compute_presence_decision`,
  `LongTermMemory.remember_timeline_event`, `PersonalModel`; internamente F
  (`_sync_session_with_presence`, `_recent_conversation_history`) e N (`_active_task_context`).
- **Estado**: E `_presence_turns_since_address`; escreve chaves no `_turn_trace`. L `presence`,
  `personal_model`, `long_term_memory`, `response_language`.
- **Nota**: o objeto `presence` é lido por 21 métodos fora deste grupo (permissões de memória e
  ferramentas). Isso é consumo normal de uma dependência, não lógica de Presence espalhada.

### 2.5 Ciclo de vida da resposta (C, T, U)

- **Código**: C (754–820), T (6821–7074), U (7272–7930); funções de módulo 9546–9886 e
  12070–12235. ≈970 linhas de métodos, 28 funções de módulo.
- **Entradas**: `app.py` e o controller chamam `respond` e `get_last_turn_telemetry`;
  `_complete_turn` é chamado em 68 sítios; `_route_is_authorized` por 3 métodos.
- **Saídas**: `interaction_interpreter` (`interpret_shadow`, decisão de rota), `response_language`
  (`looks_mixed_for_language`, `localize_deterministic_response`), `conversation_guidance`,
  `voice_critic` (`detect_semantic_conflict`, `detect_subject_swap`), `memory_recall`
  (`detect_unsupported_memory_claim`), `response_composer`; internamente S, F (`_remember_pair`),
  W (`_record_memory_grounding`, `_update_pending_user_intent`).
- **Estado**: E `_turn_trace`, `_last_turn_telemetry`, `_last_response_blocked`,
  `_last_selected_path`, `_canonical_interpretation`, `_routing_decision`; E
  `_work_session.export_focus` e `.prepared_output` (em `_complete_turn`). L `llm` (contadores de
  chamadas), `_work_session`, `active_contexts`, `_last_workspace_command`, flags de debug.
- **Nota**: `_finalize_response` é onde vivem os guards anti-invenção (invariante §1.2).
  `_end_turn_trace` (502 linhas) lê estado de quase todas as outras responsabilidades para montar
  a telemetria de que os evals dependem.

### 2.6 Fronteira de coordenação voz/Core (B, J, R)

- **Código**: B (711–752), J (2205–2252), `report_workspace_capacity` (4474–4505), R. ≈215 linhas.
- **Entradas**: `prototype_web_ui/controller.py` chega ao engine por `responder.__self__` e
  `getattr`: `set_progress_listener` (l. 55), `cancel_current_request` (l. 1360),
  `report_workspace_capacity` (l. 1435), `consume_ui_events` (l. 2119) e `llm` para o
  warmup (l. 454, 477). Dentro do engine, `_queue_ui_event` é chamado por 23 métodos (49 sítios),
  `_queue_semantic_object_sync` por 9; `is_cancel_requested` e `_emit_progress` só são usados na
  reescrita de documentos (5930–6121).
- **Saídas**: `UIEventAdapter.serialize`, `voice_input` (`check_microphone`,
  `play_last_voice_input`, `voice_status_report`), `tts_manager` (`set_active_voice`, estado).
  Nenhum método deste grupo chama outros métodos do engine.
- **Estado**: E `_cancel_event` (com `_cancel_lock`), `_progress_listener`, `_pending_ui_events`,
  `_workspace_capacity_revision`, `_work_session.capacity`. L 13 atributos `voice_*` recebidos no
  construtor, `tts_manager`.
- **Notas**:
  - O pipeline de voz (STT, VAD, barge-in, wake word) vive no controller. O engine recebe texto e
    devolve texto; a fronteira no `conversation.py` é só cancelamento, progresso, eventos e estado.
  - Não existe contrato: o controller descobre os métodos por nome em tempo de execução.
  - `respond` corre numa thread de trabalho. Só o cancelamento tem lock; `_pending_ui_events` e
    `_work_session.capacity` não têm. O controller pode limpar a conversa e reportar capacidade a
    meio de um turno: ver KNOWN_ISSUES §13.

## 3. Ordem de extração

Regras para todos os passos:

- Um passo, um commit. Antes e depois: `pytest` completo e `scripts/battery_driver.py` (cenários
  1–3 e 11–14). Passos marcados com ⚙ tocam em threads ou no ciclo de vida do controller e pedem
  também `scripts/flaky_check.py --runs 40`.
- O engine mantém métodos finos com o mesmo nome que delegam no módulo novo. Os 68 `_complete_turn`,
  os 49 `_queue_ui_event` e os acessos do controller por `getattr` não mudam.
- Os nomes que os testes importam de `assistant.conversation` continuam importáveis de lá (58
  ficheiros de teste importam o módulo; além de `AssistantEngine`, importam `ActiveTaskContext`,
  `WorkSession`, `_active_document_followup_action`, `_validate_rewrite_draft`,
  `_detect_unsupported_tool_claim`, entre outros).
- Mover código sem o alterar. Qualquer mudança de comportamento fica para um commit à parte.

"Move" = linhas que saem do `conversation.py`. O ficheiro encolhe um pouco menos, porque ficam os
métodos que delegam.

| # | Passo | Módulo novo | Move | Risco | Porquê nesta posição |
|---|---|---|---|---|---|
| 1 | Dataclasses de estado (180–381) | `assistant/work_session.py` | ≈200 | Baixo | Sem lógica; todos os passos seguintes precisam de importar estes tipos sem importar o engine |
| 2 ⚙ | Cancelamento e progresso (B) | `assistant/request_control.py` | ≈40 | Baixo | Não chama nada; 3 atributos próprios; primeiro pedaço da fronteira voz/Core |
| 3 ⚙ | Fila de eventos da UI (J) | `assistant/ui_event_outbox.py` | ≈45 | Baixo | Não chama nada; um atributo próprio. Bom momento para decidir o lock |
| 4 | Funções de telemetria (9667–9886) | `assistant/turn_telemetry.py` | ≈220 | Baixo | Funções puras, um só consumidor (`_end_turn_trace`) |
| 5 | Presence (S) | `assistant/presence_turn.py` | ≈250 | Baixo–médio | Escreve um só atributo; 4 chamadas para fora. Tem de vir antes do passo 8, que a consome |
| 6 ⚙ | Cola do `SessionManager`: inatividade, fecho, resumo, sincronização com a Presence | `assistant/session_lifecycle.py` | ≈90 | Médio | Pequeno, mas `close_session` corre no fecho da app (cenário 14, hoje PARCIAL) |
| 7 | Montagem e impressão do trace (`_end_turn_trace`, `_print_turn_trace`) | `assistant/turn_telemetry.py` | ≈560 | Médio | Grande mas só de leitura; os evals validam o resultado. `_turn_trace` continua a ser um dicionário do engine |
| 8 | Guards de finalização (`_finalize_response`, 9546–9665, 12070–12235) | `assistant/response_guard.py` | ≈500 | Médio–alto | É o invariante §1.2. Sai como função que recebe texto e contexto e devolve texto, `blocked` e sinais |
| 9 | Entrada e fecho do turno (`respond`, erros, `_complete_turn`, `_begin_turn_trace`) | `assistant/response_lifecycle.py` | ≈170 | Médio | Depois de 5, 7 e 8 só resta a sequência; `export_focus` passa a ser pedido ao dono do `WorkSession` |
| 10 | Contexto para o agente (8134–8247) | `assistant/agent_context.py` | ≈115 | Baixo–médio | Só leitura de memória, sessão e observador |
| 11 | Ferramentas diretas: fast route, estado do sistema, atividade de aplicações, delegação, chamada ao `Agent` | `assistant/tool_orchestration.py` | ≈330 | Médio | Só o `_respond_inner` os chama; o cenário 13 da bateria cobre o caminho de ferramenta |
| 12 | Pesquisa e operações sobre conteúdo (K) | `assistant/tool_orchestration.py` | ≈420 | Médio–alto | Escreve `_active_operation_*` e emite eventos; depende do passo 3 |
| 13 | Funções de task de atividade (13608–14363) | `assistant/task_activity.py` | ≈750 | Médio | Funções sobre `ActiveTaskContext`; depende do passo 1 |
| 14 | Funções de email e escrita (14653–15164) | `assistant/writing_tasks.py` | ≈510 | Médio | Idem; correções recentes nesta zona (`a69b636`, `f53ae7f`) têm testes |
| 15 | Orquestrador de tasks (N e I) | `assistant/task_orchestrator.py` | ≈890 | Alto | O mais acoplado: `WorkSession`, LLM, eventos, O e Q. Só com 1, 3, 13 e 14 feitos; o cenário 11 cobre o caso base |
| 16 | Reset de contexto transitório e novo segmento (`_reset_transient_context`, `clear_conversation`, `_try_topic_shift`) | `assistant/session_lifecycle.py` | ≈50 | Médio | Só quando cada atributo tiver dono, para o reset chamar os donos em vez de escrever 14 campos alheios |

Total movido: ≈5 100 linhas, cerca de um terço do ficheiro. Com os passos 1–16 feitos, as seis
responsabilidades do §10.1 têm módulo próprio.

Pontos de paragem naturais: depois do 4 (fronteira voz/Core e estado isolados, risco baixo),
depois do 9 (ciclo de vida da resposta fechado), depois do 12, depois do 16.

## 4. O que não extrair ainda

- **`_respond_inner` (D, 908 linhas).** A ordem das tentativas é o comportamento e cada uma devolve
  por `_complete_turn`. O interpretador canónico já corre em sombra (`_run_shadow_interpreter`,
  `_routing_decision`, `_route_is_authorized`): movê-la agora seria fixar uma cadeia que está a ser
  substituída. Encolhe sozinha à medida que os `_try_*` saem com os seus donos.
- **Documentos (Q, ≈1 100 linhas de métodos e ≈1 400 de funções).** Não está na lista do §10.1,
  inclui o método mais longo depois do routing (`_try_active_document_followup`, 540 linhas) e é o
  único consumidor de cancelamento e progresso. Risco alto para um ganho que a F1 não exige.
- **Workspace, atenção, representação e comparações (M, O, ≈1 900 linhas).** Falam com a UI só por
  eventos, que o passo 3 já isola. O resto é lógica de produto com muitos casos de fronteira e não
  atravessa a fronteira Core/UI.
- **Memória e modelo pessoal (W, V).** A R3 vai mexer aqui (perfil-núcleo, contexto por relevância).
  Extrair antes obriga a mover duas vezes.
- **Idioma (E).** Outro agente está a alterar `response_language.py`; qualquer extração agora colide.
- **Estado de voz e TTS (R) e os 13 parâmetros `voice_*` do construtor.** A R2 ainda está em curso
  em `voice_input.py`, `stt_backend.py` e `controller.py`. Da fronteira voz/Core, só os passos 2 e
  3 avançam já; o resto espera pelo fecho da R2 e pelo contrato da API do Core (F0).
- **Detetores `_asks_*` / `_looks_like_*` (12258–12990, ≈70 funções).** São baratos de mover, mas
  servem cinco responsabilidades. Cada um deve sair com quem o usa, não para um módulo "utils".
- **Um contrato formal engine↔controller** (substituir o `responder.__self__` + `getattr`). É
  trabalho da F0 e toca em `controller.py`, que está reservado.
- **`__init__` (A).** Os 36 parâmetros reduzem-se por consequência dos passos acima; refazê-lo
  primeiro mexe em `app.py`, no harness de evals e em dezenas de fixtures de teste de uma só vez.
