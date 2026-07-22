# Echo Feature Matrix

Data: 2026-07-22  
Commit auditado: `3f6cca7`

| Área | Funcionalidade | Estado | Código principal | Testes | Limitações | Próximo passo |
|---|---|---|---|---|---|---|
| Conversa | Conversa normal | IMPLEMENTADO | `assistant/conversation.py`, `assistant/response_composer.py` | `tests/test_response_pipeline.py`, `tests/test_conversation_*` | Ainda há risco de respostas longas/LLM fraco | Expandir evals de naturalidade |
| Conversa | Social fast path | IMPLEMENTADO | `AssistantEngine._try_pure_social_turn()` | `tests/test_response_pipeline.py` | Cobertura depende de frases conhecidas | Afinar sem aumentar regras soltas |
| Conversa | Conhecimento geral | PARCIAL | `_try_general_knowledge_query()` | `tests/test_memory_routing_regressions.py`, evals `general_knowledge` | Depende do modelo local; sem fontes | Separar de pesquisa real |
| Routing | Fast router | IMPLEMENTADO | `assistant/fast_router.py` | `tests/test_fast_router.py`, `tests/test_fast_router_integration.py` | Só cobre comandos simples | Manter antes de LLM/embeddings |
| Routing | Executive Function | PARCIAL | `assistant/cognition/executive_function.py` | `tests/test_cognition_executive_function.py` | Não controla todo o pipeline ainda | Dar-lhe mais autoridade modular |
| Routing | Intent Engine | PARCIAL | `assistant/cognition/intent_engine.py` | `tests/test_cognition_intent_engine.py` | Heurístico | Cobrir mais intenções reais |
| Cognição | Cognitive loop | PARCIAL | `context_builder.py`, `reflection_engine.py`, `reasoning_engine.py` | `tests/test_cognition_integration.py` | Nem sempre é chamado; maturidade inicial | Usar só quando Executive Function decide |
| Memória | Short-term memory | IMPLEMENTADO | `assistant/memory.py` | `tests/test_*memory*` | JSON simples | OK por agora |
| Memória | Long-term memory SQLite | IMPLEMENTADO/PARCIAL | `assistant/long_term_memory.py` | `tests/test_long_term_memory.py` | Dados antigos contaminados/mojibake | Plano de limpeza |
| Memória | Recall | IMPLEMENTADO | `assistant/memory_recall.py`, `_try_memory_recall_question()` | `tests/test_memory_recall.py`, evals `memory` | Focado em academic/task facts | Expandir domínios com cautela |
| Memória | Escrita explícita | IMPLEMENTADO | `parse_memory_write_command()`, `PersonalModel.remember_explicit()` | `tests/test_memory_routing_and_write.py` | Heurísticas de extracção | Melhorar confirmação e categorias |
| Memória | Escrita passiva | PARCIAL | `_maybe_extract_structured_memory()` | regressions/evals | Risco de falsos positivos | Reduzir e exigir evidência |
| Personal Model | Modelo pessoal | IMPLEMENTADO INICIAL | `assistant/personal_model.py` | `tests/test_personal_model.py` | Só 1 entrada real observada | Construir Preference/User Model gradual |
| Sessão | Session Manager | IMPLEMENTADO/PARCIAL | `assistant/session_manager.py` | `tests/test_session_manager.py` | 66 resumos, potencial ruído antigo | Curadoria e qualidade de summaries |
| Sessão | Session Reflection | PARCIAL | `assistant/session_reflection.py` | `tests/test_session_reflection.py` | Pode inferir pouco | Melhorar factos vs hipóteses |
| Tarefas | Criar tarefas | IMPLEMENTADO/PARCIAL | `LongTermMemory.create_task()` | `tests/test_tasks.py`, `tests/test_tasks_integration.py` | Linguagem natural limitada | Parser mais robusto |
| Tarefas | Concluir tarefas | IMPLEMENTADO | `complete_task`, `LongTermMemory.complete_task()` | `tests/test_task_tools.py` | Ambiguidade quando há várias | UX de escolha |
| Tarefas | Cancelar tarefas | IMPLEMENTADO | `cancel_task` | `tests/test_task_tools.py` | Igual ao anterior | UX de escolha |
| Tarefas | Adiar tarefas | IMPLEMENTADO/PARCIAL | `postpone_task` | `tests/test_task_tools.py` | Datas naturais limitadas | Parser temporal |
| Tarefas | Painel expansível | IMPLEMENTADO | `ui/main_window.py`, `task_formatter.py` | `tests/test_task_formatter.py` | Só UI clássica | Portar para Echo OS |
| Contexto | Context Observer | IMPLEMENTADO/PARCIAL | `assistant/context_observer.py` | `tests/test_context_observer.py` | Poucos snapshots reais; Windows only | Melhorar persistência e summaries |
| Contexto | Janela activa | IMPLEMENTADO | `get_active_window` | `tests/test_context_observer_integration.py` | Depende de Win32 | Validar em uso real |
| Contexto | Aplicação activa | IMPLEMENTADO | `get_active_application` | idem | Depende de processo/janela | OK |
| Contexto | Janelas abertas | IMPLEMENTADO | `get_open_windows`, Win32 enum | idem | Filtragem de ruído | Afinar blacklist |
| Contexto | VS Code sessions | PARCIAL | `ContextObserver` | testes context observer | Detecção heurística | Validar em cenários reais |
| Contexto | Git repositories | PARCIAL | `ContextObserver` | testes context observer | Heurístico | Melhorar confiança |
| Contexto | Ficheiros modificados | IMPLEMENTADO/PARCIAL | `ContextObserver` | testes context observer | Pode misturar antigo/actual | Separar fontes no resumo |
| Workspace | Listar ficheiros | IMPLEMENTADO | `list_workspace_files` | `tests/test_tools.py` | Apenas workspace | OK |
| Workspace | Ler ficheiros | IMPLEMENTADO | `read_workspace_file`, `document_reader.py` | `tests/test_tools.py` | `.txt/.md/.docx/.pdf`; sem OCR | OK |
| Workspace | Criar ficheiros | IMPLEMENTADO | `create_workspace_file` | `tests/test_tools.py` | Só `.txt`, sem overwrite | Confirmação futura |
| Desktop | Abrir apps | IMPLEMENTADO/PARCIAL | `desktop_actions.py`, `tools.py` | `tests/test_desktop_actions.py` | Lista permitida; confirmação | Melhorar foco em app existente |
| Desktop | Abrir URL | IMPLEMENTADO | `open_url`, `fast_router.py` | fast router tests | Só http/https | OK |
| Desktop | Abrir pasta/ficheiro | IMPLEMENTADO/PARCIAL | `open_folder`, `open_file` | desktop tests | Safe roots | Expandir config |
| Desktop | Abrir projecto | IMPLEMENTADO/PARCIAL | `open_project`, settings known_projects | desktop tests | Só projectos conhecidos | Integrar Session Manager |
| Pesquisa | URL Google/YouTube | IMPLEMENTADO | `fast_router.py` | fast router tests | Abre URL, não pesquisa interna | OK |
| Pesquisa | Research request | PARCIAL | `_try_research_request()`, `UIEventAdapter` | `tests/test_research_routing.py` | Sem ferramenta web real | Implementar search grounded |
| Pesquisa | Research workspace | PROTÓTIPO | `prototype_web_ui/web/echo_ui.js` | `tests/test_web_ui_prototype.py` | Visual sem dados reais | Ligar a search real |
| UI | UI clássica | IMPLEMENTADO | `ui/main_window.py` | testes indirectos | Utilitária | Manter como fallback |
| UI | Echo OS | PROTÓTIPO FUNCIONAL | `prototype_web_ui/*` | `tests/test_web_ui_prototype.py` | Sem UI final completa | Evoluir por eventos |
| Voz | STT local | PARCIAL | `voice_input.py` | `tests/test_voice_input.py` | Depende de Whisper/ffmpeg/mic | Estabilizar antes de contínuo |
| Voz | Microfone | PARCIAL | `audio_device.py` | `tests/test_audio_device.py` | Hardware variável | Testes manuais |
| Voz | TTS | NÃO IMPLEMENTADO | - | - | - | Adiar |
| Voz | Wake word | NÃO IMPLEMENTADO | - | - | - | Adiar |
| Modelos | Modelos locais | IMPLEMENTADO | `assistant/llm.py` | `tests/test_model_provider.py` | Ollama only no runtime | Corrigir settings model |
| Modelos | Provider abstraction | PREPARADO/PARCIAL | `assistant/model_provider.py` | `tests/test_model_provider.py` | Só evals usam | Integrar runtime depois |
| Modelos | Providers externos | NÃO IMPLEMENTADO | - | - | Sem Anthropic/OpenAI | Fase 1 futura |
| Evals | Harness | IMPLEMENTADO | `evals/harness.py` | `tests/test_evals_*` | Ollama only | Adicionar providers |
| Evals | Casos fixos/gerados | IMPLEMENTADO | `evals/cases` | 51 casos | Sem scoring humano preenchido | Expandir real conversations |
| Evals | Repeat/flaky | IMPLEMENTADO | `evals/comparisons.py` | `tests/test_evals_results_store.py` | Poucos runs | Usar baseline limpa |
| Observabilidade | Turn telemetry | IMPLEMENTADO | `get_last_turn_telemetry()` | evals | Debug muito verboso se flags true | Separar dev/prod config |
| Segurança | Grounding claims | IMPLEMENTADO/PARCIAL | `conversation.py`, `memory_recall.py` | evals/assertions | Heurístico | Política central |
| Segurança | Workspace guard | IMPLEMENTADO | `workspace.py`, `tools.py` | `tests/test_tools.py` | Só cobre workspace/file tools | OK |
| Ruflo | Runtime | NÃO IMPLEMENTADO | - | - | Não deve entrar agora | Manter externo |
| Ruflo | Ferramenta dev/evals | PREPARADO CONCEPTUAL | `docs/ruflo_experiment_plan.md` | - | Sem integração | Experimentar isolado |
