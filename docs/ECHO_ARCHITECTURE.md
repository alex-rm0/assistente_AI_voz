# Echo Architecture

Data: 2026-07-22  
Commit auditado: `3f6cca7`

Este documento descreve a arquitectura actual do Echo. Distingue caminhos determinísticos de chamadas ao LLM e marca os pontos onde existem ferramentas reais.

## A. Fluxo Principal

```mermaid
flowchart TD
    UI["UI clássica ou Echo OS"] --> Engine["AssistantEngine.respond()"]
    Engine --> Guard["Turn guard / telemetry"]
    Guard --> Fast["Fast router / confirmações"]
    Guard --> Presence["Presence Manager"]
    Guard --> Security["Security policy"]
    Guard --> Research["Research request router"]
    Guard --> SystemState["Context Observer tools"]
    Guard --> MemoryRoute["Memory write / recall / inventory"]
    Guard --> Social["Social fast path"]
    Guard --> General["General knowledge route"]
    Guard --> Cognition["Intent Engine + Executive Function"]
    Cognition --> ContextBuilder["Context Builder"]
    ContextBuilder --> Reflection["Reflection Engine"]
    Reflection --> Reasoning["Reasoning Engine"]
    Reasoning --> Composer["Response Composer"]
    General --> Composer
    Composer --> Model["OllamaClient / api/chat"]
    Cognition --> Agent["Agent / Tool planning"]
    Agent --> Tools["Tool Registry"]
    Tools --> Grounding["Observação grounded"]
    Grounding --> PublicAnswer["Resposta pública"]
    Model --> PublicAnswer
    Social --> PublicAnswer
    MemoryRoute --> PublicAnswer
    Fast --> PublicAnswer
    Research --> UIEvents["UIEventAdapter"]
    PublicAnswer --> Complete["_complete_turn()"]
    UIEvents --> Complete
    Complete --> UI
```

### Caminhos determinísticos

- Fast router;
- segurança;
- presença;
- conversa social simples;
- pesquisa quando não há ferramenta real;
- consulta/escrita estruturada de memória quando grounded;
- tarefas;
- perguntas sobre janela/aplicação via Context Observer;
- CLEAR/topic shift.

### Caminhos com LLM

- Response Composer;
- Agent tool selector quando heurísticas não resolvem;
- resumos via `OllamaClient.summarize_text()`;
- embeddings via `/api/embeddings`;
- Voice Critic quando activado por configuração/caminho.

## B. Fluxo De Ferramenta

```mermaid
flowchart TD
    Message["Mensagem do utilizador"] --> Intent["Validação de intenção de ferramenta"]
    Intent --> Safety["Security / allowed action"]
    Safety -->|bloqueado| Refuse["Recusa local"]
    Safety -->|permitido| Select["Selecção determinística ou Agent"]
    Select --> Registry["ToolRegistry.get(name)"]
    Registry --> Confirm{"Precisa de confirmação?"}
    Confirm -->|sim| Pending["Ação pendente"]
    Pending --> UserConfirm["Resposta sim/não normalizada"]
    UserConfirm --> Execute["Execução real"]
    Confirm -->|não| Execute
    Execute --> Observe["Observação / resultado"]
    Observe --> Ground["Registo de tools_used e grounding_sources"]
    Ground --> Answer["Resposta pública"]
```

Ferramentas reais actuais:

- workspace: listar, ler, criar ficheiro;
- tarefas: listar, concluir, cancelar, adiar;
- contexto: janela activa, aplicação activa, janelas abertas, actividade recente, snapshot, resumo;
- desktop actions: abrir app, URL, pasta, ficheiro, projecto com confirmação.

Ferramentas ausentes:

- web search real;
- browser control real;
- providers externos;
- TTS/wake word.

## C. Fluxo De Memória

```mermaid
flowchart TD
    Message["Mensagem"] --> Detect["Detecção: write / recall / inventory / follow-up"]
    Detect -->|write explícito| Parse["parse_memory_write_command()"]
    Detect -->|recall| Recall["is_memory_recall_question()"]
    Detect -->|inventário| Inventory["find_structured_facts()"]
    Parse --> Normalize["normalize_candidate_fields()"]
    Normalize --> SQLiteWrite["LongTermMemory.remember_structured_fact_with_trace()"]
    SQLiteWrite --> Confirm["Confirmação natural"]
    Recall --> Search["structured_facts search"]
    Search --> Retrieval["build_memory_retrieval()"]
    Retrieval --> Grounded{"Grounded?"}
    Grounded -->|sim| Verbalize["render_academic_event_answer()"]
    Grounded -->|não| Unknown["Admitir desconhecimento"]
    Inventory --> InventoryText["Resumo natural do que existe"]
    Confirm --> Complete["_complete_turn()"]
    Verbalize --> Complete
    Unknown --> Complete
    InventoryText --> Complete
```

### Notas

- Conversation memory e persistent memory estão separadas.
- Personal Model usa base própria.
- Session summaries usam base própria.
- O Response Composer não deve copiar memórias literalmente; deve interpretar, contextualizar e integrar.

## D. Fluxo Dos Evals

```mermaid
flowchart TD
    Cases["evals/cases/*.json"] --> Runner["python -m evals.run_evals"]
    Runner --> Harness["evals.harness"]
    Harness --> TempData["data temporária por caso"]
    Harness --> Engine["AssistantEngine isolado"]
    Engine --> TurnTelemetry["get_last_turn_telemetry()"]
    TurnTelemetry --> Assertions["evals.assertions"]
    Assertions --> Classifier["failure_classifier"]
    Assertions --> HumanReview["human_review heuristics"]
    Classifier --> Reports["report.json / report.csv / report.md"]
    Reports --> Results["evals/results/runs + latest"]
    Results --> Comparisons["repeat comparisons / flaky detection"]
```

### Garantias dos evals

- Não usa `data/` real.
- Cada caso tem directório temporário.
- Setup pode semear factos estruturados.
- Assertions verificam routing, ferramentas, linguagem, claims, memória e latência.

## E. Componentes Principais

| Camada | Código | Estado |
|---|---|---|
| Entrypoint | `app.py` | Implementado |
| UI clássica | `ui/main_window.py` | Implementado |
| Echo OS | `prototype_web_ui/*` | Protótipo funcional |
| Engine | `assistant/conversation.py` | Implementado, complexo |
| Agent | `assistant/agent.py` | Implementado/parcial |
| Tool registry | `assistant/tool_registry.py`, `assistant/tools.py` | Implementado |
| Cognição | `assistant/cognition/*` | Parcial |
| Response composer | `assistant/response_composer.py` | Implementado |
| Memória | `assistant/long_term_memory.py`, `assistant/memory.py` | Implementado/parcial |
| Personal Model | `assistant/personal_model.py` | Implementado inicial |
| Session Manager | `assistant/session_manager.py` | Implementado/parcial |
| Context Observer | `assistant/context_observer.py` | Implementado/parcial |
| Providers | `assistant/llm.py`, `assistant/model_provider.py` | Parcial |
| Evals | `evals/*` | Implementado |

## F. Pontos De Atenção

- `AssistantEngine` está a tornar-se o ponto de acoplamento de quase tudo.
- `ModelProvider` ainda não substitui `OllamaClient` no runtime.
- Echo OS tem eventos, mas ainda não tem workspaces maduros.
- Pesquisa visual existe antes da pesquisa real.
- Memória precisa de limpeza antes de ser usada como conhecimento confiável.
