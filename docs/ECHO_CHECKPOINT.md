# Echo Checkpoint

Data: 2026-07-22  
Commit auditado: `3f6cca7`  
Estado Git no início da auditoria: limpo (`git status --short` sem alterações)

Este checkpoint descreve o estado técnico real do Echo antes de adicionar novos providers ou funcionalidades. A análise foi feita a partir do código presente no repositório, não a partir de intenções anteriores.

## Atualização — Checkpoint Técnico E Funcional Completo (Routing Automático, Documentos, Estados Visuais)

Atualizado em 2026-08-07. HEAD atual: `72ddd5b`. Esta secção substitui, em termos de estado atual, as afirmações mais antigas deste ficheiro sobre routing/providers/UI — as secções seguintes (1 a 24) descrevem o estado em 2026-07-22/08-06 e ficam preservadas como histórico, mas partes delas (por exemplo "Anthropic: NÃO IMPLEMENTADO" na secção 9) já não são verdade. Esta secção é a fonte de verdade mais recente.

### 1. Estado Geral Do Projeto

O Echo é uma aplicação local. Execução principal:

```
python app.py --ui echo-os
```

(`--ui classic` continua disponível para a UI PySide6 clássica; `app.py` também aceita `--provider {ollama,anthropic}`, `--model` e `--model-mode {local,claude,automatic}` como overrides pontuais de sessão.)

Stack:

- Python
- PySide6 (UI clássica)
- QWebEngineView / QWebChannel (Echo OS)
- HTML/CSS/JavaScript (frontend do Echo OS)
- Ollama (provider local, operacional)
- Anthropic (provider opcional, condicional — ver critérios abaixo)

Modelo local atual (`config/settings.json` → `model.name` e `ollama.model`): `llama3.1:8b`.

`config/settings.json` → `model_routing.mode` está gravado como `"local"` neste momento (ficheiro não tocado nesta tarefa). O modo `automatic` — o modo em que o router decide por tarefa e que todo o trabalho recente de escalada documental pressupõe — é o modo usado nas sessões de desenvolvimento/teste recentes, selecionado via `--model-mode automatic` ou `ECHO_MODEL_MODE=automatic`; `resolve_model_routing_config()` resolve por prioridade CLI > env > `settings.json` > default (`"local"`).

Anthropic só é usado em modo `automatic` quando, simultaneamente:

- `automatic_claude_enabled=true` (`model_routing.automatic.claude_enabled` em `settings.json`, atualmente `true`);
- API key configurada (`ANTHROPIC_API_KEY`);
- `ECHO_ALLOW_PAID_MODEL_CALLS=true` (nome real da env var: ver `PAID_CALL_CONFIRMATION_ENV` em `assistant/anthropic_provider.py`);
- orçamento disponível (`ModelUsageBudget`, `daily_budget_usd=0.25`, `max_single_call_estimated_usd=0.05` em `settings.json`).

Ponto central: o `ModelRouter` escolhe o provider por **tarefa reconstruída** (`task_profile`, ver secção 5), não pelo comprimento literal da mensagem do utilizador. Um follow-up de 3 palavras ("ainda consegues melhorar") pode reconstruir uma tarefa de alta complexidade; uma mensagem longa pode ser trivial. A heurística antiga de comprimento/palavras-chave (`_complexity_reason_for_claude`) continua a existir para chamadas não documentais, mas para tarefas documentais o `task_profile` tem sempre prioridade.

### 2. Estado Do Routing

Caminhos determinísticos principais observados em `assistant/conversation.py`/`assistant/model_router.py`:

- **SOCIAL_FAST_PATH** — conversa social simples (saudações, "como estás", agradecimentos) resolvida sem qualquer chamada LLM.
- **SYSTEM_DATETIME** / consultas de estado do sistema — resolvidas via ferramentas do Context Observer, zero chamadas LLM.
- **DOCUMENT_TASK** — leitura, rewrite, refinement e ações sobre `ActiveDocumentContext`; leitura/display são deterministas (zero LLM), rewrite/refinement passam pelo `ModelRouter`.
- **GENERAL_CONVERSATION** — fallback para tudo o que não tem caminho determinístico próprio; passa pelo `ResponseComposer`/`RoutedLLM`.

Routing automático (`ModelRoutingConfig.mode == "automatic"`, `ModelRouter._automatic_decision`):

- fontes em `NO_PAID_CALL_SOURCES` (`TOOL_SELECTOR`, `MEMORY_SUMMARY`, `SESSION_SUMMARY`, `VOICE_CRITIC`, `PLANNER`) ficam sempre no provider local;
- gates de autorização, nesta ordem: `claude_enabled` → API key → `PAID_CALL_CONFIRMATION_ENV`;
- para tarefas documentais (`task_profile` presente com `task_type` a começar por `document_`), delega em `ModelRouter._document_task_decision` (ver secção 5);
- para tudo o resto, usa a heurística genérica `_complexity_reason_for_claude` + `ModelUsageBudget.can_spend`.

Reason codes atuais confirmados no código (`assistant/model_router.py`) — lista fiel, sem invenção:

- `explicit_provider`, `local_mode`, `claude_mode` — seleção de modo/provider explícita.
- `source_kept_local` — fontes em `NO_PAID_CALL_SOURCES`.
- `automatic_claude_disabled`, `missing_api_key`, `paid_calls_not_confirmed` — gates genéricos falhados (chamadas não documentais).
- `document_escalation_blocked_api_key`, `document_escalation_blocked_paid_disabled` — os mesmos gates, variante documental.
- `low_complexity` — heurística genérica não encontrou motivo para Claude.
- `document_simple_local` — tarefa documental banda `low`.
- `document_local_first` — tarefa documental banda `medium`, 1ª tentativa (ou `preferred_provider="ollama"` forçado).
- `document_local_regeneration` — banda `medium`, 2ª tentativa, ainda não justifica Claude (renomeado de `document_local_validation_failed`).
- `document_escalation_blocked_budget` — banda `high`, mas orçamento insuficiente.
- `document_escalated_after_local_failure` — banda `high`, a tentativa anterior foi Ollama e falhou validação.
- `document_regenerated_with_claude` — banda `high`, a tentativa anterior já era Claude e falhou validação (nunca confundido com falha local).
- `iterative_refinement_high_complexity` — refinamento iterativo, banda `high`, sem falha anterior a considerar.
- `document_high_complexity` / `document_claude_selected` — fallback dentro da banda `high` quando nenhuma das razões acima se aplica.
- Heurística genérica (`_complexity_reason_for_claude`, não documental): `professional_writing`, `structured_summary`, `document_review`, `document_interpret`, `document_rewrite`, `document_synthesis`, `complex_planning`, `technical_explanation`, `long_prompt`, `complex_request`.

Fallback: qualquer gate falhado devolve sempre o provider `ollama` (nunca um erro visível ao utilizador); budget gates usam `ModelUsageBudget.can_spend` (chave `daily_budget_usd`/`max_single_call_estimated_usd`); paid call authorization exige as 3 condições da secção 1 simultaneamente.

### 3. Document System

Estado atual do subsistema documental (`assistant/conversation.py`):

- workspace search fuzzy (`_workspace_search`) e leitura determinística de `.txt`/`.md`/`.pdf`/`.docx`;
- `ActiveDocumentContext` mantém o estado do documento ativo entre turnos (TTL, cache por mtime/tamanho), com os campos: `content` (original), `draft_content`, `previous_draft_content`, `draft_revision`, `previous_draft_revision`, `draft_action`, `draft_created_from_hash`, `draft_saved`, `draft_updated_at`;
- ações reconhecidas sobre o documento ativo: `display` (mostra original/conteúdo integral), `review`, `interpret`, `rewrite` (cobre tanto o 1º rewrite como o refinamento iterativo — diferenciados por `is_refinement = bool(context.draft_content)`), `display_draft`, `compare_versions`, `discard_draft`, `restore_previous_draft`, `save_draft`;
- `rewrite`/refinement criam sempre um draft transitório em memória — nunca escrevem no disco automaticamente;
- "mostra a versão melhorada" devolve o draft; "mostra o original" devolve sempre o `content` original intacto; "compara as versões" mostra ambos; "descarta as alterações" limpa o draft; "volta à versão anterior" restaura exatamente um nível de undo (`previous_draft_content`/`previous_draft_revision` — não é um histórico ilimitado);
- "guarda esta versão" pede confirmação explícita e só grava num ficheiro novo (`"<nome> (melhorado)<ext>"`) — o original nunca é sobrescrito sem pedido explícito;
- o draft (e todo o `ActiveDocumentContext`) existe apenas em memória do processo — nunca é escrito em `long_term_memory`/SQLite, não sobrevive a um reinício do `AssistantEngine`.

### 4. Iterative Document Refinement

Follow-ups implícitos de refinamento são reconhecidos por `_looks_like_draft_refinement_request` (`_DRAFT_REFINEMENT_STANDALONE_PHRASES` + heurística de par próximo comparativo/qualidade). Frases atualmente reconhecidas incluem: "ainda consegues melhor", "ainda consegues melhorar", "consegues fazer melhor", "consegues melhorar isto/isso", "tenta melhorar", "refina mais/isto/a versão", "melhora mais um pouco", "podes melhorar mais", "faz uma versão ainda melhor", "outra versão mas melhor", "não está/ficou suficientemente formal", "quero algo mais profissional", e qualquer combinação próxima (até 40 caracteres de distância) de uma palavra comparativa (`ainda`, `mais`, `outra vez`, `de novo`, `um pouco`) com uma palavra de qualidade (`melhor`, `formal`, `profissional`, `claro`, `simples`, `conciso`, `elegante`) — cobre "mais formal" isolado.

Com draft ativo (`context.draft_content` não vazio): resolve para `DOCUMENT_TASK` → `document_followup_action="iterative_refinement"` → usa o draft atual (`base_content = context.draft_content`) como base da reescrita; o documento original só é consultado como referência factual (`<DOCUMENTO_ORIGINAL_REFERENCIA>` no prompt), nunca sobrescrito.

Sem draft ativo, estas frases não inventam contexto documental — caem no routing normal (tipicamente `GENERAL_CONVERSATION`).

Mecânica de revisão: `draft_revision` incrementa a cada rewrite/refinement aceite (1 = primeiro rewrite, 2+ = cada refinamento seguinte); exatamente um nível de draft anterior é preservado (`previous_draft_content`/`previous_draft_revision`) para `restore_previous_draft`. Um turno de rewrite/refinement nunca excede 2 chamadas LLM (1ª tentativa +, se necessário, 1 regeneração corretiva). Proteções: `_is_noop_refinement` rejeita uma "melhoria" praticamente idêntica ao draft atual (similaridade `difflib` ≥ 0.97); `_validate_rewrite_draft` rejeita placeholders, respostas vazias/curtas, respostas terminadas em pergunta, comentários em vez do documento, ou perda de entidades/listas/saudação/assinatura relevantes; uma falha na 1ª tentativa dispara uma regeneração corretiva (prompt específico com o motivo da falha); se a 2ª também falhar, nenhum draft é criado e a falha é comunicada honestamente.

### 5. Document Task Complexity

Campos atuais do `task_profile` (construído em `_build_task_profile`, `assistant/conversation.py`, e consumido por `ModelRouter._document_task_decision`): `task_type` (`document_rewrite`/`document_refinement`), `document_chars`, `document_has_draft`, `document_revision_number`, `document_structure_count`, `document_named_entity_count`, `document_list_item_count`, `document_requires_fidelity`, `document_requires_full_output`, `document_previous_provider`, `document_previous_call_was_local`, `document_previous_call_was_paid`, `document_previous_validation_failed`, `document_previous_local_failure`, `document_validation_failure_reason`, `document_regeneration_attempt`, `preferred_provider`. Nenhum destes campos contém o conteúdo integral do documento — apenas contagens, booleanos e códigos curtos.

`_document_task_complexity_score(profile)` (pesos atuais, copiados do código, `assistant/model_router.py`):

- `+45.0` se `task_type == "document_refinement"`, senão `+10.0`;
- `+min(20.0, (document_named_entity_count + document_list_item_count) * 3.0)`;
- `+min(15.0, document_chars / 300.0)`;
- `+20.0` se `document_previous_local_failure`;
- `+10.0` se `document_validation_failure_reason == "placeholder_detected"`;
- `+10.0` se `document_regeneration_attempt >= 2`.

`_document_task_complexity_band(score)` — bands atuais: `low` (`score <= 15.0`), `medium` (`score <= 40.0`), `high` (`score > 40.0`). (`_DOCUMENT_COMPLEXITY_LOW_MAX = 15.0`, `_DOCUMENT_COMPLEXITY_MEDIUM_MAX = 40.0`.)

Lógica geral: leitura/display determinístico nunca chama o router (0 chamadas LLM); rewrite simples de 1ª tentativa fica local-first (banda tipicamente `medium`); refinamento iterativo é sempre banda `high` (peso base 45 já excede o limiar de 40); uma falha local relevante numa tarefa já em banda `high` pode escalar a 2ª tentativa para Claude, sujeito aos gates de orçamento/autorização da secção 1; o máximo global permanece 2 chamadas por turno. Os valores/limiares acima não foram alterados nesta tarefa — são cópia fiel do código atual.

### 6. Provider Provenance

Correção recente (commits `0f64194`/`f8b9f58`/`4c41a49`/`72ddd5b`): um teste pago real (secção 7) revelou que uma falha na 1ª tentativa era sempre assumida como falha **local**, mesmo quando essa 1ª tentativa já tinha sido servida por Claude. `previous_provider` passou a ser propagado corretamente (lido do `routing_decisions` real, nunca assumido); `document_previous_local_failure` só é `true` quando a tentativa anterior foi de facto `ollama` **e** falhou validação.

Sequências e reason codes correspondentes:

- Ollama → Claude: `document_escalated_after_local_failure`
- Claude → Claude: `document_regenerated_with_claude`
- Ollama → Ollama: `document_local_regeneration`

Telemetria de proveniência guardada por turno (`get_last_turn_telemetry()`): `provider_attempt_sequence` (lista por tentativa, nunca deduplicada), `initial_provider`, `final_provider`, `attempt_count`, `attempt_failure_reasons` (lista por tentativa), `initial_routing_reason_code`, `final_routing_reason_code`.

### 7. Teste Real Com Claude

Foi realizado um teste pago real controlado (fora desta sessão de agente — nenhuma chamada Anthropic real foi feita pelo assistente em qualquer uma das tarefas de implementação/validação). Sequência:

1. ler `email_resumo_novo`;
2. "Torna-o mais formal, mas não guardes ainda."
3. "Ainda consegues melhorar?"

Resultado observado no passo 3: `selected_path=DOCUMENT_TASK`, `provider=anthropic`, `model=claude-haiku-4-5-20251001`, `paid_call=true`, `llm_calls=2`, original mantido intacto, nova versão produzida.

Este mesmo teste revelou a telemetria errada `document_escalated_after_local_failure` para uma sequência Anthropic→Anthropic (a 1ª tentativa já tinha sido Claude, não Ollama) — a causa e a correção estão descritas na secção 6; já foi corrigida e validada (39/135/810 testes, ver secção 12). Nenhuma API key ou segredo foi registado neste checkpoint.

### 8. Fidelidade Documental

Os quatro prompts de rewrite/refinement (`_document_rewrite_prompt`, `_document_rewrite_correction_prompt`, `_document_refinement_prompt`, `_document_refinement_correction_prompt`, `assistant/conversation.py`) instruem agora explicitamente para: preservar todos os factos, nomes, destinatário, assunto, listas e assinatura do original; preservar tempos verbais e cronologia factual; não mudar quem fez o quê, quando, nem se algo está concluído, previsto ou pendente; melhorar estilo sem reinterpretar cronologia.

Decisão deliberada: não existe (ainda) um validador semântico determinístico para tempos verbais/cronologia — só a instrução no prompt, mais um teste que confirma a presença da instrução nos 4 prompts (`tests/test_automatic_routing_regressions.py`). Motivo registado: evitar falsos positivos de uma verificação heurística de gramática/tempo verbal sobre texto livre.

### 9. Cancelamento / Timeout / Progress

Cancelamento cooperativo: `AssistantEngine.begin_request()` cria um `threading.Event` novo por request; `cancel_current_request()` (thread-safe, pode ser chamado de uma worker thread Qt) marca o evento; `is_cancel_requested()` é verificado em checkpoints explícitos dentro do fluxo de rewrite. Não interrompe uma chamada `requests.post` síncrona já em execução (a chamada ao provider corre até ao fim); o cancelamento impede que uma 2ª tentativa/regeneração ou a criação de um draft aconteçam depois desse ponto.

Timeout total do rewrite: `DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS`, verificado antes de iniciar cada tentativa (`_remaining_seconds()`); combinado com o limite de 2 chamadas por turno.

Progress events emitidos (`_emit_progress`, nomes literais confirmados no código): `rewrite_attempt_started`, `rewrite_validation_started`, `rewrite_regeneration_started`, `rewrite_cancelled`, `rewrite_timeout`. Os progress labels correspondentes não entram no histórico de conversa (`ConversationMemory`) — são só eventos efémeros para a UI. Em cancelamento/timeout a UI regressa a `idle`.

### 10. Estados Visuais Do Echo

`prototype_web_ui/web/echo_entity.js` — `STATE_CONFIG` centralizado com os estados cognitivos: `idle`, `listening`, `thinking`, `reading`, `working`, `speaking`, `error`. Papéis espaciais (`layoutRole`, ortogonais ao estado cognitivo, via `ROLE_INTENSITY`): `normal` (implícito), `compact`, `focus` — só escalam intensidade (nunca tamanho/posição, que continuam a cargo do Adaptive Layout, fonte de verdade para posição).

Transições entre estados são suaves (glide, não corte abrupto); `prefers-reduced-motion` reduz a intensidade do movimento (fator `0.4`) em vez de o eliminar — Echo continua a "respirar" visivelmente mesmo com movimento reduzido. `reading` é acionado via a telemetria já existente (`execution_path=="document_task"`); `working` cobre o decurso de um rewrite/refinement; `speaking` é deliberadamente saltado em cancelamento/timeout (não há resposta falada para uma resposta que não existe). Em `compact`/`focus`, `ROLE_INTENSITY` nunca chega a `0` (mínimo `0.75` em `focus`) — o Echo nunca desaparece completamente.

### 11. Memória E Conversação

Estado atual conhecido (sem alterações nesta tarefa): o histórico curto (`ConversationMemory`, `data/history.json`) é enviado ao modelo em cada turno; o Voice Critic usa esse mesmo histórico; a memória persistente (SQLite — `LongTermMemory`, `PersonalModel`, `SessionManager`) existe e é consultada, mas não é despejada literalmente no prompt; o `ContextObserver` regista snapshots de atividade e pode informar respostas sobre o estado do computador; há recuperação de contexto entre sessões (session summaries, "onde ficámos?"); existe cuidado ativo (guards) para não ecoar literalmente texto de memória antiga como se fosse a resposta atual. `GENERAL_CONVERSATION` e `SOCIAL_FAST_PATH` continuam a ser os caminhos para conversa sem tarefa determinística própria — `SOCIAL_FAST_PATH` não responde de forma rígida a saudações com conteúdo relevante a seguir (ver `_has_relevant_content_after_greeting` em `assistant/response_composer.py`); referências naturais a contexto recente são feitas quando há contexto disponível, nunca inventadas.

### 12. Testes Atuais

- `tests/test_model_router.py` → **39 passed**
- `tests/test_automatic_routing_regressions.py` → **135 passed**
- suite completa (`pytest`) → **810 passed**, 0 falhas

Todas as respostas Anthropic nos testes automáticos usam `FakeProvider` — nenhuma chamada Anthropic real acontece na suite. A única chamada Anthropic real associada a este trabalho é o teste manual pago controlado descrito na secção 7, feito fora da execução dos testes automáticos.

### 13. Commits Recentes Importantes

```
22754b1 feat: add spatial and behavioral echo states
275a00f test: cover echo visual states and progress labels
5ca08fc feat: support iterative document refinement
e7c8771 test: cover iterative document refinements

f9aa925 feat: score reconstructed document task complexity
94680c2 feat: propagate document routing profiles
26873e4 feat: route complex document refinements by task profile
085cad7 test: cover document complexity routing policy
a1fb3fc test: cover document provider escalation workflows

0f64194 fix: track document rewrite provider provenance
f8b9f58 fix: report document regeneration routing accurately
4c41a49 test: cover document rewrite provider sequences
72ddd5b test: verify local failure provider provenance
```

HEAD atual: `72ddd5b`.

### 14. Worktree Local

No momento desta atualização, esperado e **não commitado**:

- `modified: config/settings.json`
- `untracked: workspace/Reuniões Direção - Notas.pdf`
- `untracked: workspace/email_resumo_novo.txt`

Estes ficheiros não foram tocados por esta ou pelas tarefas anteriores de routing/provenance — permanecem fora de qualquer commit deliberadamente.

### 15. Next Development Step

Retomar desenvolvimento funcional/visual do Echo a partir desta base estável, priorizando a próxima evolução da experiência de workspace e comportamento da entidade.

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

## Atualização — Diagnóstico E Patches Do Subsistema Documental

Atualizado em 2026-08-06:

O `AssistantEngine` ganhou, numa sessão anterior a este registo, um subsistema documental completo: leitura, resumo e criação de ficheiros de workspace (`.txt`/`.md`/`.pdf`/`.docx`) com pending tasks (`PendingWritingTask`, `DocumentTaskRequest`), contexto de documento ativo (`ActiveDocumentContext`, com TTL e cache por mtime/tamanho) e pesquisa fuzzy de ficheiros (`_workspace_search`). Este subsistema ainda não estava descrito nas secções seguintes deste checkpoint; foi commitado como baseline em `1b36661`.

Foi feito um diagnóstico dedicado (pedidos muito semelhantes seguiam caminhos diferentes; `ActiveDocumentContext` parecia perder-se em follow-ups ambíguos) e implementados três patches independentes, cada um em commit isolado:

- **PATCH A** (`6a9b471`) — Resolução determinística de leitura/display/conteúdo integral. Corrigiu: `_explicit_summary_request` cego a negação (tratava "sem resumir" como pedido de resumo); vocabulário de paráfrases de conteúdo integral demasiado estreito no follow-up face à primeira leitura; `_extract_semantic_file_reference` a capturar fragmentos relacionais ("que está") em vez do nome real do ficheiro; resposta do tipo "lê o ficheiro" → "email_resumo_novo" (stem nu, sem extensão) a entrar em loop por falta de resolução fuzzy.
- **PATCH B** (`4457abe`) — Fallback grounded para o `ActiveDocumentContext`. Adicionou `_try_active_document_grounded_fallback`: quando nenhum marcador específico de follow-up é reconhecido mas a mensagem parece uma opinião/sugestão sobre o documento ativo, a resposta continua grounded no ficheiro (`grounding_sources=['WORKSPACE_FILE']`) em vez de cair em `GENERAL_CONVERSATION`. Corre só depois de todos os handlers mais específicos (pending task, follow-up explícito, writing task, fast route, topic shift, research, system/tool intent) terem decidido não reclamar o turno, preservando a limpeza de contexto em mudança de assunto ("vamos falar de outra coisa").
- **PATCH C** (`15ca0ad`) — Alargamento dos marcadores review/interpret/rewrite em `_active_document_followup_action`, partilhando as tabelas de sinónimos introduzidas no PATCH B em vez de duplicar listas fixas. Mensagens como "tens alguma sugestão para melhorar o mail?" passam a ser classificadas como `review` já neste estágio inicial, não só pelo fallback genérico do PATCH B.

Validação: suite de testes cresceu de 719 para 726 (`pytest` completo), sem falhas; zero chamadas Anthropic reais durante o diagnóstico e implementação (só `ollama`/`llama3.1:8b` e `FakeProvider` nos testes); frontend e `config/settings.json` não foram tocados.

Dívida ainda por resolver, fora do âmbito destes três patches: telemetria `composer_call_count` continua a ser um rótulo estático da rota escolhida, não prova de chamada LLM concluída; `configured_model_mode_source` continua a ler de `self.llm.settings` em vez de `self.model_runtime`, ficando vazio nos caminhos que passam pelo Composer. (Nota: este segundo ponto foi corrigido em `2a05588` — ver secção seguinte. `composer_call_count` continua por resolver.)

## Atualização — Fluxo De Rewrite Com Draft Transitório E Validação Estrutural

Atualizado em 2026-08-06 (continuação):

Uma regressão encontrada em validação manual — pedidos naturais de rewrite como "Torna-o mais formal, mas não guardes ainda." caíam em `GENERAL_CONVERSATION`, porque `_active_document_followup_action` só reconhecia frases exatas ("torna mais formal"), quebrando com pronomes presos por hífen ("torna-o") — foi corrigida e o fluxo de rewrite foi reforçado em seis commits isolados:

- **`c88a55c`** — Deteção natural de rewrite (`_looks_like_rewrite_request`: verbo+qualidade por proximidade via `find_near_pair_span`, em vez de frases exatas). `corrige`/`corrigir` movidos dos marcadores de review para rewrite (estavam em ambas as listas; review, verificado primeiro, ganhava sempre).
- **`e430175`** — Estado de draft transitório: `ActiveDocumentContext` ganha `draft_content`/`draft_action`/`draft_created_from_hash`/`draft_saved`/`draft_updated_at`. Novas ações de follow-up: `display_draft`, `compare_versions`, `discard_draft`, `save_draft` (pede confirmação e grava num ficheiro novo, nunca sobrescreve o original).
- **`69da610`** — Validação estrutural do draft antes de o aceitar. `_extract_document_anchors` extrai assunto/saudação/assinatura/itens de lista/entidades do documento **original**, de forma genérica (sem nomes hardcoded). `_validate_rewrite_draft` rejeita respostas vazias, demasiado curtas, terminadas em pergunta, com frases de comentário/recusa, ou que percam entidades/listas/saudação/assinatura importantes. Uma regeneração corretiva é tentada se a 1ª resposta falhar a validação; se a 2ª também falhar, não é criado draft nenhum.
- **`2ee79f5`** — Perguntas de opinião curtas e sem antecedente ("O que achas?" depois de "Vamos falar de outra coisa.") passam a receber "Sobre o quê?" deterministicamente, evitando uma resposta adivinhada que o guard de memória depois substituía por uma frase enlatada.
- **`2a05588`** — `configured_model_mode_source` deixa de ficar vazio em turnos com chamada LLM real em modo `local`/`automatic`.
- **`38271a9`** — Testes cobrindo tudo o que precede.

Fluxo documental atual:

- Leitura integral determinística do ficheiro ativo (zero chamadas LLM).
- Continuidade via `ActiveDocumentContext` entre turnos (TTL, cache por mtime/tamanho).
- `review` (sugestões), `interpret` (explicação) e `rewrite` (texto completo reescrito) são ações distintas, cada uma com deteção de linguagem natural própria.
- `rewrite` cria sempre um draft transitório em memória — nunca escreve no disco automaticamente.
- "mostra a versão melhorada" devolve o draft; "mostra o original" devolve sempre o conteúdo original; "compara as duas versões" mostra ambos; "descarta as alterações" limpa o draft.
- "guarda esta versão" pede confirmação explícita e, só se aceite, grava num ficheiro novo — o original nunca é sobrescrito automaticamente.
- Um rewrite só é aceite depois de validado estruturalmente contra âncoras extraídas do original; uma resposta inválida tem direito a uma regeneração corretiva; se ambas falharem, a falha é comunicada honestamente e nenhum draft é criado.
- O draft nunca é escrito em `long_term_memory`/SQLite — existe só em `ActiveDocumentContext`, em memória.
- Perguntas de opinião curtas sem contexto recebem uma clarificação simples em vez de uma resposta adivinhada.

Limitações atuais:

- O draft existe apenas durante a sessão/processo atual — não sobrevive a um reinício do `AssistantEngine`.
- Paráfrases muito agressivas do modelo podem ser rejeitadas pela validação estrutural mesmo sendo uma reescrita válida — os limiares são heurísticos (proporção de comprimento, entidades, listas), não uma comparação de significado.
- Guardar cria sempre uma nova versão (`"<nome> (melhorado)<ext>"`); não existe mecanismo para substituir o ficheiro original.
- `ActiveDocumentContext` (incluindo o draft) não é persistido entre reinícios da aplicação.

Validação: `tests/test_automatic_routing_regressions.py` — 89 passed; `tests/test_memory_verbalization.py` — 18 passed; `pytest` completo — 736 passed, 0 failed. Zero chamadas Anthropic reais; frontend e `config/settings.json` não tocados.

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
- `_try_document_task()` / `_try_pending_document_task()` / `_try_active_document_followup()` / `_try_active_document_grounded_fallback()`;
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
- briefing/session continuity;
- leitura/resumo/criação de ficheiros da workspace, grounded em conteúdo real (subsistema documental, ver "Atualização — Diagnóstico E Patches Do Subsistema Documental").

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
