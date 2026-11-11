# Problemas conhecidos

Atualização de 2026-11-11: a ronda de correções do email, cartão e grounding da
conversa está registada em [LIVE_CONVERSATION_STABILIZATION.md](docs/LIVE_CONVERSATION_STABILIZATION.md).
Baseline dessa ronda: **2086 passed, 1 xfailed**. As notas históricas abaixo
mantêm a data e o estado da respetiva observação; não são uma lista atualizada
de todas as correções posteriores.

Data: 2026-10-21
Estado dos testes: pytest: 1808 passed

Ponto de estabilização. Estes problemas estão identificados e **ainda não
foram corrigidos**. Cada entrada tem o sintoma, a evidência e o estado.

---

## 1. Autoridade do idioma da resposta (response_language)

- **Sintoma:** com `response_language=en-GB` e o utilizador a falar
  português, o Echo respondeu em português. A resposta do composer também
  inventou atividade ("Estava a trabalhar num projeto até à noite").
- **Evidência (log real de 2026-10-21):**
  - `voice_reviewed=no`: o voice critic **não** correu.
  - `response_before_voice_critic` era a resposta inventada do composer.
    `response_after_voice_critic` passou a ser um literal fixo em português:
    "Não tenho essa informação sobre a tua atividade real — só o Context
    Observer sabe isso."
  - A substituição foi feita por `_detect_ungrounded_activity_claim`
    (definida em `assistant/conversation.py:11627`; o literal está em
    `assistant/conversation.py:6656`).
  - Esse literal não passa por `localize_deterministic_response` nem é
    apanhado por `looks_mixed_for_language` (`assistant/response_language.py`).
    Por isso sai em português mesmo com en-GB.
  - A mensagem expõe ao utilizador o nome interno "Context Observer".
- **Estado:** parcialmente corrigido (2026-11-09). A política está decidida
  no §7.
  - **Corrigido:**
    - O composer usa um system prompt inteiramente inglês com en-GB, tanto
      em `compose()` como em `regenerate()` (commit `551816b`). Com pt-PT
      não mudou.
    - Os 6 textos dos guards de finalização em `_complete_turn` passam por
      `localize()` em PT e EN (commit `bbc27cc`): tool claim, memory recall
      claim, memory claim, duração histórica, atividade e workspace. O
      texto da atividade deixou de expor "Context Observer".
    - Com en-GB, `compose()` e `regenerate()` acrescentam "(Reply in
      English.)" ao fim da última mensagem do utilizador. Isto só afeta o
      payload; o histórico e a UI não mudam (commit `d1135a8`). Antes, a
      instrução de idioma estava só na mensagem `system`.
    - **Validação com o Ollama real e en-GB (5 corridas cada): 15/15 em
      inglês.** "Qual é a capital de França?" passou de 1/3 para 5/5.
      "Explica-me o que é um algoritmo numa frase" deu 5/5. "Olá, tudo
      bem? Diz-me uma curiosidade." deu 5/5. Por isso não foi preciso
      acrescentar uma verificação de idioma com regeneração.
    - A página e o card da task acompanham o idioma (commit `268e6df`):
      `<html lang>`, os rótulos "TAREFA"/"TASK" e os botões, o objetivo
      ("comparar opções"/"compare options"). Ao mudar de idioma a meio de
      uma task, o mesmo card é atualizado (mesmo `task_id`).
    - Mais três textos visíveis passaram por `localize()`:
      - a resposta das ferramentas do Context Observer sem dados
        (`get_active_window`, `get_active_application`,
        `get_open_windows`, `get_recent_activity`);
      - o subtítulo do card de task, que mostrava o estado interno cru
        ("ready"). Agora mostra "pronta"/"ready", "a recolher
        informação"/"collecting details", etc., sem mudar o valor interno;
      - o card do email (`compose_email`): título "Email para"/"Email to",
        "Para:", "Contexto:", "Objetivo:", "Assunto:", os marcadores de
        estado e os papéis e finalidades.
  - **Falta:**
    - **Sem verificação do idioma gerado:** se o modelo voltar a responder
      em português, nada o deteta nem regenera, além do
      `looks_mixed_for_language`, que só apanha respostas que misturam as
      duas línguas.
    - **Detetores dos guards só em português:** os marcadores (ex.: "se
      não me engano", "pesquisei", "lembro-me") só conhecem frases em
      português, por isso uma resposta inventada em inglês não é
      bloqueada.
    - **WRITING_TASK em en-GB:** "Ajuda-me a escrever um email" responde
      com o literal "Para quem é o email?" em português (caminho
      determinístico do `WRITING_TASK`, fora do composer). A frase longa
      do log de 11/11 deu "Who is the email for?", por isso há pelo menos
      dois caminhos e só um está traduzido.
    - **Literais PT fora deste bloco:** ver a lista no relatório de
      2026-11-09. Por exemplo, `_local_writing_fallback` devolve um email
      fixo em português. As ferramentas técnicas do Context Observer
      (`get_last_context_snapshot`, `get_current_activity_summary` e
      `get_raw_context_snapshot`) continuam só em português e não recebem
      `response_language`.

## 2. Campos de trace `voice_critic_*` com nome enganador

- **Sintoma:** os campos `response_before_voice_critic` e
  `response_after_voice_critic` sugerem uma medição do voice critic.
- **Evidência:** no caso acima mudaram de valor com `voice_reviewed=no`. Na
  prática medem a diferença antes/depois dos guards de resposta, não o critic.
- **Estado:** aberto.

## 3. Saudação formal não apanhada pelo social fast path

- **Sintoma:** "Olá, como está?" (forma formal) não segue pelo social fast
  path.
- **Evidência:** no log real foi `GENERAL_CONVERSATION` →
  `RESPONSE_COMPOSER`, com uma chamada ao LLM (`llm_calls=1`).
- **Estado:** corrigido. `is_pure_social_turn` passou a aceitar as formas
  formais como frases exatas ("olá, como está?", "bom dia, como está?",
  "como vai?", "hello, how are you?"). Por serem exatas, "como está o
  projeto?" continua fora do social fast path. Teste:
  `tests/test_formal_greeting_social.py`. A resposta continua a ser a
  informal "Estou bem. E tu?".

## 4. Embeddings nunca funcionaram

- **Sintoma:** a pesquisa semântica da memória de longo prazo nunca
  funcionou; só existe o fallback por texto.
- **Evidência:**
  - `LongTermMemory` / `llm.embed` chamam `/api/embeddings` com o modelo de
    chat (`llama3.1:8b`). O Ollama responde HTTP 500, e cada chamada a frio
    demora cerca de 7 s (medido: 7,7 s).
  - 0 de 216 memórias em `data/long_term_memory.sqlite` têm embedding.
  - Uma destas chamadas corre no fecho da app (resumo de sessão). Desde a
    Phase 7.6.7 corre com a janela já escondida, mas continua a atrasar o
    fim do processo.
  - Bateria de 2026-10-29: o "Olá" demorou 22 s, todos gastos em
    `POST /api/embeddings` → 500 em 21,9 s.
- **Estado:** mitigado: desligado. Em `config/settings.json`, `memory` tem
  `"embeddings_enabled": false`, e a chave em falta também conta como
  desligado. `LongTermMemory._embed` devolve `None` sem fazer o pedido; o
  código dos embeddings mantém-se. A pesquisa continua a usar o fallback
  de texto, com resultados idênticos (12 consultas sobre uma cópia da
  base real). Na bateria seguinte houve 0 pedidos a `/api/embeddings` e o
  "Olá" passou a 104 ms. Falta, se um dia se quiser pesquisa semântica,
  usar um modelo próprio de embeddings (ex.: `nomic-embed-text`) e
  preencher as memórias existentes.

## 5. Memória poluída com resumos de sessão

- **Sintoma:** a memória de longo prazo está dominada por resumos de sessão
  sem conteúdo útil, um por cada arranque e fecho da app.
- **Evidência:** 202 de 216 memórias começam por "Resumo de sessao". São
  texto padrão (hora de início e fim, motivo de fecho, atividade genérica,
  ficheiros como `output.txt`/`.pytest_cache`), sem factos relevantes.
  Como o texto inclui as etiquetas `projeto=`, `tarefas=`, etc., as
  pesquisas "projeto", "tarefa" e "assistenteIA" só devolviam resumos.
- **Estado:** mitigado.
  - **Sessões vazias:** já não são gravadas. Só passa para a memória de
    longo prazo uma sessão com tarefas, ferramentas ou decisões
    (`session_summary_has_content`). `files_touched` não conta, porque
    são os ficheiros modificados recentemente segundo o Context Observer.
    O resumo continua a ser guardado no `session_manager.sqlite`.
  - **Limpeza:** `scripts/cleanup_session_summaries.py --apply`
    (2026-10-30) removeu 164 resumos vazios. As memórias passaram de 219
    para 55, e a timeline de 218 para 54. O backup está em
    `data/long_term_memory.backup-20261030-141920.sqlite`.
  - **Pesquisa genérica:** os 41 resumos restantes ficam de fora de
    `LongTermMemory.search` e `context_for`. Continuam guardados e
    aparecem na timeline por data. A continuidade de sessão lê do
    `session_manager`.
  - **Em aberto:** a deteção de decisões (`_track_decision` em
    `assistant/session_manager.py`) apanha frases que não são decisões,
    como "Da ultima vez, estivemos a trabalhar…" ou "Vou sugerir alguns
    destinos…". Por isso, alguns resumos sem conteúdo real ficam guardados.

## 6. Arranques a frio

- **Sintoma:** latência alta na primeira utilização de cada componente.
- **Evidência:**
  - Whisper medium: cerca de 9 s a frio (o primeiro fallback de wake na
    sessão mediu `stt_ms=9184`).
  - Pocket TTS: arranque do servidor e carregamento do modelo na primeira
    fala (`pocket_tts_server_start` → `ready` em cerca de 5 s no log real).
  - Ollama: carregamento a frio do modelo na primeira chamada.
- **Estado:** aberto.

## 7. Política de idioma

- **Sintoma:** não estava definido o que o Echo deve fazer quando o
  utilizador fala português com a definição en-GB (responder em en-GB,
  seguir o idioma falado, ou perguntar).
- **Evidência:** é o caso real do problema 1. O STT deteta `pt`, a definição
  é `en-GB` e o TTS usa a voz inglesa para texto em português.
- **Estado:** decidido (2026-11-09): `response_language` é uma **definição
  fixa**. Com en-GB, toda a resposta visível sai em inglês, mesmo que o
  utilizador fale português. Só muda com um comando explícito ("responde
  em português"). O que já cumpre esta política e o que falta está no §1.

## 8. Phase 8.1.1: UI não mostrava respostas depois do STANDBY

- **Sintoma:** em Voice Mode a resposta era gerada e falada pelo TTS, mas
  não aparecia na UI. Depois de entrar em STANDBY, nenhum sinal
  controller → JS chegava à página.
- **Evidência:** `_on_wake_prewarm_finished` estava ligado a um sinal sem
  `@Slot()`. O PySide acrescentava então um slot dinâmico ao QMetaObject do
  controller já publicado no QWebChannel. Reproduzido e verificado com
  QWebEngine real (`tests/test_ui_response_delivery.py`).
- **Estado:** corrigido no código e coberto por testes. **Falta a
  validação manual na app real** (microfone e janela reais).

## 9. Critérios de comparação fora da lista fixa

- **Sintoma:** numa comparação, um critério fora da lista fixa (ecrã,
  peso…) não fica registado. "o critério é o ecrã" vai para
  `GENERAL_CONVERSATION` (resposta do LLM) e o card continua com "Ainda
  falta: critérios de decisão".
- **Evidência:** `_criterion_mentions` (`assistant/task_intelligence.py`)
  só conhece câmara, bateria, suporte, preço e localização. O caminho dos
  objetos `kind="criterion"` (`extract_comparison_criteria`) só corre em
  operações "comparar …" e não alimenta `infer_unknowns`. Coberto por um
  teste `xfail` em `tests/test_comparison_criterion_statements.py`. As
  declarações simples com critérios da lista ("o critério é a bateria",
  "critério: preço", "decidir pela autonomia") já ficam registadas.
- **Estado:** aberto.

## 10. Qualidade das respostas do composer (notas, sem correção)

- **Echo inventa estado próprio em inglês:** com en-GB, a "Olá, tudo bem?
  Diz-me uma curiosidade." o Echo respondeu "Not much to report, just the
  usual system checks." (Ollama real, 2026-11-09). É estado inventado,
  mas os detetores dos guards só conhecem frases em português (§1), por
  isso uma afirmação destas em inglês não é bloqueada.
- **"Diz-me uma curiosidade" responde com uma pergunta:** nas 5 corridas
  da mesma validação, o Echo perguntou se o utilizador queria uma
  curiosidade ("Did you want a simple fact or a more interesting
  curiosity?") em vez de a dizer.
- **Estado:** aberto.

## 11. Fecho da app durante o arranque do Pocket TTS

- **Sintoma:** se a app fechar enquanto o prewarm do Pocket TTS (`9f86f03`)
  ainda está a arrancar o servidor, a janela esconde-se logo, mas o
  processo só sai quando o arranque acaba. Na bateria de 2026-11-11 foram
  2993 ms (antes 54 ms), com o servidor a demorar 14 s; isolado demora
  ~5–6 s. Com um servidor preso, a espera ia até ao timeout de carregamento
  (60 s por omissão).
- **Estado:** corrigido.
  - **Corrigido em `assistant/tts.py` (commit "fix: cancellable Pocket TTS
    startup"):** `PocketTTSLocalProvider.shutdown()` fecha o processo em
    arranque sem esperar pelo `_lock`. Quando o stdout do servidor acaba, o
    `start()` falha logo em vez de esperar pelo timeout. Testes:
    `tests/test_pocket_tts_cancellable_start.py`.
  - **Corrigido em `prototype_web_ui/controller.py` (commit `2b1871f`, do
    `stab/paralelo`, integrado no main pelo merge):** `request_shutdown()`
    para o TTS manager quando há um prewarm em curso, por isso o fecho já
    não espera pelo arranque. O prewarm automático espera pela UI pronta e
    mais 3000 ms; a entrada explícita em STANDBY/conversa continua
    imediata. O teste do microfone saiu do caminho de arranque. Medições e
    limites do teste em `docs/STARTUP_SHUTDOWN_TIMINGS.md` (fecho durante
    o prewarm: 4190 -> 58 ms).
