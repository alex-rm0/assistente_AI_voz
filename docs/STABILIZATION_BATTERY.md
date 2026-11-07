# Bateria de estabilização — Echo OS

> Os cenários 1–3 e 11–14 correm automaticamente com `scripts/battery_driver.py` (app real + Ollama real, QWebEngine offscreen). Escreve em `data/`: faz backup antes. Instruções no topo do script.

Ronda: ____  Data: ____  Commit: `git log -1 --oneline` → ________

Como correr: `python app.py --ui echo-os *> output.txt` (PowerShell). Um cenário de cada vez, pela ordem.
Marca **OK / FALHA / PARCIAL**. Em FALHA, escreve o que viste e guarda o `output.txt` dessa ronda.
Não corrijas nada a meio — primeiro corre tudo, depois fazemos a triagem.

| # | Cenário | O que fazer | O que deve acontecer | Log a procurar | Resultado | Notas |
|---|---|---|---|---|---|---|
| 1 | Arranque | Abrir a app | Janela abre < ~10 s, sem erro visível | `signals ligados`, `page_load_finished=True` | OK | Automático (QWebEngine offscreen, `app.main()` real), corrida de 2026-11-07 no commit `fb08eb2`. `page_load_finished=True` e `signals ligados` presentes, sem erro da app. Janela visível aos 4,7 s e UI pronta (`window.echoController`) aos 5,5 s. Na 1.ª corrida (2026-10-29) foram 14 s / 15,5 s; a diferença não foi isolada (cache do SO, embeddings já desligados). O offscreen continua a gerar milhares de linhas de erro GPU do Chromium (7 952 nesta corrida), artefacto do offscreen. |
| 2 | Chat escrito | Escrever "Olá" + Enter | Resposta aparece, curta, na língua certa, cursor volta ao campo | `[UI_RESPONSE] event=received` + `render_complete visible=true` | OK | Automático. `SOCIAL_PATH`/`SOCIAL_FAST_PATH`, `llm_calls=0`, resposta "Hi. How are you?" (en-GB já definido). `[UI_RESPONSE] event=received chars=16` e `render_complete chars=16 visible=true`; o foco voltou a `#voiceIn`. 106 ms (antes 21,9 s, por causa dos embeddings; ver KNOWN_ISSUES §4, mitigado). |
| 3 | Chat escrito 2 | Logo a seguir: "Qual é a capital de França?" | Resposta nova substitui a anterior, sem inventar | idem | OK | Automático. `GENERAL_CONVERSATION`/`RESPONSE_COMPOSER`, `llm_calls=1`, resposta "Paris." (correta), que substituiu a anterior (`cleared reason=new_message` → `render_complete chars=6 visible=true`). 3,6 s, incluindo o carregamento do modelo de chat no Ollama (antes 16,6 s). |
| 4 | Entrar em standby | Clicar VOICE MODE | Botão passa a STANDBY | `standby_entered` | | |
| 5 | Wake | Dizer "Echo" / "Acorda, Echo" | Botão → ON, ack aparece no ecrã E é falado, < ~2 s | `wake_intent_matched`, `[UI_RESPONSE] event=received` | | |
| 6 | Conversa por voz | Dizer "Olá, como estás?" | Resposta aparece e é falada; não inventa atividade | `transcript_ready`, `render_complete visible=true` | | |
| 7 | Rearm | Esperar após a resposta | Resposta continua visível; volta a ouvir | `rearm_listening`, `listening_started` | | |
| 8 | Barge-in | Fazer uma pergunta longa e falar por cima da resposta | TTS pára, a tua fala é captada | `[BARGE_IN] event=confirmed` | | |
| 9 | STOP | Durante uma resposta falada, carregar STOP | Fala pára de imediato, sem erro | `playback ... reason=` stop | | |
| 10 | Sleep | Dizer "podes ir dormir" | Volta a STANDBY, deixa de responder | `standby_entered` | | |
| 11 | Task | Escrever "Ajuda-me a comparar dois portáteis" e depois "o critério é a bateria" | Card da task aparece e a 2.ª mensagem continua a mesma task | `task_id` igual nos dois turnos | OK | Automático. `TASK_START` e depois `TASK_CONTINUATION`, sempre `"task_id": "task-1"`, `llm_calls=0`. O critério passou a ser registado (commits `624bdf4`, `4845e48`). 2.ª resposta: "Got it: the battery matters in the decision. Still need to understand options to compare." (já não repete a 1.ª). O card passou a ter "PRIORITIES the battery matters a lot in the decision" e "STILL MISSING options to compare" (já sem "decision criteria"). Continua a misturar idiomas: rótulo "TAREFA" e objetivo "comparar opcoes". Critérios fora da lista fixa (ex.: "o ecrã") não são suportados (KNOWN_ISSUES §9). |
| 12 | Mudança de idioma | Escrever "responde em inglês" e depois "olá" | Passa a inglês e mantém; UI/task também | `response_language=en-GB` | PARCIAL | Automático. Igual à 1.ª corrida: `LANGUAGE_COMMAND` → "Sure. I'll respond in English."; depois "olá" → "Hi. How are you?" (`SOCIAL_FAST_PATH`). `response_language` já era `en-GB` antes do cenário (`[TTS_AVAILABILITY] response_language=en-GB` no arranque), por isso a *mudança* continua por demonstrar; é preciso repetir a partir de pt-PT. UI/task não acompanham: `<html lang>` continua `pt-PT` e o card mantém "TAREFA"/"comparar opcoes". |
| 13 | Ferramenta | "Que janela tenho aberta?" | Responde com dados reais (Context Observer), sem inventar | `tools_used=[...]` não vazio | OK | Automático. `TOOL_PATH`/`TOOL_RESULT`, `tools_used=['get_active_window']`, `llm_calls=0`, resposta real "Active window detected: WhatsApp" (a janela em primeiro plano nesse momento), `render_complete visible=true`. 545 ms. Corrigido no commit `7fe9162` (marcadores do routing e varrimento git sem venvs/caches; antes 35 s). Na 1.ª corrida dava `tools_used=[]` e uma resposta inventada. |
| 14 | Shutdown | Fechar a janela | Fecha < ~2 s, sem crash nem processo pendurado | `[SHUTDOWN] event=completed` | OK | Automático. `close()` devolveu em 20 ms com a janela já escondida. `[SHUTDOWN] component=session_close event=finished elapsed_ms=18`, `[SHUTDOWN] event=completed total_ms=18`; event loop e processo terminaram com código 0, sem traceback. Já não há pedidos a `/api/embeddings` no fecho (KNOWN_ISSUES §4). |

## Observações gerais (latência, coisas estranhas, sensação)

-
