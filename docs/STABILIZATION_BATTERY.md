# Bateria de estabilização — Echo OS

> Os cenários 1–3 e 11–14 correm automaticamente com `scripts/battery_driver.py` (app real + Ollama real, QWebEngine offscreen). Escreve em `data/`: faz backup antes. Instruções no topo do script.

Ronda: ____  Data: ____  Commit: `git log -1 --oneline` → ________

Como correr: `python app.py --ui echo-os *> output.txt` (PowerShell). Um cenário de cada vez, pela ordem.
Marca **OK / FALHA / PARCIAL**. Em FALHA, escreve o que viste e guarda o `output.txt` dessa ronda.
Não corrijas nada a meio — primeiro corre tudo, depois fazemos a triagem.

| # | Cenário | O que fazer | O que deve acontecer | Log a procurar | Resultado | Notas |
|---|---|---|---|---|---|---|
| 1 | Arranque | Abrir a app | Janela abre < ~10 s, sem erro visível | `signals ligados`, `page_load_finished=True` | OK | Corrida de regressão de 2026-11-11 no commit `87b2094` (comparada com a de `3dc5d7c`). Janela visível aos 3,3 s e UI pronta aos 4,1 s (antes 4,5 s / 5,3 s). `page_load_finished=True` e `signals ligados`, sem erro. O prewarm do Pocket TTS (`9f86f03`) arranca logo no início (`[TTS_PREWARM] event=started`) sem atrasar a UI. |
| 2 | Chat escrito | Escrever "Olá" + Enter | Resposta aparece, curta, na língua certa, cursor volta ao campo | `[UI_RESPONSE] event=received` + `render_complete visible=true` | OK | `SOCIAL_PATH`, `llm_calls=0`, 107 ms (antes 105 ms), "Olá! Como estás?" em pt-PT (o driver repõe pt-PT no início). Foco de volta em `#voiceIn`. |
| 3 | Chat escrito 2 | Logo a seguir: "Qual é a capital de França?" | Resposta nova substitui a anterior, sem inventar | idem | OK | `GENERAL_CONVERSATION`, `llm_calls=1`, 221 ms com o Ollama já carregado (antes 2,3 s, que incluía carregar o modelo), "A capital da França é Paris." |
| 4 | Entrar em standby | Clicar VOICE MODE | Botão passa a STANDBY | `standby_entered` | | |
| 5 | Wake | Dizer "Echo" / "Acorda, Echo" | Botão → ON, ack aparece no ecrã E é falado, < ~2 s | `wake_intent_matched`, `[UI_RESPONSE] event=received` | | |
| 6 | Conversa por voz | Dizer "Olá, como estás?" | Resposta aparece e é falada; não inventa atividade | `transcript_ready`, `render_complete visible=true` | | |
| 7 | Rearm | Esperar após a resposta | Resposta continua visível; volta a ouvir | `rearm_listening`, `listening_started` | | |
| 8 | Barge-in | Fazer uma pergunta longa e falar por cima da resposta | TTS pára, a tua fala é captada | `[BARGE_IN] event=confirmed` | | |
| 9 | STOP | Durante uma resposta falada, carregar STOP | Fala pára de imediato, sem erro | `playback ... reason=` stop | | |
| 10 | Sleep | Dizer "podes ir dormir" | Volta a STANDBY, deixa de responder | `standby_entered` | | |
| 11 | Task | Escrever "Ajuda-me a comparar dois portáteis" e depois "o critério é a bateria" | Card da task aparece e a 2.ª mensagem continua a mesma task | `task_id` igual nos dois turnos | OK | Igual à corrida anterior: `TASK_START` e `TASK_CONTINUATION`, `llm_calls=0`, ~108 ms cada. 2.ª resposta "Percebi: a bateria conta na decisão. Falta perceber opções a comparar."; card "TAREFA Comparação pronta … PRIORIDADES a bateria pesa bastante na decisão AINDA FALTA opções a comparar". |
| 12 | Mudança de idioma | Escrever "responde em inglês" e depois "olá" | Passa a inglês e mantém; UI/task também | `response_language=en-GB` | OK | Igual: parte de pt-PT, "Sure. I'll respond in English.", `[UI_LANGUAGE] lang=en-GB`, o mesmo card passa a "TASK Comparison ready … compare options …", e "olá" dá "Hi. How are you?". |
| 13 | Ferramenta | "Que janela tenho aberta?" | Responde com dados reais (Context Observer), sem inventar | `tools_used=[...]` não vazio | OK | `TOOL_PATH`, `tools_used=['get_active_window']`, `llm_calls=0`, "Active window detected: ChatGPT" (a janela em primeiro plano). 1098 ms, antes 542 ms: mais lento, mas abaixo do limite e sem mudança no código deste caminho; provavelmente o varrimento do Context Observer em concorrência com o arranque do Pocket. |
| 14 | Shutdown | Fechar a janela | Fecha < ~2 s, sem crash nem processo pendurado | `[SHUTDOWN] event=completed` | **PARCIAL (regressão)** | A janela escondeu-se em 2 ms e não houve crash (código 0), mas o processo só saiu 3,0 s depois: `[SHUTDOWN] component=tts_prewarm event=finished elapsed_ms=2983`, `event=completed total_ms=2993` (antes 54 ms). Causa: o prewarm do Pocket (`9f86f03`) ainda estava a arrancar quando a bateria fechou a app (servidor 14:05:33 → 14:05:47, 14 s nesta corrida contra ~5 s isolado), e o shutdown espera por ele. Só acontece se a app fechar durante o arranque do Pocket; a janela não fica "Não responde". |
| 15 | Pedido vago | Logo a seguir ao 3 (sem task aberta): "Marca isso" | Pergunta o que marcar, sem LLM e sem afirmar que o fez | `selected_path=VAGUE_ACTION_CLARIFICATION`, `llm_calls=0` | OK | Novo (commits `f6c2634`/`c51fbc4`). `VAGUE_ACTION_CLARIFICATION`/`FAST_ROUTE`, `llm_calls=0`, 112 ms, "O que queres que marque?". Corre antes do 11 de propósito: com uma task aberta, a Presence trata-a como antecedente e não pede pergunta. Corrido com uma cópia do driver com este passo a mais (o `scripts/battery_driver.py` não foi alterado). |

## Observações gerais (latência, coisas estranhas, sensação)

-
