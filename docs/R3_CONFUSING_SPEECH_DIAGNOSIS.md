# R3 passo 2 — diagnóstico e proposta

Data: 2026-11-12. **Só diagnóstico. A regra abaixo não está implementada nem aprovada.**
Referências: [North Star](ECHO_NORTH_STAR.md) e [plano acordado](R3_PLAN.md).
Não foram alterados prompts, routing, voz, UI ou memória de produção nesta ronda.

## Sinais existentes e fronteira onde se perdem

- `assistant/stt_runner.py` devolve, por segmento, `avg_logprob`, `no_speech_prob`
  e `compression_ratio` do faster-whisper. O caminho openai-whisper também fornece segmentos.
- `assistant/voice_input.py::_whisper_confidence_signals()` calcula média simples dos
  `avg_logprob`, máximo de `no_speech_prob` e máximo de `compression_ratio`. Não há
  ponderação pela duração dos segmentos. `no_speech_prob` torna-se `no_speech_probability`.
- `decide_stt_acceptance()` já rejeita silêncio, texto vazio, `no_speech_probability >= 0.75`,
  `avg_logprob <= -1.25`, `compression_ratio >= 3.2` e repetição excessiva.
- `_log_stt_decision()` escreve `[VOICE_STT]` com esses valores, RMS, silêncio e decisão.
  O `VoiceSttDecision` é local: não existe um snapshot de confiança associado ao turno do engine.
- `transcribe()` devolve **só texto**. `EchoVoiceWorker.transcriptionReady = Signal(str)`
  e `_handle_voice_transcription()` transportam apenas esse texto até `submitMessage()`.
  O `AssistantEngine.respond()` não recebe confiança nem origem STT por turno.
- Uma rejeição por confiança baixa usa hoje `VoiceNoSpeechDetected`, tal como silêncio.
  O worker emite texto vazio; o controller volta a ouvir sem criar uma resposta conversacional.
  Silêncio e fala incompreensível precisam de continuar a ser decisões distintas.

Não se deve ler uma variável global ou o último resultado mutável do transcritor no engine:
warmup, wake, barge-in e uma captura seguinte podem fazê-lo pertencer a outro turno.
Uma integração futura precisa de metadados imutáveis por captura, com sessão/modo/identidade,
antes da decisão de clareza. Texto escrito não deve herdar confiança da última gravação.

## Exemplos observados

`output.txt` de 12/11, analisado apenas para diagnóstico; não é versionado:

| Texto do turno | avg_logprob | no_speech_probability | Consequência |
|---|---:|---:|---|
| «Boa, boa. Queria escrever o mail contigo. Ajudas-me?» | -0.30688 | 0.00000 | Pedido válido; proteger o intent de escrita. |
| Resposta com os dois nomes dos destinatários | -0.74510 | 0.00000 | Campo válido apesar da confiança menor e de não ser uma frase completa. |
| «Podes escrever o primeiro rascunho, mas o nome da Karina está mal.» | -0.22159 | 0.00000 | Não é STT confuso; era uma falha de workflow, já corrigida. |
| «Consegues trocar o nome da Karina e o K por um C?» | -0.27405 | 0.00000 | Linguagem informal válida; não pedir repetição. |
| «Primeiro guarda no workspace, não fecheira TXT o email, por favor.» | -0.49697 | 0.00000 | Há corrupção fonética, mas existe intenção operacional reconhecível; não rejeitar só pelo vocabulário. |

Num log anterior anexado, «Tia, como esta?» teve `avg_logprob=-0.80690` e
`no_speech_probability=0.00000`: pode haver palavra mal ouvida sem falta de fala.
Não é possível deduzir a intenção original apenas destes números.

«Querias que eu ver um mal contigo, mas jostas-me.» está preservada no caso
`log_garbled_mal_contigo_en`, marcado `source=real_log`. Não recuperámos confiança
associada a essa captura nos logs analisados; **não lhe atribuímos valores inventados**.
Os restantes fragmentos novos da avaliação são `source=authored`, não gravações reais.

`avg_logprob` não é uma probabilidade calibrada de compreensão. `no_speech_probability`
estima ausência de fala, não coerência semântica. Um corte isolado em -0.6 rejeitaria
os nomes válidos acima. Valores do tiny de wake não devem usar os limiares do turbo de conversa.

## Proposta determinística para validação, não implementação

1. Manter segurança, comandos de controlo e intenções canónicas reconhecidas protegidos.
   Uma resposta válida ao campo atual (nomes, datas, números, «sim»/«não») também fica protegida.
   Ser curta, informal, ter um nome desconhecido ou uma gralha não é razão suficiente para rejeitar.
2. Calcular sinais de texto num módulo próprio, por exemplo `assistant/speech_clarity.py`:
   truncagem que acaba numa ligação incompleta, várias ligações gramaticais sem conteúdo
   recuperável e combinações claramente quebradas. Não usar lista negra de palavras ou
   de exemplos da suite, nem tratar toda a gramática imperfeita como incompreensível.
   Texto citado, perguntas sobre gramática e pausas naturais exigem controlos negativos.
3. Sem intent/campo reconhecido, pedir repetição quando existirem **dois sinais fortes de
   texto quebrado**, ou **um sinal forte mais confiança acústica suspeita**.
   Candidatos para experimentar em dados rotulados: `avg_logprob < -0.9` ou
   `no_speech_probability > 0.5`. São hipóteses de calibração, não limiares decididos.
   Dados de confiança ausentes não equivalem a confiança baixa.
4. Se não houver sinais suficientes, não rejeitar automaticamente. Um erro confiante do
   STT pode escapar a esta regra: não prometemos zero invenções com os dados atuais.
5. Quando a regra disparar, responder diretamente, sem LLM/embeddings/Agent nem ações:
   - en-GB: «Sorry, I didn't quite catch that — could you say it again?»
   - pt-PT: «Desculpa, não percebi bem — podes repetir?»
   O idioma vem de `response_language`, não do idioma ouvido. Não consumir campos da task
   nem guardar o texto confuso como facto. Silêncio mantém o rearme silencioso atual.

Antes de integrar: validar em pares áudio/transcrição revistos pelo Alexandre, incluindo
nomes, termos técnicos, correções, datas, exportações, sotaque, pausas e trocas de idioma.
Medir falsas rejeições separadamente das invenções. A suite textual não tem confiança STT:
não permite concluir como se comportará o ramo acústico nem inventar esses metadados.

## Baseline do passo 1

- Dois runs locais sequenciais com `llama3.1:8b`, depois de fechar o jogo e verificar a GPU.
- 41 casos / 57 turnos por corrida; zero exceções. Nenhuma chamada Anthropic.
- Com as verificações finais, cada corrida pediu clarificação em **2 de 10** turnos confusos.
- Factos do mundo sem invenções detetadas: **93.0% / 94.7%**; estado de sistema verificado:
  **91.1% / 92.9%**; sem repetição: **100% / 94.7%**. Idioma e segunda pessoa: **100%**.
- Nos mesmos 35 casos antigos, o agregado `no_invented_state` anterior de 96.9% baixa para
  **75.0% / 77.8%** ao reavaliar as respostas antigas com as regras novas. As respostas atuais
  dão **77.1% / 80.6%**. Não é evidência de melhoria causal: há variação do modelo e correções
  de runtime entre as rondas. Há muitos itens em revisão humana; não os contamos como sucesso.
- Os reports originais foram preservados. As respostas novas também foram reapreciadas offline
  depois de fechar falsos negativos do avaliador; essa reapreciação fez zero chamadas de modelo.
- Reports locais ignorados: `evals/results/conversation/r3_step1_run1_revalidated/`,
  `r3_step1_run2_revalidated/` e `r3_step1_comparison.md`. As respostas brutas das duas novas
  corridas estão em `r3_step1_llama3.1-8b_run1/` e `r3_step1_llama3.1-8b_run2/`.
- Validação final: 2274 testes passaram, 1 xfailed; bateria Qt 1–3/11–14 passou.
  O backup de `data/` foi reposto, incluindo o perfil local, com os 52 hashes iguais.

**Próxima decisão:** validar esta proposta e os critérios de falsas rejeições antes de escrever
a regra de produção. O passo 2 não está concluído funcionalmente.
