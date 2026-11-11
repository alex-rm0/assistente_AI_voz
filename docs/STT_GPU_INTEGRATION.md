# STT GPU experimental na Echo OS

O VoiceTranscriber aceita `ECHO_STT_PROVIDER=faster-whisper` para usar
large-v3-turbo CUDA float16. Sem essa variável continua a usar
openai-whisper e o modelo existente em settings.json. Não se alterou
config/settings.json, a .venv da app, VAD, captura, wake word ou TTS.

## Testar no worktree paralelo

No PowerShell, a partir de assistenteIA:

```powershell
$env:ECHO_STT_PROVIDER = 'faster-whisper'
$env:ECHO_STT_PYTHON = "$PWD\.venv_stt_bench\Scripts\python.exe"
$env:HF_HOME = "$PWD\.hf_cache"
.\.venv\Scripts\python.exe ..\assistenteIA-par\app.py --ui echo-os --provider ollama --model-mode local
```

Esta é a app do worktree paralelo: usa a configuração e os dados desse
worktree, não o histórico pessoal do worktree principal. Não executar
simultaneamente duas apps a captar o mesmo microfone. A instalação CUDA
já existente é reutilizada; não instalar faster-whisper/torch CUDA na .venv
da app. O launcher habitual de main ainda não contém esta integração.

Para voltar ao backend anterior, fechar a app, executar
`$env:ECHO_STT_PROVIDER = 'openai-whisper'` e voltar a iniciar.
As variáveis acima só afetam processos iniciados a partir deste PowerShell.
`ECHO_STT_MODEL` permite escolher outro modelo compatível explicitamente;
o default deste backend é `large-v3-turbo`.

## Comportamento

- Modelo carregado à primeira transcrição, num processo persistente isolado.
  Não existe import de faster-whisper/torch CUDA no processo da app.
- A transcrição continua no worker QThread já existente. A primeira frase
  paga o carregamento; frases seguintes reutilizam o mesmo modelo.
- Idioma e initial_prompt seguem o VoiceTranscriber existente. Idioma de
  entrada continua independente do idioma da resposta e da voz TTS.
- Confiança por segmento passa pelos mesmos filtros de silêncio, baixa
  confiança e repetição. Não existe correção automática de comandos.
- Não há fallback CPU silencioso. Runtime ausente, erro CUDA, crash ou
  timeout produzem erro de voz; o utilizador pode selecionar o backend
  anterior explicitamente e reiniciar.
- Logs `[STT_BACKEND]` identificam provider, modelo, CUDA e latência, sem
  texto transcrito. Encerrar a Echo OS fecha o subprocesso STT.

## Validação e limites

Teste real integrado com as gravações existentes: duas frases aceites,
dois silêncios rejeitados, WAVs preservados e subprocesso encerrado.
Primeira frase ~2.68 s com load; seguinte ~184 ms, incluindo diagnósticos.
Testes unitários usam subprocessos/modelos falsos, não descarregam modelos.

A validação com Qt/microfone ao vivo continua necessária. O erro
«Abre o Pixel» -> «Abro pixel» não foi mascarado. O replay também revelou
cortes prematuros do VAD em pausas e voz baixa; os thresholds não foram
alterados nesta integração. Estes resultados não provam precisão perfeita
nem latência ponta a ponta de conversa contínua.
