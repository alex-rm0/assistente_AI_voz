# AssistenteIA - Desenvolvimento

## Ambiente Python no Windows

Versao recomendada: Python 3.11 de 64 bits instalado a partir de <https://www.python.org/downloads/windows/>.

Evita usar o Python da Microsoft Store para este projeto. Se `.\.venv\Scripts\python.exe` mostrar erros com `WindowsApps`, apaga e recria o ambiente virtual.

Verificar instalacoes disponiveis:

```powershell
py -0p
```

Recriar o ambiente virtual:

```powershell
cd C:\Users\alexm\.vscode\projects\assistenteIA
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Se nao tiveres Python 3.11 no `py -0p`, instala-o pelo instalador oficial e marca a opcao para adicionar o Python ao launcher.

## Executar a aplicacao

Garante que o Ollama esta aberto e que o modelo configurado existe:

```powershell
ollama list
```

Arrancar a aplicacao:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Arrancar a nova interface Echo OS com consola em UTF-8:

```powershell
$env:PYTHONUTF8="1"
chcp 65001
python app.py --ui echo-os
```

Se aparecerem textos como `NÃ£o` ou `EstratÃ©gias` na consola, confirma primeiro
com os logs `[UTF8 DEBUG]`: a consola do Windows pode mostrar UTF-8 como ANSI,
mesmo quando as strings internas continuam corretas.

### Modelo e provider

Por defeito, o Echo usa Ollama local com `llama3.1:8b`.

A configuracao principal fica em `config/settings.json`:

```json
"model": {
  "provider": "ollama",
  "name": "llama3.1:8b"
}
```

A configuracao antiga `ollama.model` continua suportada por compatibilidade.

Prioridade de escolha:

```text
argumentos CLI > variaveis de ambiente > settings.json > defaults
```

Exemplos:

```powershell
python app.py --ui echo-os --provider ollama --model llama3.1:8b
```

```powershell
$env:ECHO_MODEL_PROVIDER="ollama"
$env:ECHO_MODEL_NAME="llama3.1:8b"
python app.py
```

Anthropic esta preparado, mas protegido para evitar custos acidentais:

```powershell
$env:ECHO_MODEL_PROVIDER="anthropic"
$env:ECHO_MODEL_NAME="<model-id>"
$env:ANTHROPIC_API_KEY="<chave>"
$env:ECHO_ALLOW_PAID_MODEL_CALLS="true"
python app.py --ui echo-os --provider anthropic --model <model-id>
```

Sem `ANTHROPIC_API_KEY`, o Echo mostra um erro claro. Sem
`ECHO_ALLOW_PAID_MODEL_CALLS=true`, nenhuma chamada Anthropic e executada.
Nunca coloques a chave em `settings.json`.

No arranque, a consola mostra:

```text
[MODEL CONFIG]
provider=ollama
provider_source=settings.json
model=llama3.1:8b
model_source=settings.json
```

Telemetria de turnos regista provider, modelo, latencia, tokens e custo
estimado quando o provider devolver essa informacao.

## Correr testes

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

Testar apenas o router rapido:

```powershell
python -m pytest tests\test_fast_router.py tests\test_fast_router_integration.py
```

## Debug de performance

Em `config/settings.json`, ativa:

```json
"DEBUG_PERFORMANCE": true
```

Os logs mostram:

```text
[AssistenteIA PERF] pedido recebido: 0.0 ms
[AssistenteIA PERF] router rapido: 2.1 ms
[AssistenteIA PERF] chamada Ollama /api/chat: 15432.7 ms
[AssistenteIA PERF] chamada Ollama /api/embeddings: 820.4 ms
[AssistenteIA PERF] resposta total: 15480.2 ms
```

Para comandos resolvidos pelo router rapido nao deve aparecer chamada a `/api/chat` nem a `/api/embeddings`.

## Testar microfone

A configuracao de voz fica em `config/settings.json`:

```json
"voice": {
  "enabled": true,
  "input_device": "default",
  "auto_select_input": true,
  "silent_rms_threshold": 0.001,
  "sample_rate": 44100,
  "channels": 1,
  "probe_duration_seconds": 0.5,
  "min_record_seconds": 2,
  "preroll_ms": 500,
  "ready_delay_ms": 200
}
```

Opcoes principais:

- `input_device`: `"default"`, indice numerico como `"1"`, ou parte do nome do microfone.
- `auto_select_input`: se `true`, o AssistenteIA testa outros inputs quando o default parece silencioso.
- `silent_rms_threshold`: limite abaixo do qual o microfone parece silencioso.
- `sample_rate`, `channels`, `probe_duration_seconds`: parametros usados no teste e na gravacao.
- `min_record_seconds`: duracao minima da gravacao antes de enviar para o Whisper.
- `preroll_ms`: pequeno buffer inicial preservado para evitar cortar as primeiras palavras.
- `ready_delay_ms`: tempo de aquecimento depois de abrir o stream antes de mostrar `A ouvir`.

Na interface, usa o botao **Testar microfone**. O resultado mostra:

```text
Microfone usado: ...
Indice: ...
Nivel RMS: ...
Duracao gravada/testada: ...
Sample rate: ...
Canais: ...
Estado: silencioso/baixo/normal/saturado
```

Se o microfone predefinido estiver silencioso e `auto_select_input=true`, o AssistenteIA sugere o input com melhor sinal.

Durante a gravacao pelo botao **Mic**, o ultimo audio captado fica guardado em:

```text
data/debug/last_voice_input.wav
```

Isto permite confirmar se o problema esta na captacao ou na transcricao. Podes pedir no chat:

```text
reproduz ultimo audio
estado da voz
```

O comando `estado da voz` mostra `voice.enabled`, dispositivo configurado, escolha automatica, ffmpeg, Whisper, modelo, idioma, sample rate e estado do microfone.

Ao carregar em **Mic**, espera pelo estado `A ouvir...` antes de falar. A app mostra primeiro `Preparar...` enquanto abre o stream do microfone. Mesmo assim, conserva um pequeno pre-roll para apanhar o arranque da fala caso comeces ligeiramente cedo.

### Troubleshooting de voz

- Microfone errado: altera `voice.input_device` para o indice numerico ou parte do nome do microfone.
- Volume baixo: aumenta o ganho do microfone no Windows e confirma se o RMS aparece como `baixo`.
- Ruido/saturacao: se o estado aparecer como `saturado`, reduz o ganho ou afasta o microfone.
- ffmpeg: se aparecer `ffmpeg: em falta`, instala o ffmpeg e garante que fica no `PATH`.
- Idioma errado: mantem `voice.language` como `"pt"`. O Whisper recebe tambem um prompt inicial para portugues de Portugal.
- Gravacao curta: se falares menos do que `voice.min_record_seconds`, a app avisa e nao envia para transcricao util.

## Comandos rapidos

Estes comandos devem ser resolvidos sem LLM:

```text
abre o youtube
abrir youtube
abre o google
abre https://www.google.com
abre www.youtube.com
limpar conversa
testar microfone
```

### Sites rapidos configuraveis

Os atalhos de sites ficam em `config/quick_sites.json`.

Exemplo:

```json
{
  "quick_sites": {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "chatgpt": "https://chatgpt.com",
    "github": "https://github.com"
  }
}
```

Depois de alterares este ficheiro, reinicia a aplicacao e usa:

```text
abre gmail
abre o chatgpt
abre github
```

Usa apenas URLs `http://` ou `https://`. Se um atalho tiver `file://`, `javascript:`, `data:` ou apontar para um executavel, o router rapido deve recusar.

### Pesquisa rapida

O router rapido tambem suporta pesquisas no Google e YouTube sem chamar o LLM:

```text
pesquisa no google por gatos british shorthair
procura no google por temperaturas historicas em Coimbra
pesquisar no google comandos powershell basicos
pesquisa no youtube por tutorial python tkinter
procura no youtube por musica lo-fi
pesquisar no youtube como instalar python 3.11 windows
```

A pesquisa apenas abre uma URL segura depois de confirmacao:

```text
Google:  https://www.google.com/search?q=<termo>
YouTube: https://www.youtube.com/results?search_query=<termo>
```

Termos perigosos dentro da pesquisa sao tratados como texto. Por exemplo, `pesquisa no google por file:///C:/Windows/System32/cmd.exe` pesquisa esse texto no Google; nao abre `file://` nem executa nada.

Para URLs, o AssistenteIA pede confirmacao antes de abrir. Exemplo:

```text
Utilizador: abre o youtube
AssistenteIA: Queres que abra este URL? https://www.youtube.com
Utilizador: sim
AssistenteIA: Abri o URL: https://www.youtube.com
```

## Pedidos que continuam a ir para o LLM

Estes pedidos exigem interpretacao ou geracao:

```text
Ajuda-me a melhorar a arquitetura do AssistenteIA.
Resume este texto por pontos.
Explica-me como organizar o projeto.
Que abordagem sugeres para reduzir latencia?
```

## Notas de seguranca

O router rapido nao executa comandos de terminal.

Pedidos como estes devem ser recusados:

```text
executa dir
corre powershell
abre o terminal
abre file:///C:/Windows/System32/cmd.exe
abre javascript:alert(1)
abre programa.exe
abre ../ficheiro
apaga este ficheiro
move esta pasta
```

As Desktop Actions permitidas continuam limitadas a ferramentas registadas e com confirmacao quando aplicavel. Ficheiros e pastas so devem ser abertos dentro da workspace ou de projetos conhecidos.

## Pesquisa no Echo OS

Pedidos explicitos de pesquisa entram no caminho `RESEARCH_REQUEST` antes da
conversa geral:

```text
pesquisa sobre Picasso
quero que facas uma pesquisa sobre Picasso
procura informacao sobre Picasso
pesquisa na internet sobre Picasso
encontra fontes sobre Picasso
verifica online sobre Picasso
```

Se existir uma ferramenta real registada (`web_search` ou `research_web`), o
Echo executa-a e envia eventos semanticos para a nova interface:

```text
research_started
research_results_ready
research_completed
```

Se nao existir ferramenta real, responde de forma honesta:

```text
Ainda nao tenho uma ferramenta de pesquisa ligada.
```

Nesse caso pode emitir `research_failed`, mas nao mostra cartoes falsos nem
promete pesquisar mais tarde.

## Desktop Actions

A configuracao fica em `config/settings.json`:

```json
"desktop_actions": {
  "enabled": true,
  "default_browser": "chrome",
  "default_email": "gmail",
  "known_projects": {
    "assistenteIA": "C:/Users/alexm/.vscode/projects/assistenteIA"
  }
}
```

Aliases naturais suportados:

```text
abre o mail
abre o gmail
abre o browser
abre o codigo
abre o projeto assistente
abre os documentos
abre downloads
```

Todas as acoes reais pedem confirmacao. Se uma app ja estiver aberta, o agente
consulta o Context Observer e pergunta se queres traze-la para a frente. Pastas
fora da workspace continuam bloqueadas, exceto aliases explicitos para
`Documents` e `Downloads`. Nunca uses Desktop Actions para executar comandos
shell arbitrarios.
