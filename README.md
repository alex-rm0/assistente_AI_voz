# AssistenteIA

AssistenteIA e uma aplicacao desktop local para Windows 11 feita em Python, PySide6 e Ollama.

## Documentos centrais

- [MANIFESTO.md](docs/MANIFESTO.md): identidade e missão do Echo.
- [PRODUCT_PRINCIPLES.md](docs/PRODUCT_PRINCIPLES.md): critérios de decisão do produto.
- [COGNITIVE_MODEL.md](docs/COGNITIVE_MODEL.md): teoria central de como o Echo pensa.
- [COGNITIVE_ARCHITECTURE.md](docs/COGNITIVE_ARCHITECTURE.md): componentes que implementam esse pensamento.
- [PERSONAL_MODEL.md](docs/PERSONAL_MODEL.md): o que o Echo conhece sobre o Alexandre.
- [VOICE_AND_CONVERSATION.md](docs/VOICE_AND_CONVERSATION.md): como o Echo comunica.
- [UI_PHILOSOPHY.md](docs/UI_PHILOSOPHY.md): filosofia da interface como presença.
- [ROADMAP.md](docs/ROADMAP.md): ordem de implementação por capacidades cognitivas.
- [architecture.md](docs/architecture.md): arquitetura técnica atual.

## Funcionalidades atuais

- Janela desktop com conversa, caixa de texto e botao Enviar.
- Interface Echo OS experimental com workspace adaptativo inicial para pesquisa.
- Conversa normal com um modelo local via Ollama.
- Runtime baseado em providers, com Ollama por defeito e Anthropic preparado de forma opcional e protegida.
- Modelo configuravel em `config/settings.json`, variaveis de ambiente ou argumentos CLI.
- Agent Loop para decidir entre resposta direta e uso de uma ou varias ferramentas.
- Presence Manager com estados globais de funcionamento.
- Context Observer passivo para registar aplicacao ativa, janela ativa, ficheiros recentes, projeto aberto e tempo por atividade.
- Context Interpreter para transformar snapshots tecnicos em resumo humano util.
- Context Reasoning para transformar observacoes em conclusoes suportadas por evidencias.
- Context Manager automatico com multiplos contextos ativos por mensagem.
- Planner para transformar mensagem, contexto, memoria, tarefas e ferramentas num plano antes da resposta.
- Session Manager para resumir sessoes de trabalho e recuperar continuidade entre aberturas da aplicacao.
- Sistema de delegacao para escolher entre resolver localmente, preparar contexto para ChatGPT/Codex ou sugerir ferramenta externa.
- Desktop Actions seguras com confirmacao para abrir apps, URLs, ficheiros, pastas e projetos conhecidos.
- Deteccao deterministica de pedidos de pesquisa (`RESEARCH_REQUEST`) com resposta honesta quando nao existe ferramenta real ligada.
- Tool Registry com ferramentas registadas automaticamente.
- Ferramentas de estado do computador baseadas no Context Observer.
- Listagem de ficheiros dentro de `workspace`.
- Leitura segura de ficheiros `.txt` e `.md` dentro de `workspace`.
- Leitura segura de documentos `.docx` e `.pdf` dentro de `workspace`.
- Criacao segura de ficheiros `.txt` dentro de `workspace`, sem sobrescrever.
- Resumo de ficheiros pequenos usando Ollama.
- Identificacao automatica de contextos: pessoal, trabalho, tecnico, produtividade, viagens e social.
- Painel opcional de debug de contextos quando `DEBUG_AGENT=true`.
- Memoria de conversa em `data/history.json`.
- Memoria permanente em SQLite em `data/long_term_memory.sqlite`.
- Personal Model em SQLite em `data/personal_model.sqlite`, com categorias, evidencias e niveis de confianca.
- Sistema de tarefas e lembretes guardado na memoria permanente.
- Voice Input com Whisper local, teste de microfone e diagnostico de audio.
- Comandos de memoria permanente:
  - `lembra-te que...`
  - `guarda isto...`
  - `nao te esquecas que...`
  - `esquece...`
  - `o que sabes sobre...`
- Botao `Limpar conversa`.

## Estrutura

```text
app.py
requirements.txt
config/
  settings.json
assistant/
  conversation.py
  context_manager.py
  agent.py
  session_manager.py
  presence_manager.py
  context_observer.py
  delegation.py
  llm.py
  long_term_memory.py
  memory.py
  security.py
  tools.py
  tool_registry.py
ui/
  main_window.py
data/
workspace/
docs/
  MANIFESTO.md
  PRODUCT_PRINCIPLES.md
  COGNITIVE_MODEL.md
  COGNITIVE_ARCHITECTURE.md
  VOICE_AND_CONVERSATION.md
  UI_PHILOSOPHY.md
  PERSONAL_MODEL.md
  ROADMAP.md
  VISION.md
  architecture.md
```

## Requisitos

- Windows 11
- Python 3.11 ou superior
- Ollama instalado e em execucao
- Modelo local instalado, por defeito `llama3.2`

## Instalacao

Na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se a `.venv` der erro como `No Python at ...`, recria o ambiente:

```powershell
Remove-Item .venv -Recurse -Force
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativacao:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Preparar o Ollama

Instala o modelo configurado:

```powershell
ollama pull llama3.2
```

Confirma que o Ollama esta ativo:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11434 -Method Get
```

Deve responder:

```text
Ollama is running
```

## Executar

Com o ambiente virtual ativo:

```powershell
python app.py
```

Ou diretamente:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Testes

Depois de instalar os requirements:

```powershell
pytest
```

Ou:

```powershell
python -m pytest tests
```

## Exemplos de uso

```text
lista a pasta
le teste.txt
le o ficheiro exemplo.docx
resume o ficheiro teste.txt
resume o ficheiro relatorio.pdf
cria um ficheiro chamado nota.txt com este texto Isto e uma nota.
lista os ficheiros da workspace e resume o primeiro
le o ficheiro exemplo.txt e cria uma nota com os pontos principais
procura documentos sobre RVCC e diz-me quais parecem relevantes
analisa os ficheiros existentes e sugere uma organizacao
lembra-te que prefiro respostas curtas
lembra-me de terminar o relatorio
lembra-me disto amanha
o que tenho para fazer hoje?
o que sabes sobre respostas?
qual e o meu nome?
```

## Seguranca

As ferramentas de ficheiros so trabalham dentro da pasta `workspace`.

Bloqueado nesta fase:

- apagar ficheiros;
- mover ficheiros;
- executar comandos;
- aceder fora da `workspace`;
- sobrescrever ficheiros existentes.

Antes de criar ficheiros, o Agent Loop pede confirmacao. Responde `sim` para criar ou `nao` para cancelar.

## Configuracao

Edita `config/settings.json` para alterar:

- nome da aplicacao;
- URL local do Ollama;
- modelo;
- pasta `workspace`;
- ficheiros de memoria.
- estado inicial em `default_presence`;
- logs simples de debug em `debug`.
- debug do Agent Loop e dos contextos em `DEBUG_AGENT`.
- debug bruto do Context Observer em `DEBUG_CONTEXT`.

Configuracao de voz:

```json
"voice": {
  "enabled": true,
  "model": "base",
  "language": "pt",
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

Na interface, usa **Testar microfone** para ver microfone usado, nivel RMS,
duracao testada, sample rate e se o audio parece silencioso, baixo, normal ou
saturado. O botao **Mic** grava, transcreve com Whisper local e coloca o texto
na caixa de mensagem para revisao; nao envia automaticamente.

Quando carregas em **Mic**, a app mostra primeiro `Preparar...` e so depois
`A ouvir...`. Comeca a falar quando vires `A ouvir...`; existe tambem um pequeno
pre-roll configuravel para evitar cortar as primeiras palavras.

O ultimo audio gravado fica em `data/debug/last_voice_input.wav`. Para o ouvir,
escreve:

```text
reproduz ultimo audio
```

Para diagnostico geral:

```text
estado da voz
```

Troubleshooting rapido:

- se o microfone estiver errado, altera `voice.input_device`;
- se o RMS estiver baixo, aumenta o ganho do microfone no Windows;
- se aparecer saturado, reduz o ganho ou afasta o microfone;
- se `ffmpeg` estiver em falta, instala-o e confirma que esta no `PATH`;
- se o idioma sair errado, confirma `voice.language = "pt"`.

Configuracao de Desktop Actions:

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

Exemplos:

```text
abre o mail
abre o gmail
abre o browser
abre o codigo
abre o projeto assistente
abre os documentos
abre downloads
abre https://www.google.com
```

Antes de abrir algo, o AssistenteIA pede confirmacao. Se uma aplicacao ja
estiver aberta e for detetada pelo Context Observer, responde que ja esta aberta
e pergunta se queres traze-la para a frente. As acoes executadas ficam registadas
na timeline. O AssistenteIA nunca executa comandos shell arbitrarios.

## Contextos automaticos

O AssistenteIA ja nao usa perfis manuais escolhidos pelo utilizador. Em cada mensagem, o Context Manager identifica automaticamente um ou varios contextos relevantes:

- `PERSONAL_CONTEXT`
- `WORK_CONTEXT`
- `TECH_CONTEXT`
- `PRODUCTIVITY_CONTEXT`
- `TRAVEL_CONTEXT`
- `SOCIAL_CONTEXT`

Cada contexto tem descricao, memoria associada e peso de relevancia. O Agent Loop recebe estes contextos antes de gerar qualquer resposta.

Exemplos:

```text
Ajuda-me a planear ferias para a Australia
```

Contextos esperados:

```text
TRAVEL_CONTEXT
PERSONAL_CONTEXT
```

```text
Tenho um erro neste projeto Python
```

Contextos esperados:

```text
TECH_CONTEXT
WORK_CONTEXT
```

Configuracao do Context Observer:

- `context_observer.enabled`;
- `context_observer.interval_seconds`;
- `context_observer.recent_files_limit`;
- `context_observer.summary_min_seconds`;
- `context_observer.db_file`.

## Estados de presenca

- `ACTIVE_CONVERSATION`: responde, usa ferramentas e pode pedir confirmacoes.
- `PASSIVE_MONITORING`: nao responde; reservado para acompanhamento passivo e registo futuro de atividade.
- `FOCUS_MODE`: semelhante ao modo passivo, pensado para reduzir interrupcoes.
- `PRIVATE_MODE`: responde sem gravar memoria, observar atividade ou indexar contexto.
- `OFFLINE`: tudo desligado.

## Context Observer

O Context Observer e passivo. Nao executa comandos, nao abre ficheiros, nao altera ficheiros e nao interrompe o utilizador.

O Context Interpreter transforma o snapshot bruto num resumo humano, agrupando
aplicacoes por desenvolvimento, comunicacao, produtividade, navegacao e sistema.
Por defeito ignora ruido tecnico como `Program Manager`, `TextInputHost`,
`ApplicationFrameHost` e processos sem janela util.

O interpretador enriquece o snapshot, mas nao substitui os dados observados. A
ferramenta `get_last_context_snapshot` mostra o resumo interpretado e tambem um
bloco `Snapshot bruto`. As ferramentas `get_open_windows` e `get_active_window`
continuam a devolver diretamente a lista real de janelas e a janela ativa.

Se `DEBUG_CONTEXT=true`, a ferramenta `get_last_context_snapshot` inclui tambem
um bloco `[DEBUG_CONTEXT]` com informacao bruta resumida.

Perguntas sobre janelas, aplicacoes abertas, programa ativo ou atividade recente
devem ser respondidas apenas com dados do Context Observer. Se nao houver dados,
a resposta correta e:

```text
Consigo tentar observar o computador, mas ainda não tenho dados suficientes. Experimenta mudar de janela ou aguardar alguns segundos.
```

Ferramentas ligadas ao Context Observer:

- `get_active_window`
- `get_active_application`
- `get_open_windows`
- `get_recent_activity`
- `get_last_context_snapshot`
- `get_current_activity_summary`

Na primeira versao regista em `data/context_observer.sqlite`:

- aplicacao ativa;
- janela ativa;
- janelas abertas, quando disponiveis;
- processos ativos;
- sessoes VSCode detetadas;
- repositorios Git no projeto;
- ficheiros recentemente utilizados no Windows;
- ficheiros recentemente modificados no projeto;
- projeto atualmente aberto, quando consegue inferir pelo titulo da janela;
- tempo passado em cada atividade.

A memoria permanente nao recebe todos os eventos. O observer agrega sessoes e guarda apenas resumos uteis, por exemplo:

```text
Entre as 09:00 e as 11:00 o Alexandre trabalhou no projeto AssistenteIA usando VSCode e Git.
```

A observacao so corre quando o estado de presenca permite observar atividade, por exemplo em `PASSIVE_MONITORING` ou `FOCUS_MODE`. Em `PRIVATE_MODE` e `OFFLINE`, nao observa.

## Delegacao

O AssistenteIA pode decidir nao resolver tudo localmente. Quando o pedido for mais adequado para outro destino, prepara contexto em vez de executar a acao.

Estrategias possiveis:

- resolver localmente;
- preparar prompt para ChatGPT;
- preparar prompt para Codex;
- preparar contexto para uma ferramenta externa.

Exemplo:

```text
Este pedido e melhor resolvido pelo Codex.
Vou preparar o contexto.
```

## Memoria permanente

A memoria permanente fica numa base SQLite local em `data/long_term_memory.sqlite` e e separada do historico da conversa.

Categorias usadas:

- `perfil_utilizador`
- `projetos`
- `conversas`
- `preferencias`
- `tarefas`
- `relacoes`

O assistente pesquisa a memoria permanente antes de responder, quando o estado de presenca permite gravar/usar memoria. A pesquisa usa embeddings do Ollama quando disponiveis e uma pesquisa textual simples como alternativa.

## Personal Model

O Personal Model fica em `data/personal_model.sqlite` e guarda conhecimento
estruturado sobre o Alexandre. Cada entrada tem categoria, chave, descricao,
confianca, evidencias, origem e estado.

Categorias iniciais:

- `identidade`
- `vida`
- `trabalho`
- `estudos`
- `projetos`
- `ferramentas`
- `preferencias`
- `habitos`
- `relacoes`
- `objetivos`

Comandos:

```text
lembra-te que prefiro mapas mentais
guarda isto: uso o VS Code para programar
nao te esquecas que estou a trabalhar em RVCC
o que sabes sobre mim?
o que sabes sobre os meus estudos?
o que sabes sobre as minhas ferramentas?
o que sabes sobre mim com detalhes?
esquece mapas mentais
corrige isto: prefiro respostas em portugues de Portugal
```

Conhecimento com confianca inferior a 60% e apresentado como hipotese. A partir
de 90%, e apresentado como conhecimento forte. O Echo nao deve transformar
hipoteses em factos sem evidencias suficientes.

## Timeline pessoal

A timeline pessoal regista eventos importantes com data, projeto e pessoas associadas.

Exemplos:

```text
Ontem estivemos a trabalhar no projeto AssistenteIA.
Na semana passada falamos sobre ferias na Australia.
Ha tres meses comecaste este projeto.
O que fizemos ontem?
Em que estavamos a trabalhar?
Quando comecamos este projeto?
```

## Continuidade de sessao

O Session Manager guarda resumos compactos das sessoes de trabalho em
`data/session_manager.sqlite`. Nao guarda tudo em bruto; tenta preservar apenas
factos uteis como projeto principal, atividade, ficheiros tocados, ferramentas
usadas, tarefas alteradas, decisoes tomadas e proximo passo sugerido. Ao fechar
uma sessao, o resumo util tambem e promovido para a memoria permanente/timeline.
As respostas ao utilizador passam por uma camada de reflexao para evitar logs
tecnicos e apresentar continuidade em linguagem natural.

Exemplos:

```text
Onde ficamos?
Resume a ultima sessao.
O que fizemos hoje?
O que mudou desde a ultima vez?
Qual e o proximo passo?
```

## Tarefas e lembretes

As tarefas ficam guardadas na base SQLite da memoria permanente e podem ter projeto e data associada.

Exemplos:

```text
Lembra-me de terminar o relatorio.
Lembra-me disto amanha.
O que tenho para fazer hoje?
Que tarefas tenho?
```

Nesta versao, os lembretes ficam registados e podem ser consultados, mas ainda nao disparam notificacoes automaticas.

## Notas de desenvolvimento

A arquitetura esta documentada em `docs/architecture.md`.

Para adicionar uma ferramenta futura, regista uma funcao com `@tool_registry.register(...)`. O nucleo da aplicacao nao precisa de ser alterado.
