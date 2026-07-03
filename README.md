# AssistenteIA

AssistenteIA e uma aplicacao desktop local para Windows 11 feita em Python, PySide6 e Ollama.

## Funcionalidades atuais

- Janela desktop com conversa, caixa de texto e botao Enviar.
- Conversa normal com um modelo local via Ollama.
- Modelo configuravel em `config/settings.json`.
- Agent Loop para decidir entre resposta direta e uso de uma ou varias ferramentas.
- Presence Manager com estados globais de funcionamento.
- Context Observer passivo para registar aplicacao ativa, janela ativa, ficheiros recentes, projeto aberto e tempo por atividade.
- Context Interpreter para transformar snapshots tecnicos em resumo humano util.
- Context Reasoning para transformar observacoes em conclusoes suportadas por evidencias.
- Context Manager automatico com multiplos contextos ativos por mensagem.
- Sistema de delegacao para escolher entre resolver localmente, preparar contexto para ChatGPT/Codex ou sugerir ferramenta externa.
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
- Sistema de tarefas e lembretes guardado na memoria permanente.
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
