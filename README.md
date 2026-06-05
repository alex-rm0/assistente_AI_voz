# AssistenteIA

AssistenteIA e uma aplicacao desktop local para Windows 11 feita em Python, PySide6 e Ollama.

## Funcionalidades atuais

- Janela desktop com conversa, caixa de texto e botao Enviar.
- Conversa normal com um modelo local via Ollama.
- Modelo configuravel em `config/settings.json`.
- Tool Registry com ferramentas registadas automaticamente.
- Listagem de ficheiros dentro de `workspace`.
- Leitura segura de ficheiros `.txt` e `.md` dentro de `workspace`.
- Criacao segura de ficheiros `.txt` dentro de `workspace`, sem sobrescrever.
- Resumo de ficheiros pequenos usando Ollama.
- Memoria de conversa em `data/history.json`.
- Memoria permanente em SQLite em `data/long_term_memory.sqlite`.
- Comandos de memoria permanente:
  - `lembra-te que...`
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
resume o ficheiro teste.txt
cria um ficheiro chamado nota.txt com este texto Isto e uma nota.
lembra-te que prefiro respostas curtas
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

## Configuracao

Edita `config/settings.json` para alterar:

- nome da aplicacao;
- URL local do Ollama;
- modelo;
- pasta `workspace`;
- ficheiros de memoria.

## Notas de desenvolvimento

A arquitetura esta documentada em `docs/architecture.md`.

Para adicionar uma ferramenta futura, regista uma funcao com `@tool_registry.register(...)`. O nucleo da aplicacao nao precisa de ser alterado.
