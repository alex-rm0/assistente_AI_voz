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
apaga este ficheiro
move esta pasta
```

As Desktop Actions permitidas continuam limitadas a ferramentas registadas e com confirmacao quando aplicavel. Ficheiros e pastas so devem ser abertos dentro da workspace ou de projetos conhecidos.
