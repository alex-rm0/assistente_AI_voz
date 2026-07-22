# Echo Baseline

Data de actualização: 2026-07-22  
Commit actual: `3f6cca7`  
Provider operacional actual: `ollama`

Este documento regista o estado da baseline da Fase 0. Nesta tarefa foi corrigido o ambiente Python e a resolução do modelo, mas ainda não foi criada uma baseline oficial marcada porque existem alterações de código e documentação por commitar.

## 1. Ambiente Python

### Diagnóstico pedido

Executado antes da correcção:

| Comando | Resultado |
|---|---|
| `where.exe python` | Não encontrou `python` no PATH. |
| `python --version` | Falhou: `python` não reconhecido. |
| `python -c "import sys; print(sys.executable)"` | Falhou: `python` não reconhecido. |
| `python -m pip --version` | Falhou: `python` não reconhecido. |
| `python -m pytest --version` | Falhou: `python` não reconhecido. |

Repetido depois da correcção na mesma sessão PowerShell:

| Comando | Resultado |
|---|---|
| `where.exe python` | Continua sem encontrar `python` global no PATH da sessão. |
| `python --version` | Continua a falhar sem a `.venv` activada. |
| `python -m pip --version` | Continua a falhar sem a `.venv` activada. |
| `python -m pytest --version` | Continua a falhar sem a `.venv` activada. |

Isto é esperado nesta consola. A baseline reproduzível do projecto deve usar sempre `.\.venv\Scripts\python.exe` ou uma shell onde `.\.venv\Scripts\Activate.ps1` tenha sido activado com sucesso.

Diagnóstico directo da `.venv` antiga:

- `.venv\pyvenv.cfg` apontava para `C:\Users\alexm\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0`;
- `.venv\Scripts\python.exe` falhava com `No Python at ...WindowsApps...`;
- `pip.exe` e `pytest.exe` existiam, mas dependiam do interpretador quebrado;
- `py`, `winget`, `choco` e `uv` não estavam disponíveis no PATH.

Causa da inconsistência: uma consola podia mostrar `(.venv)` por ter activado os scripts, mas o executável real dentro da `.venv` apontava para uma instalação Python da Microsoft Store inexistente ou inacessível. O prompt activado não garantia que o interpretador funcionasse.

### Correcção feita

- Instalado Python oficial 3.11.9 no perfil do utilizador:
  `C:\Users\alexm\AppData\Local\Programs\Python\Python311\python.exe`
- `.venv` antiga renomeada para backup:
  `.venv.broken-msstore-20260722`
- `.venv` recriada com Python 3.11.9 real;
- `requirements.txt` instalado;
- `pytest` e `requests` confirmados.

Estado actual:

| Comando | Resultado |
|---|---|
| `.venv\Scripts\python.exe --version` | `Python 3.11.9` |
| `.venv\Scripts\python.exe -c "import sys; print(sys.executable)"` | `C:\Users\alexm\.vscode\projects\assistenteIA\.venv\Scripts\python.exe` |
| `.venv\Scripts\python.exe -m pip --version` | `pip 26.1.2` |
| `.venv\Scripts\python.exe -m pytest --version` | `pytest 8.4.2` |
| `requests` | `2.34.2` |

Nota: `Activate.ps1` pode continuar bloqueado pela Execution Policy. Nesse caso usar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Ou correr comandos directamente com:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 2. Resolução Do Modelo

Antes desta correcção:

- `config/settings.json` tinha `gemma3:12b`;
- o runtime usava `llama3.1:8b` por causa da constante/default em `assistant/llm.py`;
- `settings["ollama"]["model"]` não era respeitado por `app.py`.

Agora existe uma função única:

- `app.resolve_ollama_model(...)`

Prioridade:

1. argumento CLI `--model`;
2. `ECHO_MODEL_NAME`;
3. `OLLAMA_MODEL` como compatibilidade legacy;
4. `settings["ollama"]["model"]`;
5. default seguro `llama3.1:8b`.

Validação actual:

| Entrada | Resultado |
|---|---|
| sem env e sem CLI | `('gemma3:12b', 'settings.json')` |
| `ECHO_MODEL_NAME=llama3.1:8b` | `('llama3.1:8b', 'ECHO_MODEL_NAME')` |
| `--model cli-model` | `('cli-model', 'cli')` |

No arranque são impressos:

```text
model=<modelo>
model_source=<origem>
```

A telemetry do turno também inclui:

- `model`;
- `model_source`.

## 3. Testes

Executado com a `.venv` real:

```powershell
.\.venv\Scripts\python.exe -m compileall app.py assistant ui prototype_web_ui evals tests
.\.venv\Scripts\python.exe -m pytest
```

Resultados:

| Validação | Resultado |
|---|---|
| `compileall` | PASS |
| `pytest` | `497 passed in 20.95s` |
| `evals real_conversation` smoke | `10 passed, 0 failed, 0 exceptions` |

Novos testes adicionados:

- `tests/test_model_resolution.py`

Cobrem:

- CLI vence tudo;
- `ECHO_MODEL_NAME` vence legacy/settings;
- `OLLAMA_MODEL` funciona como legacy;
- settings é usado sem env;
- default é usado sem configuração;
- strings vazias são ignoradas;
- `model` e `model_source` aparecem na telemetry.

## 4. Evals E Baseline Oficial

Última baseline guardada antes desta tarefa:

| Campo | Valor |
|---|---|
| Run | `2026-07-19_21-28-19__fixed-generated__ollama__llama3.1-8b__r1` |
| Provider | `ollama` |
| Modelo | `llama3.1:8b` |
| Casos | 51 |
| Passaram | 51 |
| Falharam | 0 |
| Git dirty | `true` |

Essa baseline continua útil como referência histórica, mas não é uma baseline limpa.

Não foi criada nova baseline oficial nesta tarefa porque houve alterações de código e documentação. O passo correcto é:

1. rever diff;
2. autorizar commit;
3. criar commit;
4. correr baseline marcada no commit limpo.

Comandos para promover baseline depois do commit:

```powershell
.\.venv\Scripts\python.exe -m compileall app.py assistant ui prototype_web_ui evals tests
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m evals.run_evals --include-generated --mark-baseline
.\.venv\Scripts\python.exe -m evals.run_evals --category real_conversation
```

Critérios:

- working tree limpa antes do run;
- `git_dirty=false` no metadata final;
- provider `ollama`;
- modelo e `model_source` registados;
- resultados completos.

Smoke executado durante a Fase 0:

```powershell
.\.venv\Scripts\python.exe -m evals.run_evals --category real_conversation --model gemma3:12b --output-dir data\phase0_eval_smoke
```

Resultado do smoke:

| Campo | Valor |
|---|---|
| Provider | `ollama` |
| Modelo | `gemma3:12b` |
| Casos | 10 |
| Passaram | 10 |
| Falharam | 0 |
| Excepções | 0 |
| Git dirty | `true` |
| Baseline | `false` |

## 5. Limitações Ainda Aceites

- Não foi criada baseline oficial nova antes de commit.
- AnthropicProvider não foi implementado.
- Ruflo não foi integrado.
- Pesquisa real continua indisponível.
- Memória persistente não foi limpa nesta tarefa.
- Echo OS continua protótipo.
