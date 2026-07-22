# Echo Web UI Prototype / Echo OS UI

Protótipo isolado e primeira UI alternativa para validar PySide6 + `QWebEngineView` + `QWebChannel` com a interface visual do Echo.

A UI clássica continua disponível. A nova UI usa o `AssistantEngine` existente apenas quando arrancada por `app.py --ui echo-os`.

## Como arrancar o protótipo isolado

```powershell
cd C:\Users\alexm\.vscode\projects\assistenteIA
.\.venv\Scripts\Activate.ps1
python prototype_web_ui\run_prototype.py
```

Este modo usa um responder de teste e não chama Ollama.

## Como arrancar com o backend real

```powershell
python app.py --ui echo-os
```

UI clássica:

```powershell
python app.py --ui classic
```

Se o `.venv` ainda apontar para o Python da Microsoft Store quebrado, recria o ambiente virtual antes de testar.

## O que valida

- Janela PySide6 com `QWebEngineView`.
- HTML, CSS e JavaScript locais.
- Entidade animada em canvas 2D.
- Comunicação JavaScript -> Python por `QWebChannel`.
- Comunicação Python -> JavaScript por sinais Qt.
- Estados visuais: `idle`, `thinking`, `speaking`, `error`.
- Resposta real pelo `AssistantEngine` quando usado via `app.py --ui echo-os`.
- Processamento fora da thread principal.

## Controller

Python regista uma única instância de `EchoUIController` no `QWebChannel` como `echoController`.

- `submitMessage(str)`
- `cancelCurrentRequest()`
- `receiveMessage(str)` como compatibilidade
- `setState(str)`

Sinais:

- `responseReady(str)`
- `stateChanged(str)`
- `errorOccurred(str)`
- `requestStarted(str)`
- `requestFinished()`

## Estados de debug

Botões discretos no canto inferior direito:

- `1`: idle
- `2`: thinking
- `3`: speaking
- `4`: error

Também podes usar as mesmas teclas quando o input não estiver focado.

## Design

Baseado em `Echo OS.dc.html`, mas sem:

- `support.js`
- `DCLogic`
- `<x-dc>`
- placeholders `{{ }}`
- Google Fonts/CDN
- runtime do Design Component

As fontes usam fallbacks locais do sistema. A estrutura está pronta para adicionar Geist/Geist Mono locais no futuro.

## Limitações atuais

- Ainda não há voz nesta UI.
- Ainda não há workspace dinâmico nem cartões ligados a ferramentas.
- O cancelamento real depende de suporte futuro do backend.
