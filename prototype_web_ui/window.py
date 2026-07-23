from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView

from prototype_web_ui.controller import EchoUIController


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


class EchoWebPage(QWebEnginePage):
    """Web page that forwards JavaScript console output to the Python terminal."""

    def javaScriptConsoleMessage(self, level, message: str, line_number: int, source_id: str) -> None:
        print(
            f"[Echo UI JS] level={level} "
            f"line={line_number} "
            f"source={source_id} "
            f"message={message}"
        )


class EchoOSWindow(QWebEngineView):
    """Alternative Echo OS UI backed by QWebEngineView."""

    def __init__(
        self,
        responder: Callable[[str], str],
        *,
        title: str = "Echo",
        clear_conversation: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
        get_telemetry: Callable[[], dict | None] | None = None,
        model_runtime: Any | None = None,
    ) -> None:
        super().__init__()
        self.on_close = on_close
        self.setWindowTitle(title)
        self.resize(1328, 860)

        self.setPage(EchoWebPage(self))
        self.controller = EchoUIController(
            responder,
            self,
            clear_conversation=clear_conversation,
            get_telemetry=get_telemetry,
            model_runtime=model_runtime,
        )
        print(
            "[Echo UI DEBUG] registered object name=echoController",
            f"controller_id={id(self.controller)}",
        )

        self.channel = QWebChannel(self.page())
        self.channel.registerObject("echoController", self.controller)
        self.page().setWebChannel(self.channel)
        self.loadFinished.connect(self._handle_load_finished)

        self.load(QUrl.fromLocalFile(str((WEB_DIR / "index.html").resolve())))

    def closeEvent(self, event) -> None:
        self.controller.shutdown()
        if self.on_close is not None:
            self.on_close()
        super().closeEvent(event)

    def _handle_load_finished(self, ok: bool) -> None:
        print(f"[Echo UI DEBUG] page_load_finished={ok}")
        self.page().runJavaScript(
            """
            console.log("[Echo UI JS] python load probe", {
              QWebChannel: typeof QWebChannel,
              qt: typeof qt,
              transport: Boolean(window.qt && qt.webChannelTransport),
              echoResponse: Boolean(document.getElementById("echoResponse")),
              echoEntity: Boolean(window.echoEntity)
            });
            """
        )
        QTimer.singleShot(
            250,
            lambda: self.page().runJavaScript(
                """
                const responseElement = document.getElementById("echoResponse");
                console.log("[Echo UI JS] python delayed probe", {
                  initialized: Boolean(window.__echoChannelInitialized),
                  controller: Boolean(window.echoController),
                  state: document.body.dataset.echoState || null,
                  responseText: responseElement ? responseElement.textContent : ""
                });
                """
            ),
        )
