from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from prototype_web_ui.controller import EchoUIController
from assistant.voice_input import VoiceTranscriber


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
        voice_available: bool = False,
        voice_model: str = "base",
        voice_language: str = "pt",
        voice_sample_rate: int = 44100,
        voice_input_device: str | int | None = "default",
        voice_auto_select_input: bool = True,
        voice_silent_rms_threshold: float = 0.001,
        voice_channels: int = 1,
        voice_probe_duration: float = 0.5,
        voice_min_record_seconds: float = 2.0,
        voice_preroll_ms: int = 500,
        voice_ready_delay_ms: int = 200,
        voice_missing_dependencies: list[str] | tuple[str, ...] | None = None,
        tts_manager: Any | None = None,
        get_response_language: Callable[[], str] | None = None,
    ) -> None:
        super().__init__()
        self.on_close = on_close
        self._close_in_progress = False
        self._close_completed = False
        self.setWindowTitle(title)
        self.resize(1328, 860)

        self.setPage(EchoWebPage(self))
        voice_transcriber = (
            VoiceTranscriber(
                model_name=voice_model,
                language=voice_language or "pt",
                # Phase 7.5 (section 10/12): input_language for the NEW
                # Voice-Conversation-Mode VAD path is deliberately threaded
                # from this SAME existing voice.language config value (not
                # from response_language -- see section 10's decoupling
                # requirement), so today's config ("pt") preserves exactly
                # today's real behavior with zero risk. The class-level
                # default is "auto" for anyone not explicitly configuring
                # "pt"/"en" -- flipping the real config value to "auto" is
                # deferred until a human validates it with a live
                # microphone (see the Phase 7.5 report).
                input_language=voice_language or "auto",
                sample_rate=voice_sample_rate,
                input_device=voice_input_device,
                auto_select_input=voice_auto_select_input,
                silent_rms_threshold=voice_silent_rms_threshold,
                channels=voice_channels,
                probe_duration=voice_probe_duration,
                min_record_seconds=voice_min_record_seconds,
                preroll_ms=voice_preroll_ms,
                ready_delay_ms=voice_ready_delay_ms,
            )
            if voice_available
            else None
        )
        self.controller = EchoUIController(
            responder,
            self,
            clear_conversation=clear_conversation,
            get_telemetry=get_telemetry,
            model_runtime=model_runtime,
            voice_available=voice_available,
            voice_transcriber=voice_transcriber,
            voice_missing_dependencies=voice_missing_dependencies,
            tts_manager=tts_manager,
            get_response_language=get_response_language,
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
        # Phase 7.6.7 (section 9): closing used to run controller.shutdown()
        # and on_close() synchronously right here, on the GUI thread -- a
        # real close sat ~8s in on_close's session-summary embedding call
        # (cold Ollama model) and Windows marked the window "Não
        # responde". Now the first close request is refused, the window is
        # hidden at once, and the controller drains its workers while the
        # event loop keeps running; once that finishes, close() is called
        # again and this time accepted.
        if self._close_completed:
            super().closeEvent(event)
            return
        event.ignore()
        if self._close_in_progress:
            return
        self._close_in_progress = True
        self.hide()
        self.controller.shutdown_async(self._finish_close, session_close=self.on_close)

    def _finish_close(self) -> None:
        self._close_completed = True
        self.close()
        # A window that was already hidden doesn't trigger Qt's
        # quit-on-last-window-closed when close() is accepted (verified
        # offscreen), so quit explicitly once nothing else is visible.
        app = QApplication.instance()
        if app is not None and app.quitOnLastWindowClosed() and not any(
            widget.isVisible() for widget in app.topLevelWidgets()
        ):
            app.quit()

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
