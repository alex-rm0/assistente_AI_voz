from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal, Slot


class EchoBridge(QObject):
    """Small QWebChannel bridge used only by the web UI prototype."""

    responseReady = Signal(str)
    stateChanged = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"

    @Slot(str)
    def receiveMessage(self, text: str) -> None:
        message = (text or "").strip()
        if not message:
            self.errorOccurred.emit("Escreve uma mensagem primeiro.")
            self.setState("error")
            QTimer.singleShot(900, lambda: self.setState("idle"))
            return

        self.setState("thinking")
        QTimer.singleShot(700, lambda: self._finish_test_response(message))

    @Slot(str)
    def setState(self, state: str) -> None:
        clean_state = (state or "").strip().lower()
        if clean_state not in {"idle", "thinking", "speaking", "error"}:
            clean_state = "error"
        self._state = clean_state
        self.stateChanged.emit(clean_state)

    @Slot(str)
    def sendTestResponse(self, text: str) -> None:
        self.responseReady.emit(text)

    def _finish_test_response(self, message: str) -> None:
        self.setState("speaking")
        self.responseReady.emit(f"Recebi: {message}")
        QTimer.singleShot(1100, lambda: self.setState("idle"))
