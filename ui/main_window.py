from __future__ import annotations

from collections.abc import Callable
from html import escape

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ChatWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, responder: Callable[[str], str], message: str) -> None:
        super().__init__()
        self.responder = responder
        self.message = message

    def run(self) -> None:
        try:
            self.finished.emit(self.responder(self.message))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(
        self,
        app_name: str,
        model_name: str,
        responder: Callable[[str], str],
        clear_history: Callable[[], None],
        initial_messages: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.responder = responder
        self.clear_history = clear_history
        self.thread: QThread | None = None
        self.worker: ChatWorker | None = None

        self.setWindowTitle(app_name)
        self.resize(820, 620)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setPlaceholderText("A conversa aparece aqui.")

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Escreve uma mensagem...")
        self.input_box.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)

        self.clear_button = QPushButton("Limpar conversa")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_conversation)

        model_label = QLabel(f"Modelo: {model_name}")
        model_label.setObjectName("modelLabel")

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_box, 1)
        input_layout.addWidget(self.clear_button)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addWidget(model_label)
        layout.addWidget(self.chat_area, 1)
        layout.addLayout(input_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f7fb;
            }
            QTextEdit, QLineEdit {
                background: #ffffff;
                border: 1px solid #c9d2df;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background: #1f6feb;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #93a4b8;
            }
            QPushButton#secondaryButton {
                background: #64748b;
            }
            QLabel#modelLabel {
                color: #425466;
                font-size: 12px;
            }
            """
        )

        self._load_initial_messages(initial_messages or [])

    def send_message(self) -> None:
        message = self.input_box.text().strip()
        if not message or self.thread is not None:
            return

        self.input_box.clear()
        self._append_message("Utilizador", message)
        self._set_waiting(True)

        self.thread = QThread()
        self.worker = ChatWorker(self.responder, message)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_response)
        self.worker.failed.connect(self._handle_error)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.failed.connect(self._cleanup_worker)
        self.thread.start()

    def clear_conversation(self) -> None:
        if self.thread is not None:
            return

        self.clear_history()
        self.chat_area.clear()

    def _handle_response(self, response: str) -> None:
        self._append_message("AssistenteIA", response)

    def _handle_error(self, error: str) -> None:
        self._append_message("AssistenteIA", error)
        QMessageBox.warning(self, "AssistenteIA", error)

    def _cleanup_worker(self) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
        if self.worker is not None:
            self.worker.deleteLater()
        self.thread = None
        self.worker = None
        self._set_waiting(False)

    def _set_waiting(self, waiting: bool) -> None:
        self.input_box.setDisabled(waiting)
        self.send_button.setDisabled(waiting)
        self.clear_button.setDisabled(waiting)
        self.send_button.setText("A responder..." if waiting else "Enviar")

    def _append_message(self, author: str, message: str) -> None:
        escaped = escape(message).replace("\n", "<br>")
        self.chat_area.append(f"<b>{author}:</b><br>{escaped}<br>")

    def _load_initial_messages(self, messages: list[dict[str, str]]) -> None:
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                self._append_message("Utilizador", content)
            elif role == "assistant":
                self._append_message("AssistenteIA", content)
