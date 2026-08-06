from __future__ import annotations

import threading
from collections.abc import Callable
from html import escape

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from assistant.voice_input import VoiceTranscriber, check_microphone


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


class VoiceWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(self, transcriber: VoiceTranscriber) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            text = self.transcriber.record_and_transcribe(self.stop_event, self.status_changed.emit)
            self.status_changed.emit("Concluido.")
            self.finished.emit(text)
        except Exception as exc:
            self.failed.emit(str(exc))


class MicrophoneTestWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        sample_rate: int,
        input_device: str | int | None,
        auto_select_input: bool,
        silent_rms_threshold: float,
        channels: int,
        probe_duration: float,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.auto_select_input = auto_select_input
        self.silent_rms_threshold = silent_rms_threshold
        self.channels = channels
        self.probe_duration = probe_duration

    def run(self) -> None:
        try:
            self.finished.emit(
                check_microphone(
                    sample_rate=self.sample_rate,
                    input_device=self.input_device,
                    auto_select=self.auto_select_input,
                    silent_rms_threshold=self.silent_rms_threshold,
                    channels=self.channels,
                    probe_duration=self.probe_duration,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MessageInput(QTextEdit):
    submit_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptRichText(False)
        self.setTabChangesFocus(True)
        self.setMinimumHeight(40)
        self.setMaximumHeight(92)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key_Return, Qt.Key_Enter):
            if modifiers & Qt.ControlModifier:
                self.insertPlainText("\n")
                return
            if modifiers == Qt.NoModifier:
                self.submit_requested.emit()
                return

        if key == Qt.Key_Escape:
            self.clear()
            return

        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(
        self,
        app_name: str,
        model_name: str,
        responder: Callable[[str], str],
        clear_history: Callable[[], None],
        change_presence: Callable[[str], None],
        get_presence_state: Callable[[], str],
        get_pending_tasks: Callable[[], str],
        get_pending_task_count: Callable[[], int],
        get_tasks_panel_expanded: Callable[[], bool],
        set_tasks_panel_expanded: Callable[[bool], None],
        presence_names: list[str],
        active_presence: str,
        on_close: Callable[[], None] | None = None,
        debug_contexts: bool = False,
        get_context_debug: Callable[[], str] | None = None,
        voice_available: bool = False,
        voice_status: str = "",
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
        initial_messages: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.responder = responder
        self.clear_history = clear_history
        self.change_presence = change_presence
        self.get_presence_state = get_presence_state
        self.get_pending_tasks = get_pending_tasks
        self.get_pending_task_count = get_pending_task_count
        self.get_tasks_panel_expanded = get_tasks_panel_expanded
        self.set_tasks_panel_expanded = set_tasks_panel_expanded
        self.on_close = on_close
        self.debug_contexts = debug_contexts
        self.get_context_debug = get_context_debug
        self.thread: QThread | None = None
        self.worker: ChatWorker | None = None
        self.voice_thread: QThread | None = None
        self.voice_worker: VoiceWorker | None = None
        self.microphone_test_thread: QThread | None = None
        self.microphone_test_worker: MicrophoneTestWorker | None = None
        self.voice_available = voice_available
        self.voice_sample_rate = voice_sample_rate
        self.voice_input_device = voice_input_device
        self.voice_auto_select_input = voice_auto_select_input
        self.voice_silent_rms_threshold = voice_silent_rms_threshold
        self.voice_channels = voice_channels
        self.voice_probe_duration = voice_probe_duration
        self.voice_min_record_seconds = voice_min_record_seconds
        self.voice_preroll_ms = voice_preroll_ms
        self.voice_ready_delay_ms = voice_ready_delay_ms
        self.voice_transcriber = VoiceTranscriber(
            model_name=voice_model,
            language=voice_language or "pt",
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

        self.setWindowTitle(app_name)
        self.resize(820, 620)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setPlaceholderText("A conversa aparece aqui.")

        self.input_box = MessageInput()
        self.input_box.setPlaceholderText("Escreve uma mensagem...")
        self.input_box.submit_requested.connect(self.send_message)

        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)

        self.voice_button = QPushButton("Mic")
        self.voice_button.setObjectName("voiceButton")
        self.voice_button.setToolTip("Gravar audio com Whisper local")
        self.voice_button.clicked.connect(self.toggle_voice_input)
        self.voice_button.setEnabled(self.voice_available)

        self.voice_status_label = QLabel(voice_status or ("Voz desligada" if not self.voice_available else "Voz pronta"))
        self.voice_status_label.setObjectName("voiceStatusLabel")

        self.microphone_test_button = QPushButton("Testar microfone")
        self.microphone_test_button.setObjectName("secondaryButton")
        self.microphone_test_button.clicked.connect(self.test_microphone)
        self.microphone_test_button.setEnabled(self.voice_available)

        self.clear_button = QPushButton("Limpar conversa")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_conversation)

        self.clear_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_shortcut.activated.connect(self.clear_conversation)

        model_label = QLabel(f"Modelo: {model_name}")
        model_label.setObjectName("modelLabel")

        presence_label = QLabel("Presenca:")
        presence_label.setObjectName("presenceLabel")

        self.active_presence_label = QLabel(f"Estado: {active_presence}")
        self.active_presence_label.setObjectName("activePresenceLabel")

        self.presence_combo = QComboBox()
        self.presence_combo.setObjectName("presenceCombo")
        for name in presence_names:
            self.presence_combo.addItem(name)
        self.presence_combo.setCurrentText(active_presence)
        self.presence_combo.currentTextChanged.connect(self._on_presence_changed)

        top_bar = QHBoxLayout()
        top_bar.addWidget(model_label)
        top_bar.addStretch()
        top_bar.addWidget(self.active_presence_label)
        top_bar.addWidget(presence_label)
        top_bar.addWidget(self.presence_combo)

        self.context_debug_area = QTextEdit()
        self.context_debug_area.setReadOnly(True)
        self.context_debug_area.setObjectName("contextDebugArea")
        self.context_debug_area.setMaximumHeight(96)
        self.context_debug_area.setPlaceholderText("Contextos ativos aparecem aqui em modo debug.")
        self.context_debug_area.setVisible(self.debug_contexts)

        self.tasks_expanded = bool(self.get_tasks_panel_expanded())

        self.tasks_toggle_button = QPushButton()
        self.tasks_toggle_button.setObjectName("tasksToggleButton")
        self.tasks_toggle_button.clicked.connect(self._toggle_tasks_panel)

        self.tasks_area = QTextEdit()
        self.tasks_area.setReadOnly(True)
        self.tasks_area.setObjectName("tasksArea")
        self.tasks_area.setMaximumHeight(120)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_box, 1)
        input_layout.addWidget(self.voice_button)
        input_layout.addWidget(self.microphone_test_button)
        input_layout.addWidget(self.voice_status_label)
        input_layout.addWidget(self.clear_button)
        input_layout.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.context_debug_area)
        layout.addWidget(self.chat_area, 1)
        layout.addWidget(self.tasks_toggle_button)
        layout.addWidget(self.tasks_area)
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
            QPushButton#voiceButton {
                background: #0f766e;
                min-width: 44px;
                padding-left: 10px;
                padding-right: 10px;
            }
            QLabel#modelLabel {
                color: #425466;
                font-size: 12px;
            }
            QLabel#presenceLabel, QLabel#activePresenceLabel, QLabel#voiceStatusLabel {
                color: #425466;
                font-size: 12px;
            }
            QPushButton#tasksToggleButton {
                background: #e2e8f0;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 10px;
                text-align: left;
                font-size: 12px;
                font-weight: 600;
            }
            QComboBox#presenceCombo {
                background: #ffffff;
                border: 1px solid #c9d2df;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 120px;
            }
            QTextEdit#contextDebugArea {
                background: #f8fafc;
                color: #334155;
                border: 1px dashed #cbd5e1;
                border-radius: 6px;
                font-size: 12px;
            }
            QTextEdit#tasksArea {
                background: #ffffff;
                color: #334155;
                border: 1px solid #d8e0ea;
                border-radius: 6px;
                font-size: 12px;
            }
            """
        )

        self._load_initial_messages(initial_messages or [])
        self._refresh_tasks()
        self._focus_input()

    def _on_presence_changed(self, presence_name: str) -> None:
        if self.thread is not None:
            return
        self.change_presence(presence_name)
        self.active_presence_label.setText(f"Estado: {presence_name}")

    def send_message(self) -> None:
        message = self.input_box.toPlainText().strip()
        if not message or self.thread is not None:
            self._focus_input()
            return

        self.input_box.clear()
        self._focus_input()
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

    def toggle_voice_input(self) -> None:
        if not self.voice_available:
            return

        if self.voice_worker is not None:
            self.voice_status_label.setText("A processar...")
            self.voice_button.setDisabled(True)
            self.voice_worker.stop()
            return

        if self.thread is not None:
            self._focus_input()
            return

        self.voice_thread = QThread()
        self.voice_worker = VoiceWorker(self.voice_transcriber)
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.status_changed.connect(self.voice_status_label.setText)
        self.voice_worker.finished.connect(self._handle_voice_text)
        self.voice_worker.failed.connect(self._handle_voice_error)
        self.voice_worker.finished.connect(self._cleanup_voice_worker)
        self.voice_worker.failed.connect(self._cleanup_voice_worker)
        self.voice_button.setText("Parar")
        self.voice_status_label.setText("Preparar...")
        self.voice_thread.start()

    def test_microphone(self) -> None:
        if not self.voice_available or self.microphone_test_thread is not None:
            self._focus_input()
            return

        self.microphone_test_thread = QThread()
        self.microphone_test_worker = MicrophoneTestWorker(
            self.voice_sample_rate,
            self.voice_input_device,
            self.voice_auto_select_input,
            self.voice_silent_rms_threshold,
            self.voice_channels,
            self.voice_probe_duration,
        )
        self.microphone_test_worker.moveToThread(self.microphone_test_thread)
        self.microphone_test_thread.started.connect(self.microphone_test_worker.run)
        self.microphone_test_worker.finished.connect(self._handle_microphone_test_success)
        self.microphone_test_worker.failed.connect(self._handle_microphone_test_error)
        self.microphone_test_worker.finished.connect(self._cleanup_microphone_test_worker)
        self.microphone_test_worker.failed.connect(self._cleanup_microphone_test_worker)
        self.microphone_test_button.setDisabled(True)
        self.voice_status_label.setText("A testar microfone...")
        self.microphone_test_thread.start()

    def clear_conversation(self) -> None:
        if self.thread is not None:
            return

        self.clear_history()
        self.chat_area.clear()
        self._refresh_tasks()
        self._focus_input()

    def _handle_response(self, response: str) -> None:
        self._append_message("AssistenteIA", response)
        self._refresh_presence_state()
        self._refresh_tasks()
        self._refresh_context_debug()

    def _handle_error(self, error: str) -> None:
        self._append_message("AssistenteIA", error)
        self._refresh_presence_state()
        self._refresh_tasks()
        QMessageBox.warning(self, "AssistenteIA", error)

    def _handle_voice_text(self, text: str) -> None:
        current_text = self.input_box.toPlainText().strip()
        if current_text:
            self.input_box.setPlainText(f"{current_text}\n{text}")
        else:
            self.input_box.setPlainText(text)
        duration = getattr(self.voice_transcriber, "last_audio_duration_seconds", 0.0)
        rms = getattr(self.voice_transcriber, "last_audio_rms", 0.0)
        self.voice_status_label.setText(f"Pronto. Audio: {duration:.1f}s, RMS {rms:.5f}")
        self._focus_input()

    def _handle_voice_error(self, error: str) -> None:
        self.voice_status_label.setText("Erro na voz")
        QMessageBox.warning(self, "AssistenteIA", f"Falha na voz: {error}")
        self._focus_input()

    def _handle_microphone_test_success(self, message: str) -> None:
        self.voice_status_label.setText("Pronto.")
        QMessageBox.information(self, "AssistenteIA", message)
        self._focus_input()

    def _handle_microphone_test_error(self, error: str) -> None:
        self.voice_status_label.setText("Microfone indisponivel")
        QMessageBox.warning(self, "AssistenteIA", f"Falha no microfone: {error}")
        self._focus_input()

    def cancel_current_request(self) -> None:
        """Cooperatively cancel the in-flight respond() call, if any.

        No cancel button exists in this window yet -- this only prepares the
        connection (mirrors prototype_web_ui/controller.py's
        cancelCurrentRequest) so a future control can call it without
        inventing new layout here.
        """
        if self.thread is None:
            return
        owner = getattr(self.responder, "__self__", None)
        cancel = getattr(owner, "cancel_current_request", None)
        if callable(cancel):
            cancel()

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
        self._focus_input()

    def _cleanup_voice_worker(self) -> None:
        if self.voice_thread is not None:
            self.voice_thread.quit()
            self.voice_thread.wait()
            self.voice_thread.deleteLater()
        if self.voice_worker is not None:
            self.voice_worker.deleteLater()
        self.voice_thread = None
        self.voice_worker = None
        self.voice_button.setText("Mic")
        self.voice_button.setEnabled(self.voice_available)
        self._focus_input()

    def _cleanup_microphone_test_worker(self) -> None:
        if self.microphone_test_thread is not None:
            self.microphone_test_thread.quit()
            self.microphone_test_thread.wait()
            self.microphone_test_thread.deleteLater()
        if self.microphone_test_worker is not None:
            self.microphone_test_worker.deleteLater()
        self.microphone_test_thread = None
        self.microphone_test_worker = None
        self.microphone_test_button.setEnabled(self.voice_available)
        self._focus_input()

    def _set_waiting(self, waiting: bool) -> None:
        self.input_box.setDisabled(waiting)
        self.send_button.setDisabled(waiting)
        self.clear_button.setDisabled(waiting)
        self.presence_combo.setDisabled(waiting)
        self.voice_button.setDisabled(waiting or not self.voice_available)
        self.microphone_test_button.setDisabled(waiting or not self.voice_available)
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

    def _refresh_presence_state(self) -> None:
        current_state = self.get_presence_state()
        self.active_presence_label.setText(f"Estado: {current_state}")
        if self.presence_combo.currentText() == current_state:
            return
        self.presence_combo.blockSignals(True)
        self.presence_combo.setCurrentText(current_state)
        self.presence_combo.blockSignals(False)

    def _refresh_context_debug(self) -> None:
        if not self.debug_contexts or self.get_context_debug is None:
            return
        self.context_debug_area.setPlainText(self.get_context_debug())

    def _refresh_tasks(self) -> None:
        count = self.get_pending_task_count()
        arrow = "v" if self.tasks_expanded else ">"
        self.tasks_toggle_button.setText(f"{arrow} Tarefas pendentes ({count})")
        self.tasks_area.setVisible(self.tasks_expanded)
        self.tasks_area.setPlainText(self.get_pending_tasks())

    def _toggle_tasks_panel(self) -> None:
        self.tasks_expanded = not self.tasks_expanded
        self.set_tasks_panel_expanded(self.tasks_expanded)
        self._refresh_tasks()
        self._focus_input()

    def _focus_input(self) -> None:
        if not self.input_box.isEnabled():
            return
        self.input_box.setFocus(Qt.OtherFocusReason)
        cursor = self.input_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.input_box.setTextCursor(cursor)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if self.isActiveWindow() and not self.isMinimized():
            self._focus_input()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._focus_input()

    def closeEvent(self, event) -> None:
        if self.voice_worker is not None:
            self.voice_worker.stop()
            self._cleanup_voice_worker()
        if self.microphone_test_thread is not None:
            self._cleanup_microphone_test_worker()
        context_observer_timer = getattr(self, "context_observer_timer", None)
        if context_observer_timer is not None:
            context_observer_timer.stop()
        context_observer = getattr(self, "context_observer", None)
        if context_observer is not None:
            flush_summary = getattr(context_observer, "flush_summary", None)
            if flush_summary is not None:
                flush_summary()
        if self.on_close is not None:
            self.on_close()
        super().closeEvent(event)
