"""Lazy bridge to local STT in a separate Python environment."""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


class FasterWhisperBackend:
    def __init__(self, *, python=None, model="large-v3-turbo", process_factory=None,
                 load_timeout=60, transcription_timeout=45):
        root = Path(__file__).resolve().parents[1]
        self.python = Path(python or os.environ.get("ECHO_STT_PYTHON") or root / ".venv_stt_bench/Scripts/python.exe")
        self.model = model
        self.process_factory = process_factory or subprocess.Popen
        self.load_timeout = load_timeout
        self.transcription_timeout = transcription_timeout
        self.process = None
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        atexit.register(self.close)

    def _receive(self, timeout):
        try:
            payload = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            self._terminate()
            raise RuntimeError("Tempo limite do STT GPU excedido.") from exc
        if not payload.get("ok"):
            self._terminate()
            raise RuntimeError(payload.get("error") or "O processo STT GPU terminou.")
        return payload

    def _read(self, process, responses):
        try:
            for line in process.stdout:
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    responses.put(payload)
        finally:
            responses.put({"ok": False, "error": "O processo STT GPU terminou."})

    def _start(self):
        if self.closed:
            raise RuntimeError("O STT GPU já foi encerrado.")
        if self.process is not None and self.process.poll() is None:
            return
        if not self.python.is_file():
            raise RuntimeError("Runtime STT GPU não encontrado. Configura ECHO_STT_PYTHON.")
        self.responses = queue.Queue()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.setdefault("HF_HOME", str(Path.cwd() / ".hf_cache"))
        self.process = self.process_factory(
            [str(self.python), "-u", str(Path(__file__).with_name("stt_runner.py")), "--model", self.model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env=env,
        )
        if self.closed:
            self._terminate()
            raise RuntimeError("O STT GPU já foi encerrado.")
        threading.Thread(target=self._read, args=(self.process, self.responses), daemon=True).start()
        ready = self._receive(self.load_timeout)
        if ready.get("device") != "cuda":
            self._terminate()
            raise RuntimeError("O STT GPU não confirmou CUDA; fallback CPU desativado.")
        print(f"[STT_BACKEND] provider=faster-whisper model={self.model} device=cuda model_load_ms={ready.get('model_load_ms', 0):.0f}", flush=True)

    def transcribe(self, audio_path, *, language, initial_prompt):
        with self.lock:
            self._start()
            started = time.perf_counter()
            request = {"path": str(Path(audio_path).resolve()), "language": language, "initial_prompt": initial_prompt}
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                self._terminate()
                raise RuntimeError("Não consegui comunicar com o STT GPU.") from exc
            result = self._receive(self.transcription_timeout)
            print(f"[STT_BACKEND] provider=faster-whisper model={self.model} device=cuda transcription_ms={(time.perf_counter()-started)*1000:.0f}", flush=True)
            return result["result"]

    def warmup(self):
        """Start the worker and run one short silent transcription, so the
        first real turn pays neither the model load nor the first CUDA
        inference (~4.7 s + ~1.3 s). Blocking: call it off the UI thread.
        Raises like transcribe() (no CUDA, out of memory, closed)."""
        import tempfile
        import wave

        handle, name = tempfile.mkstemp(prefix="echo_stt_warmup_", suffix=".wav")
        os.close(handle)
        path = Path(name)
        try:
            with wave.open(str(path), "wb") as clip:
                clip.setnchannels(1)
                clip.setsampwidth(2)
                clip.setframerate(16000)
                clip.writeframes(b"\x00\x00" * 8000)  # 0.5 s of silence
            self.transcribe(path, language="pt", initial_prompt="")
        finally:
            path.unlink(missing_ok=True)

    def _terminate(self):
        process, self.process = self.process, None
        if process is None:
            return
        # Reap off the caller's thread; shutdown must not stall the Qt GUI.
        def reap():
            try:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    # Windows venv launchers can own a second Python process.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=5)
                else:
                    process.kill()
                process.wait()
            finally:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
        threading.Thread(target=reap, daemon=True).start()

    def close(self):
        self.closed = True
        self._terminate()
