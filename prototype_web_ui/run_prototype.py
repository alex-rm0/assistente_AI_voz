from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype_web_ui.window import EchoOSWindow


def main() -> int:
    app = QApplication(sys.argv)

    def test_responder(message: str) -> str:
        return f"Recebi: {message}"

    window = EchoOSWindow(test_responder, title="Echo Web UI Prototype")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
