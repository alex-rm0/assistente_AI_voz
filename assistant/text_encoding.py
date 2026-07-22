from __future__ import annotations


MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "�", "ß", "Ò", "Ú", "¾", "þ")


def has_mojibake_markers(value: str) -> bool:
    text = str(value or "")
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def debug_text_encoding(label: str, value: str, limit: int = 40) -> None:
    text = str(value or "")
    print(
        f"[UTF8 DEBUG] {label}",
        "repr=",
        repr(text),
        "codepoints=",
        [hex(ord(ch)) for ch in text[:limit]],
        "mojibake_markers=",
        has_mojibake_markers(text),
    )
