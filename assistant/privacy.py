"""Privacy classes for data sent to a model (North Star §6).

The class belongs to the data of one request, not to a module. The model
router checks `provider_allowed` at its boundary, so callers only have to say
what they are sending. A request that says nothing is PRIVATE.
"""

from __future__ import annotations

import re
import unicodedata

PUBLIC = "PUBLIC"          # general knowledge, a public web page
PERSONAL = "PERSONAL"      # ordinary conversation, a common preference, the core profile
PRIVATE = "PRIVATE"        # a personal document from the workspace, a sensitive conversation
RESTRICTED = "RESTRICTED"  # email subject and body, restricted work information
DEFAULT_PRIVACY_CLASS = PRIVATE
_ORDER = (PUBLIC, PERSONAL, PRIVATE, RESTRICTED)
LOCAL_PROVIDERS = frozenset({"ollama", "fake", "local"})


def normalize_privacy_class(value: object) -> str:
    """A known class, or PRIVATE for anything missing or unknown."""
    text = str(value or "").strip().upper()
    return text if text in _ORDER else DEFAULT_PRIVACY_CLASS


def stricter(first: object, second: object) -> str:
    a, b = normalize_privacy_class(first), normalize_privacy_class(second)
    return a if _ORDER.index(a) >= _ORDER.index(b) else b


def provider_allowed(provider: str, privacy_class: object, *, explicit_document_request: bool = False) -> bool:
    """May data of this class be sent to this provider?

    Local providers may see everything. A cloud provider may see PUBLIC and
    PERSONAL data; PRIVATE data only when the user's own turn is an explicit
    request about that document; RESTRICTED data never.
    """
    if str(provider or "").strip().lower() in LOCAL_PROVIDERS:
        return True
    resolved = normalize_privacy_class(privacy_class)
    if resolved in (PUBLIC, PERSONAL):
        return True
    return resolved == PRIVATE and bool(explicit_document_request)


# Text that the engine itself writes around workspace document content. Finding
# one in a payload means a document is in it, whatever the caller declared.
#
# There is no marker for RESTRICTED: it is for mail that Echo reads from a
# mailbox (F5, not built yet), and that content will carry the class from its
# source. An email Alexandre dictates a draft of is his own request, not mail
# content, and is not RESTRICTED.
_DOCUMENT_MARKERS = re.compile(r"ficheiro lido|conteudo extraido|acao documental\s*:|documento ativo\s*:")


def content_privacy_class(texts: list[str]) -> str | None:
    """The class implied by known markers in the payload, or None when there are none."""
    joined = "\n".join(str(text or "") for text in texts).lower()
    joined = "".join(ch for ch in unicodedata.normalize("NFKD", joined) if not unicodedata.combining(ch))
    return PRIVATE if _DOCUMENT_MARKERS.search(joined) else None
