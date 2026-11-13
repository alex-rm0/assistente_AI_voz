"""Privacy classes (North Star §6). The class belongs to concrete data, not to a module."""
from __future__ import annotations

from enum import Enum

from assistant.core.contracts._base import ContractError


class PrivacyClass(str, Enum):
    PUBLIC = "PUBLIC"
    PERSONAL = "PERSONAL"
    PRIVATE = "PRIVATE"
    RESTRICTED = "RESTRICTED"


_RANK = {cls: rank for rank, cls in enumerate(PrivacyClass)}

# Alexandre, 12/11: what he writes or says to the Echo OS (window/voice) is
# PERSONAL until a classifier says otherwise. The contracts still require a
# class; this is the value the Echo OS adapter passes.
ECHO_OS_DEFAULT_PRIVACY_CLASS = PrivacyClass.PERSONAL


def derived_privacy_class(*sources):
    """Derived content inherits the most restrictive class of its sources.

    Reclassifying down is a deliberate, explicit rule elsewhere, never this function.
    """
    if not sources:
        raise ContractError("privacy_class: derived content needs at least one source")
    try:
        return max((PrivacyClass(s) for s in sources), key=_RANK.__getitem__)
    except ValueError as exc:
        raise ContractError(f"privacy_class: {exc}") from None
