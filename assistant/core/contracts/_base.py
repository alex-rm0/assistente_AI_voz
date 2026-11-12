"""Shared machinery for Core contracts: coercion, validation, freezing and JSON.

Every contract is a frozen dataclass that subclasses `Contract`. Construction
coerces JSON-shaped values (enum strings, ISO timestamps, nested dicts) into
typed values and then runs the contract's own invariants, so a contract built
in code and one read from JSON go through exactly the same checks.

Versioning: `contract_version` is "MAJOR.MINOR". A reader accepts its own major
only. Unknown fields are an error at the reader's minor (they are typos) and are
dropped from a newer minor (additive, optional fields). Fields listed in
`_FORBIDDEN_FIELDS` are rejected at any version.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import types
import typing
from collections.abc import Mapping
from datetime import datetime
from enum import Enum

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")


class ContractError(ValueError):
    """A value does not satisfy a Core contract."""


def _freeze(value, name):
    if isinstance(value, Mapping):
        return types.MappingProxyType({str(k): _freeze(v, name) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v, name) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{name}: non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ContractError(f"{name}: {type(value).__name__} is not JSON-compatible")


def _thaw(value):
    if isinstance(value, Contract):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _coerce(tp, value, name, allow_empty):
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        tp = args[0]
    if value is None:
        raise ContractError(f"{name}: required")
    origin = typing.get_origin(tp)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ContractError(f"{name}: expected a list")
        item_type = typing.get_args(tp)[0]
        return tuple(_coerce(item_type, v, f"{name}[{i}]", False) for i, v in enumerate(value))
    if tp is Mapping or origin is Mapping:
        if not isinstance(value, Mapping):
            raise ContractError(f"{name}: expected an object")
        return _freeze(value, name)
    if isinstance(tp, type) and issubclass(tp, Enum):
        if isinstance(value, bool):
            raise ContractError(f"{name}: invalid value {value!r}")
        try:
            return tp(value)
        except ValueError:
            raise ContractError(f"{name}: invalid value {value!r}") from None
    if isinstance(tp, type) and issubclass(tp, Contract):
        if isinstance(value, tp):
            return value
        if isinstance(value, Mapping):
            return tp.from_dict(value)
        raise ContractError(f"{name}: expected {tp.__name__}")
    if tp is datetime:
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ContractError(f"{name}: not an ISO 8601 timestamp") from None
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ContractError(f"{name}: expected a timezone-aware timestamp")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ContractError(f"{name}: expected a boolean")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(f"{name}: expected a finite number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ContractError(f"{name}: expected a string")
        if not allow_empty and not value.strip():
            raise ContractError(f"{name}: must not be empty")
        return value
    raise TypeError(f"{name}: unsupported contract field type {tp!r}")  # programming error


class Contract:
    CONTRACT: typing.ClassVar[str]
    VERSION: typing.ClassVar[str] = "1.0"
    _ALLOW_EMPTY: typing.ClassVar[frozenset] = frozenset()
    _FORBIDDEN_FIELDS: typing.ClassVar[frozenset] = frozenset()
    _FORBIDDEN_REASON: typing.ClassVar[str] = "field not allowed in this contract"

    def __post_init__(self):
        if self.contract_version is None:
            object.__setattr__(self, "contract_version", self.VERSION)
        if self.contract_version != self.VERSION:
            raise ContractError(f"contract_version: {self.CONTRACT} objects are built at {self.VERSION}")
        hints = typing.get_type_hints(type(self))
        for f in dataclasses.fields(self):
            if f.name != "contract_version":
                value = _coerce(hints[f.name], getattr(self, f.name), f.name, f.name in self._ALLOW_EMPTY)
                object.__setattr__(self, f.name, value)
        self._check()

    def _check(self):
        """Contract-specific invariants. Raise ContractError."""

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ContractError(f"{cls.CONTRACT}: expected an object")
        data = dict(data)
        if data.pop("contract", None) != cls.CONTRACT:
            raise ContractError(f"contract: expected {cls.CONTRACT!r}")
        version = data.pop("contract_version", None)
        match = _VERSION_RE.match(version) if isinstance(version, str) else None
        major, minor = map(int, _VERSION_RE.match(cls.VERSION).groups())
        if match is None or int(match.group(1)) != major:
            raise ContractError(f"contract_version: unsupported {version!r} for {cls.CONTRACT} {cls.VERSION}")
        forbidden = sorted(set(data) & cls._FORBIDDEN_FIELDS)
        if forbidden:
            raise ContractError(f"{', '.join(forbidden)}: {cls._FORBIDDEN_REASON}")
        known = {f.name for f in dataclasses.fields(cls)} - {"contract_version"}
        unknown = sorted(set(data) - known)
        if unknown:
            if int(match.group(2)) <= minor:
                raise ContractError(f"unknown field(s) for {cls.CONTRACT} {version}: {', '.join(unknown)}")
            for name in unknown:
                del data[name]
        for f in dataclasses.fields(cls):
            if (f.name not in data and f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING):
                raise ContractError(f"{f.name}: required")
        return cls(**data)

    @classmethod
    def from_json(cls, text):
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{cls.CONTRACT}: invalid JSON ({exc})") from None
        return cls.from_dict(data)

    def to_dict(self):
        data = {"contract": self.CONTRACT, "contract_version": self.contract_version}
        for f in dataclasses.fields(self):
            data[f.name] = _thaw(getattr(self, f.name))
        return data

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ContractError(message)


IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
