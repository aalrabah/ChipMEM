from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from dataclasses import dataclass, field

_MINT_TOKEN = object()
_SEAL_KEY = secrets.token_bytes(32)


def _seal_payload(
    task_sha256: str, identity_json: str, vector: tuple[float, ...]
) -> bytes:
    payload = json.dumps(
        {
            "identity_json": identity_json,
            "task_sha256": task_sha256,
            "vector": vector,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(_SEAL_KEY, payload, hashlib.sha256).digest()


@dataclass(frozen=True, slots=True, init=False)
class TaskEmbedding:
    """Sealed binding between an exact task artifact and its embedding vector."""

    task_sha256: str
    identity_json: str
    vector: tuple[float, ...]
    _seal: bytes = field(repr=False, compare=False, init=False)

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError("TaskEmbedding cannot be subclassed")

    def __init__(
        self,
        task_sha256: str,
        identity_json: str,
        vector: tuple[float, ...],
        *,
        _mint_token: object | None = None,
    ) -> None:
        if _mint_token is not _MINT_TOKEN:
            raise TypeError("TaskEmbedding values must be minted by ChipMEM.embed_task")
        object.__setattr__(self, "task_sha256", task_sha256)
        object.__setattr__(self, "identity_json", identity_json)
        object.__setattr__(self, "vector", tuple(float(value) for value in vector))
        self._validate_fields()
        object.__setattr__(
            self,
            "_seal",
            _seal_payload(self.task_sha256, self.identity_json, self.vector),
        )

    def _validate_fields(self) -> None:
        if len(self.task_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.task_sha256
        ):
            raise ValueError("invalid task SHA-256")
        identity = json.loads(self.identity_json)
        if not isinstance(identity, dict):
            raise TypeError("invalid embedding identity")
        if not self.vector or any(not math.isfinite(value) for value in self.vector):
            raise ValueError("invalid task embedding vector")

    def verify(self) -> None:
        self._validate_fields()
        expected = _seal_payload(self.task_sha256, self.identity_json, self.vector)
        try:
            valid = hmac.compare_digest(self._seal, expected)
        except (AttributeError, TypeError):
            valid = False
        if not valid:
            raise TypeError("TaskEmbedding integrity verification failed")

    @property
    def identity(self) -> dict:
        self.verify()
        return json.loads(self.identity_json)

    @classmethod
    def _create(
        cls, task_sha256: str, identity: dict, vector: list[float]
    ) -> TaskEmbedding:
        return cls(
            task_sha256=task_sha256,
            identity_json=json.dumps(identity, sort_keys=True, separators=(",", ":")),
            vector=tuple(float(value) for value in vector),
            _mint_token=_MINT_TOKEN,
        )
