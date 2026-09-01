from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from chipmem.embedding import TaskEmbedding
from chipmem.injector import _materialize_skills, render_skills
from chipmem.learning import learn_from_verified_run
from chipmem.retrieval import RetrievedSkill, retrieve_top_k
from chipmem.store import SkillStore, _atomic_json, _mkdir_chain_durable

_VERDICT_TOKEN = object()
_VERDICT_SEAL_KEY = secrets.token_bytes(32)


def _verdict_seal(label: str, task_sha256: str) -> bytes:
    return hmac.new(
        _VERDICT_SEAL_KEY,
        f"{label}:{task_sha256}".encode(),
        hashlib.sha256,
    ).digest()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedVerdict:
    label: str
    task_sha256: str
    _seal: bytes = field(repr=False, compare=False, init=False)

    def __init__(
        self,
        label: str,
        task_sha256: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _VERDICT_TOKEN:
            raise TypeError("verified verdicts are created only by run_harness")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "task_sha256", task_sha256)
        object.__setattr__(self, "_seal", _verdict_seal(label, task_sha256))

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError("VerifiedVerdict cannot be subclassed")

    @classmethod
    def _create(cls, label: str, task_sha256: str) -> VerifiedVerdict:
        return cls(label, task_sha256, _token=_VERDICT_TOKEN)

    def verify(self) -> None:
        try:
            valid = hmac.compare_digest(
                self._seal,
                _verdict_seal(self.label, self.task_sha256),
            )
        except (AttributeError, TypeError):
            valid = False
        if not valid:
            raise TypeError("VerifiedVerdict integrity verification failed")


class EmptyRetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedMemory:
    skills: tuple[RetrievedSkill, ...]
    injection: str

    def __post_init__(self) -> None:
        if type(self.skills) is not tuple:
            raise TypeError("PreparedMemory skills must be an exact tuple")
        if any(type(item) is not RetrievedSkill for item in self.skills):
            raise TypeError("PreparedMemory contains an invalid skill entry")


class ChipMEM:
    """Task-indexed procedural memory with verified PASS-only learning."""

    def __init__(
        self,
        state_root: str | Path,
        domain: str,
        *,
        top_k: int = 2,
        threshold: float = 0.6,
        allow_empty: bool = False,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.__mint_authority = object()
        self.__verified_verdicts: dict[
            int, tuple[VerifiedVerdict, str, str, TaskEmbedding, bool]
        ] = {}
        self.store = SkillStore(
            state_root,
            domain,
            _mint_authority=self.__mint_authority,
        )
        self.top_k = top_k
        self.threshold = float(threshold)
        self.allow_empty = bool(allow_empty)
        self.cache_directory = self.store.state_root / "task_embedding_cache" / domain

    def run_harness(
        self, task_embedding: TaskEmbedding, harness, *args, **kwargs
    ) -> VerifiedVerdict:
        self.store._require_minted_task_embedding(task_embedding)
        if not callable(harness):
            raise TypeError("harness must be callable")
        raw = harness(*args, **kwargs)
        if type(raw) is not str:
            raise TypeError("harness verdict must be a string")
        label = raw.lower()
        if label not in {"pass", "fail", "invalid"}:
            raise ValueError("harness must return pass, fail, or invalid")
        verdict = VerifiedVerdict._create(label, task_embedding.task_sha256)
        self.__verified_verdicts[id(verdict)] = (
            verdict,
            label,
            task_embedding.task_sha256,
            task_embedding,
            False,
        )
        return verdict

    def _require_verified_verdict(
        self, verdict: VerifiedVerdict, task_embedding: TaskEmbedding
    ) -> str:
        if type(verdict) is not VerifiedVerdict:
            raise TypeError("learn requires a verdict returned by run_harness")
        record = self.__verified_verdicts.get(id(verdict))
        if record is None or record[0] is not verdict:
            raise TypeError("verdict was not issued by this ChipMEM run_harness")
        verdict.verify()
        issued_label, issued_task, issued_embedding, consumed = record[1:]
        if verdict.label != issued_label or verdict.task_sha256 != issued_task:
            raise TypeError("VerifiedVerdict integrity verification failed")
        if issued_task != task_embedding.task_sha256:
            raise TypeError("verified verdict is bound to a different task")
        if issued_embedding is not task_embedding:
            raise TypeError("verified verdict is bound to a different task embedding")
        if consumed:
            raise TypeError("verified verdict was already consumed")
        self.__verified_verdicts[id(verdict)] = (
            verdict,
            issued_label,
            issued_task,
            issued_embedding,
            True,
        )
        return issued_label

    def prepare(self, task_embedding: TaskEmbedding) -> PreparedMemory:
        self.store._require_minted_task_embedding(task_embedding)
        skills = retrieve_top_k(
            self.store,
            list(task_embedding.vector),
            top_k=self.top_k,
            threshold=self.threshold,
        )
        if not skills and not self.allow_empty:
            raise EmptyRetrievalError(
                "ChipMEM retrieved no skills; explicitly allow empty retrieval for cold starts"
            )
        frozen_skills = tuple(skills)
        return PreparedMemory(
            skills=frozen_skills,
            injection=render_skills(frozen_skills),
        )

    def materialize(
        self, prepared: PreparedMemory, target: str | Path
    ) -> list[Path]:
        if type(prepared) is not PreparedMemory:
            raise TypeError("materialize requires PreparedMemory from prepare")
        snapshot = tuple(prepared.skills)
        known = set(self.store.list_skills())
        seen = set()
        sanitized = []
        for item in snapshot:
            if item.skill_id in seen or item.skill_id not in known:
                raise ValueError("prepared memory contains an unknown stored skill")
            seen.add(item.skill_id)
            expected_path = self.store.skill_dir(item.skill_id)
            entries = {path.name for path in expected_path.iterdir()}
            canonical = {"SKILL.md", "skill.json", "embeddings.json"}
            if entries != canonical or any(
                not (expected_path / name).is_file() or (expected_path / name).is_symlink()
                for name in canonical
            ):
                raise ValueError("stored skill must contain only canonical files")
            candidate_path = Path(item.path).resolve()
            if candidate_path != expected_path.resolve():
                raise ValueError("prepared memory skill path is not the stored skill")
            stored_text = self.store.read_text(item.skill_id)
            stored_metadata = self.store.read_metadata(item.skill_id)
            if item.text != stored_text:
                raise ValueError("prepared memory text is not the stored skill")
            if item.metadata != stored_metadata:
                raise ValueError("prepared memory metadata is not the stored skill")
            sanitized.append(
                RetrievedSkill(
                    skill_id=item.skill_id,
                    path=expected_path,
                    score=float(item.score),
                    text=stored_text,
                    metadata=stored_metadata,
                )
            )
        safe_skills = tuple(sanitized)
        if prepared.injection != render_skills(safe_skills):
            raise ValueError("prepared memory injection does not match stored skills")
        return _materialize_skills(safe_skills, target)

    def embed_task(self, document: str, embed, *, identity: dict) -> TaskEmbedding:
        required = {"model", "dimension", "provider"}
        if set(identity) != required:
            raise ValueError("embedding identity must contain model, dimension, provider")
        if not isinstance(identity["dimension"], int) or identity["dimension"] <= 0:
            raise ValueError("embedding dimension must be positive")
        if not all(isinstance(identity[key], str) and identity[key] for key in ("model", "provider")):
            raise ValueError("embedding model and provider must be non-empty strings")

        task_hash = hashlib.sha256(document.encode()).hexdigest()
        canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        identity_hash = hashlib.sha256(canonical_identity.encode()).hexdigest()
        _mkdir_chain_durable(self.cache_directory)
        cache = self.cache_directory / f"{task_hash}_{identity_hash}.json"
        lock_path = cache.with_suffix(".lock")

        def validate(raw, label):
            if not isinstance(raw, list):
                raise TypeError(f"{label} must be a list")
            vector = [float(value) for value in raw]
            if len(vector) != identity["dimension"] or any(
                not math.isfinite(value) for value in vector
            ):
                raise ValueError(f"{label} has invalid dimension or values")
            return vector

        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache.exists():
                payload = json.loads(cache.read_text(encoding="utf-8"))
                if payload.get("task_sha256") != task_hash or payload.get("identity") != identity:
                    raise ValueError("cached task embedding identity mismatch")
                vector = validate(payload.get("vector"), "cached task embedding")
                task_embedding = TaskEmbedding._create(task_hash, identity, vector)
                self.store._register_task_embedding(
                    task_embedding,
                    authority=self.__mint_authority,
                )
                return task_embedding

            vector = validate(embed(document), "task embedding")
            _atomic_json(
                cache,
                {
                    "identity": identity,
                    "task_sha256": task_hash,
                    "vector": vector,
                },
            )
            task_embedding = TaskEmbedding._create(task_hash, identity, vector)
            self.store._register_task_embedding(
                task_embedding,
                authority=self.__mint_authority,
            )
            return task_embedding

    def learn(
        self,
        transcript: list[dict],
        *,
        source_task: str,
        verdict: VerifiedVerdict,
        distill,
        task_embedding: TaskEmbedding,
    ) -> str | None:
        self.store._require_minted_task_embedding(task_embedding)
        verdict_label = self._require_verified_verdict(verdict, task_embedding)
        return learn_from_verified_run(
            self.store,
            transcript,
            source_task=source_task,
            verdict=verdict_label,
            distill=distill,
            task_embedding=task_embedding,
            authority=self.__mint_authority,
        )
