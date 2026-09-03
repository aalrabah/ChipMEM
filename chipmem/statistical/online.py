from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import time
from pathlib import Path

from chipmem.statistical.extraction import (
    extract_recovery,
    extract_rows,
    pre_call_feat,
)
from chipmem.statistical.model import NUDGE_THRESHOLD, AgentModel, Feat
from chipmem.statistical.nudge import WarningGenerator
from chipmem.store import _atomic_bytes, _fsync_directory, _mkdir_chain_durable


class StatisticalMemory:
    """Persistent step-online retry and recovery memory for one domain."""

    def __init__(
        self,
        state_root: str | Path,
        domain: str,
        features,
        *,
        threshold: float = NUDGE_THRESHOLD,
        planner=None,
        max_nudges: int | None = None,
    ):
        if not domain or "/" in domain or ".." in domain:
            raise ValueError("domain must be a simple non-empty name")
        if max_nudges is not None and max_nudges < 0:
            raise ValueError("max_nudges must be non-negative")
        self.state_root = Path(state_root).resolve()
        self.domain = domain
        self.features = features
        self.threshold = float(threshold)
        self.max_nudges = None if max_nudges is None else int(max_nudges)
        self.memory_dir = self.state_root / "agents" / domain / "memory"
        _mkdir_chain_durable(self.memory_dir)
        self.model_path = self.memory_dir / "model.json"
        self.experience_path = self.memory_dir / "nudge_experience.jsonl"
        self.recovery_path = self.memory_dir / "recovery_experience.jsonl"
        self.transaction_path = self.memory_dir / "predictor_transaction.json"
        self.lock_path = self.memory_dir / ".online.lock"
        self._thread_lock = threading.RLock()
        self._transcripts: dict[str, list[dict]] = {}
        self._pending: dict[str, tuple[str, object]] = {}
        self._nudge_counts: dict[str, int] = {}
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._recover_transaction()
            self.model = self._load_model()
            self._experience = self._load_jsonl(self.experience_path)
            self._recovery = self._load_jsonl(self.recovery_path)
        self.warning = WarningGenerator(
            self.model,
            self.features,
            threshold=self.threshold,
            planner=planner,
        )

    @property
    def update_count(self) -> int:
        return len(self._experience)

    @property
    def nudge_count(self) -> int:
        return sum(self._nudge_counts.values())

    def _load_model(self) -> AgentModel:
        if not self.model_path.exists():
            return AgentModel()
        payload = json.loads(self.model_path.read_bytes().decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("statistical model must be a JSON object")
        return AgentModel.from_dict(payload)

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid statistical JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError("statistical JSONL rows must be objects")
            rows.append(row)
        return rows

    def before_tool_call(self, session_id: str, tool: str, arguments) -> str | None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._thread_lock:
            if session_id in self._pending:
                raise RuntimeError("previous tool call has not completed")
            transcript = self._transcripts.setdefault(session_id, [])
            feature = pre_call_feat(transcript, tool, arguments, self.features)
            stage = self.features.stage_of(feature.ct)
            advice = None
            if (
                self.max_nudges is None
                or self._nudge_counts.get(session_id, 0) < self.max_nudges
            ):
                advice = self.warning.maybe_warn(
                    feature,
                    stage=stage,
                    recent=self._recent(transcript),
                )
                if advice:
                    self._nudge_counts[session_id] = (
                        self._nudge_counts.get(session_id, 0) + 1
                    )
            self._pending[session_id] = (tool, arguments)
            return advice

    def after_tool_call(
        self,
        session_id: str,
        tool: str,
        arguments,
        output: str,
    ) -> AgentModel:
        with self._thread_lock:
            pending = self._pending.get(session_id)
            if pending is None:
                raise RuntimeError("before_tool_call must run before after_tool_call")
            if pending != (tool, arguments):
                raise RuntimeError("completed tool call does not match pending call")
            transcript = self._transcripts.setdefault(session_id, [])
            transcript.append(
                {"tool": tool, "arguments": arguments, "output": output}
            )
            del self._pending[session_id]
            return self._record_step(session_id, transcript)

    def reconcile(
        self, session_id: str, transcript: list[dict] | None = None
    ) -> AgentModel:
        with self._thread_lock:
            if transcript is not None:
                self._transcripts[session_id] = [dict(call) for call in transcript]
            current = self._transcripts.get(session_id, [])
            model = self.model
            for end in range(1, len(current) + 1):
                model = self._record_step(session_id, current[:end])
            return model

    def transcript(self, session_id: str) -> list[dict]:
        return [dict(call) for call in self._transcripts.get(session_id, [])]

    def finalize_session(self, session_id: str, *, require_calls: bool) -> None:
        if session_id in self._pending:
            raise RuntimeError("session ended with an incomplete tool call")
        if require_calls and not self._transcripts.get(session_id):
            raise RuntimeError("statistical mode requires live tool-call hooks")
        self.reconcile(session_id)

    def _record_step(self, session_id: str, transcript: list[dict]) -> AgentModel:
        if not transcript:
            return self.model
        step = len(transcript) - 1
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._recover_transaction()
            self.model = self._load_model()
            self._experience = self._load_jsonl(self.experience_path)
            self._recovery = self._load_jsonl(self.recovery_path)
            call_bytes = json.dumps(
                transcript[step],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            call_sha256 = hashlib.sha256(call_bytes).hexdigest()
            existing_step = next(
                (
                    row
                    for row in self._experience
                    if row.get("session") == session_id
                    and row.get("step") == step
                ),
                None,
            )
            if existing_step is not None:
                if existing_step.get("call_sha256") != call_sha256:
                    raise ValueError("conflicting duplicate statistical step")
                self.warning.model = self.model
                return self.model

            item = extract_rows(transcript, self.features)[step]
            feature: Feat = item["feat"]
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            next_experience = list(self._experience)
            next_experience.append(
                {
                    "session": session_id,
                    "step": step,
                    "ts": timestamp,
                    "ct": feature.ct,
                    "rc": feature.rc,
                    "pr": feature.pr,
                    "pec": feature.pec,
                    "te": feature.te,
                    "success": item["success"],
                    "error_class": item["error_class"],
                    "call_sha256": call_sha256,
                }
            )
            self.model.retry.update(feature, item["success"])

            next_recovery = list(self._recovery)
            known = {
                (
                    str(row.get("session")),
                    row.get("failure_idx"),
                    str(row.get("strategy")),
                )
                for row in next_recovery
                if row.get("record_type") == "recovery"
            }
            for event in extract_recovery(transcript, self.features):
                mature = bool(event.get("recovered")) or (
                    step - int(event["failure_idx"])
                    >= 8
                )
                key = (
                    session_id,
                    event["failure_idx"],
                    str(event["strategy"]),
                )
                if not mature or key in known:
                    continue
                next_recovery.append(
                    {
                        "record_type": "recovery",
                        "session": session_id,
                        "step": step,
                        "ts": timestamp,
                        "failure_idx": event["failure_idx"],
                        "error": event["error"],
                        "stage": event["stage"],
                        "strategy": event["strategy"],
                        "recovered": int(event["recovered"]),
                        "plan_excerpt": event.get("plan_excerpt"),
                        "context_window": event.get("context_window", []),
                    }
                )
                known.add(key)
                self.model.recovery.update(
                    event["error"],
                    event["stage"],
                    event["strategy"],
                    int(event["recovered"]),
                    excerpt=event.get("excerpt"),
                )

            all_rows = [
                (
                    Feat(
                        ct=row.get("ct", ""),
                        rc=row.get("rc", 0),
                        pr=row.get("pr", "none"),
                        pec=row.get("pec", "none"),
                        te=row.get("te", 0),
                    ),
                    int(row.get("success", 0)),
                )
                for row in next_experience
            ]
            previous_strength = self.model.retry.parent_strength
            strength = self.model.retry.retune_parent_strength(all_rows)
            self.model.recovery.parent_strength = strength
            self.model.metadata.update(
                {
                    "domain": self.domain,
                    "training_rows": len(all_rows),
                    "sessions": len(
                        {str(row.get("session")) for row in next_experience}
                    ),
                    "parent_strength": strength,
                    "previous_parent_strength": previous_strength,
                    "updated": timestamp,
                }
            )
            self._commit(self.model, next_experience, next_recovery)
            self._experience = next_experience
            self._recovery = next_recovery
            self.warning.model = self.model
            return self.model

    def _commit(
        self,
        model: AgentModel,
        experience: list[dict],
        recovery: list[dict],
    ) -> None:
        model_bytes = json.dumps(
            model.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        experience_bytes = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
            for row in experience
        )
        recovery_bytes = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
            for row in recovery
        )
        transaction = json.dumps(
            {
                "version": 1,
                "model": model_bytes.decode("utf-8"),
                "experience": experience_bytes.decode("utf-8"),
                "recovery": recovery_bytes.decode("utf-8"),
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "experience_sha256": hashlib.sha256(
                    experience_bytes
                ).hexdigest(),
                "recovery_sha256": hashlib.sha256(recovery_bytes).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _atomic_bytes(self.transaction_path, transaction)
        self._apply_transaction(json.loads(transaction))
        self.transaction_path.unlink()
        _fsync_directory(self.memory_dir)

    def _apply_transaction(self, transaction: dict) -> None:
        required = {
            "version",
            "model",
            "experience",
            "recovery",
            "model_sha256",
            "experience_sha256",
            "recovery_sha256",
        }
        if set(transaction) != required:
            raise ValueError("statistical transaction is incomplete")
        if transaction["version"] != 1:
            raise ValueError("statistical transaction version is unsupported")
        payloads = {
            "model": transaction["model"].encode("utf-8"),
            "experience": transaction["experience"].encode("utf-8"),
            "recovery": transaction["recovery"].encode("utf-8"),
        }
        for name, payload in payloads.items():
            if hashlib.sha256(payload).hexdigest() != transaction[f"{name}_sha256"]:
                raise ValueError(f"statistical transaction {name} checksum mismatch")
        _atomic_bytes(self.model_path, payloads["model"])
        _atomic_bytes(
            self.experience_path,
            payloads["experience"],
        )
        _atomic_bytes(
            self.recovery_path,
            payloads["recovery"],
        )

    def _recover_transaction(self) -> bool:
        if not self.transaction_path.exists():
            return False
        transaction = json.loads(
            self.transaction_path.read_bytes().decode("utf-8")
        )
        self._apply_transaction(transaction)
        self.transaction_path.unlink()
        _fsync_directory(self.memory_dir)
        return True

    @staticmethod
    def _recent(transcript: list[dict]) -> str:
        return "\n".join(
            f"{call.get('tool', '')}: {str(call.get('output', ''))[:200]}"
            for call in transcript[-5:]
        )
