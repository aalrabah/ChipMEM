from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_PARENT_STRENGTH = 8.0
PARENT_STRENGTH_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
RECOVERY_HORIZON = 8
RECOVERY_MAX_EXCERPTS = 3
PLAN_MAX_STEPS = 3
NUDGE_THRESHOLD = 0.20


@dataclass(frozen=True)
class Feat:
    ct: str = ""
    rc: int = 0
    pr: str = "none"
    pec: str = "none"
    te: int = 0


def retry_bucket(retry_count: int) -> int:
    return min(int(retry_count), 3)


class HierarchicalRetry:
    def __init__(
        self,
        parent_strength: float = DEFAULT_PARENT_STRENGTH,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        if parent_strength <= 0 or prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Bayesian strengths must be positive")
        self.parent_strength = float(parent_strength)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.counts: dict[str, list[float]] = {}

    @property
    def K(self) -> float:
        return self.parent_strength

    @K.setter
    def K(self, value: float) -> None:
        self.parent_strength = float(value)

    @staticmethod
    def k_global() -> str:
        return "global"

    @staticmethod
    def k_ct(feature: Feat) -> str:
        return f"ct={feature.ct}"

    @staticmethod
    def k_ct_err(feature: Feat) -> str:
        return f"ct={feature.ct}|err={feature.pec}"

    @staticmethod
    def k_exact(feature: Feat) -> str:
        return (
            f"ct={feature.ct}|rb={retry_bucket(feature.rc)}|err={feature.pec}"
        )

    def _success_total(self, key: str) -> tuple[float, float]:
        successes, failures = self.counts.get(key, (0.0, 0.0))
        return float(successes), float(successes) + float(failures)

    def predict(self, feature: Feat) -> tuple[float, int]:
        successes, total = self._success_total(self.k_global())
        probability = (self.prior_alpha + successes) / (
            self.prior_alpha + self.prior_beta + total
        )
        for key in (
            self.k_ct(feature),
            self.k_ct_err(feature),
            self.k_exact(feature),
        ):
            successes, total = self._success_total(key)
            probability = (
                successes + self.parent_strength * probability
            ) / (total + self.parent_strength)
        _, exact_total = self._success_total(self.k_exact(feature))
        return probability, int(exact_total)

    def update(self, feature: Feat, success: int) -> None:
        if success not in {0, 1}:
            raise ValueError("success must be 0 or 1")
        for key in (
            self.k_global(),
            self.k_ct(feature),
            self.k_ct_err(feature),
            self.k_exact(feature),
        ):
            successes, failures = self.counts.setdefault(key, [0.0, 0.0])
            if success:
                self.counts[key][0] = successes + 1
            else:
                self.counts[key][1] = failures + 1

    def retune_parent_strength(
        self,
        rows: list[tuple[Feat, int]],
        grid: tuple[float, ...] = PARENT_STRENGTH_GRID,
        holdout_fraction: float = 0.2,
        minimum_rows: int = 40,
    ) -> float:
        if len(rows) < minimum_rows:
            return self.parent_strength
        if (
            not 0 < holdout_fraction < 1
            or not grid
            or any(strength <= 0 for strength in grid)
        ):
            return self.parent_strength
        stride = int(1 / holdout_fraction)
        held_out = [row for index, row in enumerate(rows) if index % stride == 0]
        training = [row for index, row in enumerate(rows) if index % stride != 0]
        if not held_out or not training:
            return self.parent_strength
        best_strength = self.parent_strength
        best_likelihood = -math.inf
        for strength in grid:
            candidate = HierarchicalRetry(
                parent_strength=strength,
                prior_alpha=self.prior_alpha,
                prior_beta=self.prior_beta,
            )
            for feature, success in training:
                candidate.update(feature, success)
            likelihood = 0.0
            for feature, success in held_out:
                probability, _ = candidate.predict(feature)
                probability = min(max(probability, 1e-9), 1 - 1e-9)
                likelihood += (
                    math.log(probability)
                    if success
                    else math.log(1 - probability)
                )
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_strength = strength
        self.parent_strength = float(best_strength)
        return self.parent_strength


class RecoveryAdvisor:
    def __init__(
        self,
        parent_strength: float = DEFAULT_PARENT_STRENGTH,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        if parent_strength <= 0 or prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Bayesian strengths must be positive")
        self.parent_strength = float(parent_strength)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.counts: dict[str, list[float]] = {}
        self.examples: dict[str, list[str]] = {}

    @property
    def K(self) -> float:
        return self.parent_strength

    @K.setter
    def K(self, value: float) -> None:
        self.parent_strength = float(value)

    @staticmethod
    def k_full(error: str, stage: str, strategy: str) -> str:
        return f"error={error}|stage={stage}|strategy={strategy}"

    @staticmethod
    def k_error_strategy(error: str, strategy: str) -> str:
        return f"error={error}|strategy={strategy}"

    @staticmethod
    def k_strategy(strategy: str) -> str:
        return f"strategy={strategy}"

    def _success_total(self, key: str) -> tuple[float, float]:
        successes, failures = self.counts.get(key, (0.0, 0.0))
        return float(successes), float(successes) + float(failures)

    def predict(
        self, error: str, stage: str, strategy: str
    ) -> tuple[float, int, int]:
        successes, total = self._success_total("global")
        probability = (self.prior_alpha + successes) / (
            self.prior_alpha + self.prior_beta + total
        )
        for key in (
            self.k_strategy(strategy),
            self.k_error_strategy(error, strategy),
            self.k_full(error, stage, strategy),
        ):
            successes, total = self._success_total(key)
            probability = (
                successes + self.parent_strength * probability
            ) / (total + self.parent_strength)
        exact_successes, exact_total = self._success_total(
            self.k_full(error, stage, strategy)
        )
        return probability, int(exact_successes), int(exact_total)

    def update(
        self,
        error: str,
        stage: str,
        strategy: str,
        success: int,
        excerpt: str | None = None,
    ) -> None:
        if success not in {0, 1}:
            raise ValueError("success must be 0 or 1")
        for key in (
            "global",
            self.k_strategy(strategy),
            self.k_error_strategy(error, strategy),
            self.k_full(error, stage, strategy),
        ):
            successes, failures = self.counts.setdefault(key, [0.0, 0.0])
            if success:
                self.counts[key][0] = successes + 1
            else:
                self.counts[key][1] = failures + 1
        if success and excerpt:
            key = self.k_full(error, stage, strategy)
            examples = self.examples.setdefault(key, [])
            bounded = excerpt[:1500]
            if bounded not in examples and len(examples) < RECOVERY_MAX_EXCERPTS:
                examples.append(bounded)

    def rank(
        self, error: str, stage: str, strategies: dict[str, str]
    ) -> list[dict]:
        ranked = []
        for name, playbook in strategies.items():
            probability, successes, total = self.predict(error, stage, name)
            ranked.append(
                {
                    "strategy": name,
                    "probability": probability,
                    "successes": successes,
                    "total": total,
                    "playbook": playbook,
                    "excerpts": list(
                        self.examples.get(self.k_full(error, stage, name), [])
                    ),
                }
            )
        ranked.sort(key=lambda row: -row["probability"])
        return ranked

    def has_data(self, error: str, stage: str, strategies: dict[str, str]) -> bool:
        for name in strategies:
            if self.counts.get(self.k_full(error, stage, name)):
                return True
            if self.counts.get(self.k_error_strategy(error, name)):
                return True
        return False


class AgentModel:
    def __init__(
        self,
        retry: HierarchicalRetry | None = None,
        recovery: RecoveryAdvisor | None = None,
        metadata: dict | None = None,
    ):
        self.retry = retry or HierarchicalRetry()
        self.recovery = recovery or RecoveryAdvisor()
        self.metadata = dict(metadata or {})

    def to_dict(self) -> dict:
        return {
            "counts": self.retry.counts,
            "prior_alpha": self.retry.prior_alpha,
            "prior_beta": self.retry.prior_beta,
            "parent_strength": self.retry.parent_strength,
            "recovery_advisor": {
                "counts": self.recovery.counts,
                "examples": self.recovery.examples,
                "prior_alpha": self.recovery.prior_alpha,
                "prior_beta": self.recovery.prior_beta,
                "parent_strength": self.recovery.parent_strength,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> AgentModel:
        retry = HierarchicalRetry(
            parent_strength=payload.get(
                "parent_strength", DEFAULT_PARENT_STRENGTH
            ),
            prior_alpha=payload.get("prior_alpha", 1.0),
            prior_beta=payload.get("prior_beta", 1.0),
        )
        retry.counts = {
            key: [float(value[0]), float(value[1])]
            for key, value in payload.get("counts", {}).items()
        }
        recovery_payload = payload.get("recovery_advisor", {})
        recovery = RecoveryAdvisor(
            parent_strength=recovery_payload.get(
                "parent_strength", DEFAULT_PARENT_STRENGTH
            ),
            prior_alpha=recovery_payload.get("prior_alpha", 1.0),
            prior_beta=recovery_payload.get("prior_beta", 1.0),
        )
        recovery.counts = {
            key: [float(value[0]), float(value[1])]
            for key, value in recovery_payload.get("counts", {}).items()
        }
        recovery.examples = {
            key: [str(item) for item in value]
            for key, value in recovery_payload.get("examples", {}).items()
        }
        return cls(retry, recovery, payload.get("metadata", {}))
