from __future__ import annotations

import re
from collections.abc import Callable

from chipmem.statistical.model import (
    NUDGE_THRESHOLD,
    PLAN_MAX_STEPS,
    AgentModel,
    Feat,
)

WARNING_TEMPLATE = (
    "You have run {command} {count} times, failing with {error} each time. "
    "Historically, retries in this situation succeed only {percent} percent "
    "of the time. Consider a different approach."
)

PLANNER_PROMPT = """A tool call in an autonomous engineering session keeps failing and needs a recovery plan.

Current failure: command {command}, error class {error}, stage {stage}.
Retry success probability (historical): {retry_percent} percent.

Ranked recovery strategies (probability, successes/total, playbook):
{strategy_lines}

Historical excerpts from successful recoveries:
{excerpts}

Recent attempts:
{recent}

Write a recovery plan with AT MOST {max_steps} numbered steps, then exactly
one line starting with "SUCCESS CHECK:" and one line starting with
"STOP IF:". Base the steps on the top strategies' playbooks and excerpts.
Keep it terse; the agent executes it with its normal tools."""


class WarningGenerator:
    def __init__(
        self,
        model: AgentModel,
        features,
        threshold: float = NUDGE_THRESHOLD,
        planner: Callable[[str], str] | None = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model = model
        self.features = features
        self.threshold = float(threshold)
        self.planner = planner

    def maybe_warn(
        self,
        feature: Feat,
        *,
        stage: str = "",
        recent: str = "",
    ) -> str | None:
        if feature.rc < 1 or feature.pr != "fail":
            return None
        probability, _ = self.model.retry.predict(feature)
        if probability >= self.threshold:
            return None
        strategies = self.features.strategies()
        if self.planner is not None and self.model.recovery.has_data(
            feature.pec, stage, strategies
        ):
            plan = self._plan(feature, probability, stage, strategies, recent)
            if plan:
                return plan
        return WARNING_TEMPLATE.format(
            command=self.features.readable_command(feature.ct),
            count=feature.rc,
            error=feature.pec,
            percent=round(100 * probability),
        )

    def _plan(
        self,
        feature: Feat,
        probability: float,
        stage: str,
        strategies: dict[str, str],
        recent: str,
    ) -> str | None:
        ranked = self.model.recovery.rank(feature.pec, stage, strategies)
        strategy_lines = "\n".join(
            f"- {row['strategy']}: {row['probability']:.0%} "
            f"({row['successes']}/{row['total']}) — {row['playbook']}"
            for row in ranked[:5]
        )
        excerpts = "\n".join(
            excerpt for row in ranked[:3] for excerpt in row["excerpts"]
        ) or "(none)"
        prompt = PLANNER_PROMPT.format(
            command=self.features.readable_command(feature.ct),
            error=feature.pec,
            stage=stage or "(unstaged)",
            retry_percent=round(100 * probability),
            strategy_lines=strategy_lines,
            excerpts=excerpts,
            recent=recent[:2000] or "(none)",
            max_steps=PLAN_MAX_STEPS,
        )
        planner = self.planner
        if planner is None:
            return None
        try:
            reply = planner(prompt)
        except Exception:  # noqa: BLE001
            return None
        if not self.plan_valid(reply):
            return None
        return "Recovery plan (historical evidence, advisory):\n" + reply.strip()

    @staticmethod
    def plan_valid(reply: str | None) -> bool:
        if not reply:
            return False
        lines = reply.splitlines()
        steps = sum(1 for line in lines if re.match(r"\s*\d+[.)]", line))
        checks = sum(
            1 for line in lines if line.strip().startswith("SUCCESS CHECK:")
        )
        stops = sum(1 for line in lines if line.strip().startswith("STOP IF:"))
        return 1 <= steps <= PLAN_MAX_STEPS and checks == 1 and stops == 1
