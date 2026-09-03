from __future__ import annotations

from chipmem.statistical.model import RECOVERY_HORIZON, Feat


def extract_rows(transcript: list[dict], features) -> list[dict]:
    rows: list[dict] = []
    per_type_count: dict[str, int] = {}
    per_type_last: dict[str, tuple[int, str]] = {}
    for index, call in enumerate(transcript):
        command_type = features.command_type(
            call.get("tool", ""), call.get("arguments")
        )
        prior_count = per_type_count.get(command_type, 0)
        if command_type in per_type_last:
            last_success, last_error = per_type_last[command_type]
            previous_result = "success" if last_success else "fail"
            previous_error = "ok" if last_success else last_error
        else:
            previous_result, previous_error = "none", "none"
        feature = Feat(
            ct=command_type,
            rc=prior_count,
            pr=previous_result,
            pec=previous_error,
            te=index,
        )
        success, error = features.classify(
            call.get("tool", ""), call.get("output", "")
        )
        rows.append(
            {
                "feat": feature,
                "success": success,
                "error_class": error,
                "cmd_type": command_type,
            }
        )
        per_type_count[command_type] = prior_count + 1
        per_type_last[command_type] = (success, error)
    return rows


def pre_call_feat(
    transcript: list[dict], tool: str, arguments, features
) -> Feat:
    command_type = features.command_type(tool, arguments)
    prior_count = 0
    last: tuple[int, str] | None = None
    for call in transcript:
        prior_type = features.command_type(
            call.get("tool", ""), call.get("arguments")
        )
        if prior_type == command_type:
            prior_count += 1
            last = features.classify(
                call.get("tool", ""), call.get("output", "")
            )
    if last is None:
        previous_result, previous_error = "none", "none"
    else:
        previous_result = "success" if last[0] else "fail"
        previous_error = "ok" if last[0] else last[1]
    return Feat(
        ct=command_type,
        rc=prior_count,
        pr=previous_result,
        pec=previous_error,
        te=len(transcript),
    )


def _classify_strategy(call: dict, previous_call: dict, features) -> str | None:
    strategies = features.strategies()
    tool = call.get("tool", "")
    arguments = str(call.get("arguments", ""))
    previous_arguments = str(previous_call.get("arguments", ""))
    low = arguments.lower()
    command_type = features.command_type(tool, call.get("arguments"))
    previous_type = features.command_type(
        previous_call.get("tool", ""), previous_call.get("arguments")
    )
    if (
        command_type == previous_type
        and arguments == previous_arguments
        and "unchanged_retry" in strategies
    ):
        return "unchanged_retry"
    if (
        tool == "read_file" or "cat " in low or "log" in low
    ) and "inspect_logs" in strategies:
        return "inspect_logs"
    if (
        low.startswith("/") or " /" in low
    ) and "absolute_hdl_paths" in strategies:
        return "absolute_hdl_paths"
    if tool in {"write_file", "edit_file"} and "simplify_edit" in strategies:
        return "simplify_edit"
    if (
        command_type == previous_type
        and arguments != previous_arguments
        and "change_tool_setup" in strategies
    ):
        return "change_tool_setup"
    if (
        tool == "shell"
        and any(word in low for word in ("ls", "find", "grep"))
        and "rebuild_filelist" in strategies
    ):
        return "rebuild_filelist"
    return None


def extract_recovery(transcript: list[dict], features) -> list[dict]:
    rows = extract_rows(transcript, features)
    stage_order = list(getattr(features, "stage_order", []))
    staged = bool(stage_order)
    order = {stage: index for index, stage in enumerate(stage_order)}
    events: list[dict] = []
    for index, row in enumerate(rows):
        if row["success"]:
            continue
        error = row["error_class"]
        failed_type = row["cmd_type"]
        stage = features.stage_of(failed_type) if staged else failed_type
        window = rows[index + 1 : index + 1 + RECOVERY_HORIZON]
        attempted: dict[str, int] = {}
        recovered_at = None
        for offset, window_row in enumerate(window):
            call = transcript[index + 1 + offset]
            previous_call = transcript[index + offset]
            strategy = _classify_strategy(call, previous_call, features)
            if strategy:
                attempted.setdefault(strategy, 0)
            if window_row["success"]:
                if staged:
                    later_stage = features.stage_of(window_row["cmd_type"])
                    if stage in order and later_stage in order:
                        if order[later_stage] >= order[stage]:
                            recovered_at = offset
                    elif window_row["cmd_type"] == failed_type:
                        recovered_at = offset
                elif window_row["cmd_type"] == failed_type:
                    recovered_at = offset
                if recovered_at is not None:
                    for name in attempted:
                        attempted[name] = 1
                    break
        stop = (
            index + 2 + recovered_at
            if recovered_at is not None
            else index + 1 + len(window)
        )
        excerpt_lines = []
        context_window = []
        for call_index, call in enumerate(transcript[index:stop], index):
            bounded = {
                "idx": call_index,
                "tool": str(call.get("tool", ""))[:80],
                "arguments": str(call.get("arguments", ""))[:300],
                "output": str(call.get("output", ""))[:500],
            }
            context_window.append(bounded)
            excerpt_lines.append(
                f"[{call_index}] {bounded['tool']}({bounded['arguments']}) -> "
                f"{bounded['output']}"
            )
        excerpt = "\n".join(excerpt_lines)[:1500] if recovered_at is not None else None
        for strategy, success in attempted.items():
            events.append(
                {
                    "error": error,
                    "stage": stage,
                    "strategy": strategy,
                    "success": success,
                    "recovered": success,
                    "failure_idx": index,
                    "plan_excerpt": excerpt if success else None,
                    "context_window": context_window,
                    "excerpt": excerpt if success else None,
                }
            )
    return events
