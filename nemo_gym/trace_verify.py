# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parseable gym_run phase / verify-detail lines.

Gated by NEMOGYM_TRACE_VERIFY=1. Lines go to stdout (gym.log for ng_run).
Format: gym_run KIND key=value key=value ...
Values are single tokens (spaces -> underscores) so tools/gym_run_trace.py
can split them.

See also NEMOGYM_TRACE_EPISODES in nemogym2mrl (trainer-side /run envelope).
"""

from __future__ import annotations

import itertools
import os
from typing import Any

TRACE_VERIFY = os.environ.get("NEMOGYM_TRACE_VERIFY", "0") == "1"
_SEQ = itertools.count()
_PID = os.getpid()


def next_run_id() -> str:
    return f"{_PID}.{next(_SEQ)}"


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value).replace(" ", "_")
    return text if text else "-"


def gym_run_log(kind: str, **fields: Any) -> None:
    if not TRACE_VERIFY:
        return
    parts = [f"gym_run {kind}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_fmt(value)}")
    print(" ".join(parts), flush=True)


def verify_trace_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull a whitelist of scoring details off a verify JSON body."""
    out: dict[str, Any] = {}
    if "reward" in payload and payload["reward"] is not None:
        out["reward"] = payload["reward"]
    if payload.get("unit_tests_time_taken") is not None:
        out["unit_tests_s"] = payload["unit_tests_time_taken"]
    if payload.get("n_tests") is not None:
        out["n_tests"] = payload["n_tests"]
    if payload.get("tests_completed") is not None:
        out["tests_completed"] = payload["tests_completed"]
    if "global_timeout" in payload:
        out["global_timeout"] = bool(payload["global_timeout"])
    if payload.get("error_code") is not None:
        out["error_code"] = payload["error_code"]
    if payload.get("error_message"):
        out["error_message"] = payload["error_message"]
    code = payload.get("extracted_model_code")
    if "extracted_model_code" in payload:
        out["extracted"] = bool(code)
    if payload.get("library_reward") is not None:
        out["library_reward"] = payload["library_reward"]
    if "judge_used" in payload:
        out["judge_used"] = bool(payload["judge_used"])
    elif "judge_evaluations" in payload:
        out["judge_used"] = payload["judge_evaluations"] is not None
    if payload.get("judge_elapsed") is not None:
        out["judge_elapsed"] = payload["judge_elapsed"]
    evals = payload.get("judge_evaluations")
    if evals is not None:
        out["n_judge_calls"] = len(evals)
        if evals:
            last = evals[-1]
            label = last.get("verdict_label") if isinstance(last, dict) else getattr(last, "verdict_label", None)
            if label:
                out["verdict"] = label
    if payload.get("n_judge_calls") is not None:
        out["n_judge_calls"] = payload["n_judge_calls"]
    if payload.get("verdict"):
        out["verdict"] = payload["verdict"]
    if payload.get("cohort_wait_s") is not None:
        out["cohort_wait_s"] = payload["cohort_wait_s"]
    if payload.get("score_elapsed_s") is not None:
        out["score_elapsed_s"] = payload["score_elapsed_s"]
    if payload.get("n_cohort") is not None:
        out["n_cohort"] = payload["n_cohort"]
    if "cohort_timed_out" in payload:
        out["cohort_timed_out"] = bool(payload["cohort_timed_out"])
    if "scored_here" in payload:
        out["scored_here"] = bool(payload["scored_here"])
    return out
