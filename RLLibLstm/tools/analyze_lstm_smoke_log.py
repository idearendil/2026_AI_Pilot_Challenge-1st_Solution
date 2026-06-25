"""Analyze SAC LSTM smoke logs for sequence replay contract issues.

The checker focuses on the failure modes seen while porting SAC LSTM:

* replay batches padded to ``T`` but carrying only 1-step ``seq_lens``;
* first probed observation feature containing one real value followed by zeros;
* missing state input/output markers in the train path.

It also prints the probed time-axis values so a human can inspect whether a
domain-specific feature appears reversed.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DEBUG_RE = re.compile(
    r"\[DogFightEnv\]\[RLlibSAC\]\[LSTM_IO\].*?"
    r"label=(?P<label>\w+).*?"
    r"obs_shape=(?P<obs_shape>\([^)]+\)|None).*?"
    r"next_obs_shape=(?P<next_obs_shape>\([^)]+\)|None).*?"
    r"actions_shape=(?P<actions_shape>\([^)]+\)|None).*?"
    r"seq_lens=(?P<seq_lens>tensor\(\[[^\]]*\]\)|None).*?"
    r"state_in=(?P<state_in>.*?) "
    r"next_state_in=(?P<next_state_in>.*?) "
    r"state_out=(?P<state_out>.*?) "
    r"next_state_out=(?P<next_state_out>.*?) "
    r"obs_probe_first_feature=\[(?P<probe>[^\]]*)\]",
    re.DOTALL,
)
Q_DEBUG_RE = re.compile(
    r"\{'label': '(?P<label>[^']+)'.*?"
    r"'q_concat_shape': (?P<q_concat_shape>\([^)]+\)).*?"
    r"'q_state_in': \{'h': (?P<q_h_shape>\([^)]+\)), "
    r"'c': (?P<q_c_shape>\([^)]+\))\}.*?"
    r"'q_state_out': \{'h': (?P<q_h_out_shape>\([^)]+\)), "
    r"'c': (?P<q_c_out_shape>\([^)]+\))\}.*?"
    r"'q_out_shape': (?P<q_out_shape>\([^)]+\)).*?"
    r"'q_action_probe_first_feature': \[(?P<action_probe>[^\]]*)\]",
    re.DOTALL,
)
EXPECTED_Q_DEBUG_LABELS = {
    "qf_rollout",
    "qf_twin_rollout",
    "q_curr_qf",
    "q_curr_qf_twin",
    "target_qf",
    "target_qf_twin",
}


@dataclass
class DebugRecord:
    """Parsed RLlib SAC LSTM debug record."""

    label: str
    obs_shape: tuple[int, ...] | None
    next_obs_shape: tuple[int, ...] | None
    actions_shape: tuple[int, ...] | None
    seq_lens: list[int]
    state_in: str
    next_state_in: str
    state_out: str
    next_state_out: str
    probe: list[float]


@dataclass
class QDebugRecord:
    """Parsed DogFightEnv recurrent Q debug record."""

    label: str
    q_concat_shape: tuple[int, ...]
    q_h_shape: tuple[int, ...]
    q_c_shape: tuple[int, ...]
    q_h_out_shape: tuple[int, ...]
    q_c_out_shape: tuple[int, ...]
    q_out_shape: tuple[int, ...]
    action_probe: list[float]


def _clean_text(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_shape(value: str) -> tuple[int, ...] | None:
    if value == "None":
        return None
    return tuple(int(item.strip()) for item in value.strip("()").split(",") if item.strip())


def _parse_seq_lens(value: str) -> list[int]:
    if value == "None":
        return []
    return [int(item) for item in re.findall(r"-?\d+", value)]


def _parse_probe(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", value)]


def _parse_records(text: str) -> list[DebugRecord]:
    records: list[DebugRecord] = []
    for match in DEBUG_RE.finditer(_clean_text(text)):
        records.append(
            DebugRecord(
                label=match.group("label"),
                obs_shape=_parse_shape(match.group("obs_shape")),
                next_obs_shape=_parse_shape(match.group("next_obs_shape")),
                actions_shape=_parse_shape(match.group("actions_shape")),
                seq_lens=_parse_seq_lens(match.group("seq_lens")),
                state_in=match.group("state_in"),
                next_state_in=match.group("next_state_in"),
                state_out=match.group("state_out"),
                next_state_out=match.group("next_state_out"),
                probe=_parse_probe(match.group("probe")),
            )
        )
    return records


def _parse_q_debug_records(text: str) -> list[QDebugRecord]:
    records: list[QDebugRecord] = []
    for match in Q_DEBUG_RE.finditer(_clean_text(text)):
        records.append(
            QDebugRecord(
                label=match.group("label"),
                q_concat_shape=_parse_shape(match.group("q_concat_shape")) or (),
                q_h_shape=_parse_shape(match.group("q_h_shape")) or (),
                q_c_shape=_parse_shape(match.group("q_c_shape")) or (),
                q_h_out_shape=_parse_shape(match.group("q_h_out_shape")) or (),
                q_c_out_shape=_parse_shape(match.group("q_c_out_shape")) or (),
                q_out_shape=_parse_shape(match.group("q_out_shape")) or (),
                action_probe=_parse_probe(match.group("action_probe")),
            )
        )
    return records


def _direction(values: list[float]) -> str:
    if len(values) < 2:
        return "single"
    diffs = [b - a for a, b in zip(values, values[1:])]
    positive = sum(diff > 1e-8 for diff in diffs)
    negative = sum(diff < -1e-8 for diff in diffs)
    if positive and not negative:
        return "increasing"
    if negative and not positive:
        return "decreasing"
    if positive or negative:
        return "mixed"
    return "flat"


def _looks_padded_probe(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    return abs(values[0]) > 1e-8 and all(abs(value) <= 1e-8 for value in values[1:])


def _format_counts(values: list[int]) -> str:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _analyze_q_debug_records(
    q_records: list[QDebugRecord],
    train_records: list[DebugRecord],
    max_seq_len: int,
    expect_q_debug: bool,
) -> list[str]:
    """Validate actor_critic recurrent Q I/O contracts."""
    errors: list[str] = []
    if not q_records:
        if expect_q_debug:
            errors.append("No q_debug records found for actor_critic scope.")
        return errors

    observed_labels = {record.label for record in q_records}
    missing_labels = EXPECTED_Q_DEBUG_LABELS - observed_labels
    if missing_labels:
        errors.append(f"Missing q_debug labels: {sorted(missing_labels)}")

    obs_dim = None
    action_dim = None
    for record in train_records:
        if record.obs_shape and len(record.obs_shape) >= 3:
            obs_dim = record.obs_shape[-1]
        if record.actions_shape and len(record.actions_shape) >= 3:
            action_dim = record.actions_shape[-1]
        if obs_dim is not None and action_dim is not None:
            break

    for index, record in enumerate(q_records):
        if len(record.q_concat_shape) != 3:
            errors.append(f"Q record {index} has invalid concat shape {record.q_concat_shape}.")
            continue
        batch_size, time_dim, concat_dim = record.q_concat_shape
        if time_dim != max_seq_len:
            errors.append(
                f"Q record {index} T dimension is {time_dim}, expected {max_seq_len}."
            )
        if obs_dim is not None and action_dim is not None:
            expected_concat_dim = obs_dim + action_dim
            if concat_dim != expected_concat_dim:
                errors.append(
                    f"Q record {index} concat dim is {concat_dim}, "
                    f"expected obs+action={expected_concat_dim}."
                )
        expected_state_prefix = (batch_size, 1)
        for state_name, shape in (
            ("q_h_shape", record.q_h_shape),
            ("q_c_shape", record.q_c_shape),
            ("q_h_out_shape", record.q_h_out_shape),
            ("q_c_out_shape", record.q_c_out_shape),
        ):
            if len(shape) != 3 or shape[:2] != expected_state_prefix:
                errors.append(
                    f"Q record {index} {state_name}={shape}, "
                    f"expected prefix {expected_state_prefix}."
                )
        if record.q_out_shape != (batch_size, time_dim):
            errors.append(
                f"Q record {index} q_out_shape={record.q_out_shape}, "
                f"expected {(batch_size, time_dim)}."
            )
        if len(record.action_probe) != time_dim:
            errors.append(
                f"Q record {index} action probe length is {len(record.action_probe)}, "
                f"expected {time_dim}."
            )
        if _looks_padded_probe(record.action_probe):
            errors.append(f"Q record {index} action probe looks padded.")
    return errors


def analyze_log(path: Path, max_seq_len: int, expect_q_debug: bool = False) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    records = _parse_records(text)
    q_records = _parse_q_debug_records(text)
    train_records = [record for record in records if record.label == "train_continuous"]
    inference_records = [record for record in records if record.label == "inference"]
    errors: list[str] = []
    warnings: list[str] = []

    if not inference_records:
        warnings.append("No inference LSTM debug records found.")
    if not train_records:
        errors.append("No train_continuous LSTM debug records found.")

    all_seq_lens = [value for record in train_records for value in record.seq_lens]
    if train_records and not all_seq_lens:
        errors.append("No seq_lens were parsed from train_continuous records.")
    if all_seq_lens and max(all_seq_lens) < max_seq_len:
        errors.append(
            "No sampled sequence reached max_seq_len. "
            f"max_seq_len={max_seq_len}, observed_max={max(all_seq_lens)}"
        )
    if all_seq_lens and all(value == 1 for value in all_seq_lens):
        errors.append("All train seq_lens are 1. Replay is still behaving like 1-step sampling.")

    for index, record in enumerate(train_records):
        if record.obs_shape and len(record.obs_shape) >= 2 and record.obs_shape[1] != max_seq_len:
            errors.append(
                f"Record {index} obs T dimension is {record.obs_shape[1]}, "
                f"expected {max_seq_len}."
            )
        if record.next_obs_shape and len(record.next_obs_shape) >= 2:
            if record.next_obs_shape[1] != max_seq_len:
                errors.append(
                    f"Record {index} next_obs T dimension is {record.next_obs_shape[1]}, "
                    f"expected {max_seq_len}."
                )
        if record.actions_shape and len(record.actions_shape) >= 2:
            if record.actions_shape[1] != max_seq_len:
                errors.append(
                    f"Record {index} action T dimension is {record.actions_shape[1]}, "
                    f"expected {max_seq_len}."
                )
        if "None" in (record.state_in, record.state_out):
            errors.append(f"Record {index} is missing actor recurrent state_in/state_out.")
        if _looks_padded_probe(record.probe):
            errors.append(
                f"Record {index} probe looks padded: first value is non-zero, "
                "remaining values are zero."
            )

    errors.extend(
        _analyze_q_debug_records(
            q_records=q_records,
            train_records=train_records,
            max_seq_len=max_seq_len,
            expect_q_debug=expect_q_debug,
        )
    )

    print("[DogFightEnv][lstm_log_analyzer] file=", path)
    print("[DogFightEnv][lstm_log_analyzer] inference_records=", len(inference_records))
    print("[DogFightEnv][lstm_log_analyzer] train_records=", len(train_records))
    if all_seq_lens:
        print("[DogFightEnv][lstm_log_analyzer] seq_lens_counts=", _format_counts(all_seq_lens))
        print("[DogFightEnv][lstm_log_analyzer] seq_lens_max=", max(all_seq_lens))
    print("[DogFightEnv][lstm_log_analyzer] q_debug_records=", len(q_records))
    if q_records:
        print(
            "[DogFightEnv][lstm_log_analyzer] q_debug_label_counts=",
            _format_counts([record.label for record in q_records]),
        )

    for index, record in enumerate(train_records[:5]):
        print(
            "[DogFightEnv][lstm_log_analyzer] "
            f"sample={index} obs_shape={record.obs_shape} "
            f"next_obs_shape={record.next_obs_shape} actions_shape={record.actions_shape} "
            f"probe_direction={_direction(record.probe)} probe={record.probe}"
        )
    for index, record in enumerate(q_records[:6]):
        print(
            "[DogFightEnv][lstm_log_analyzer] "
            f"q_sample={index} label={record.label} "
            f"q_concat_shape={record.q_concat_shape} q_state_h={record.q_h_shape} "
            f"q_out_shape={record.q_out_shape} "
            f"action_probe_direction={_direction(record.action_probe)} "
            f"action_probe={record.action_probe}"
        )

    for warning in warnings:
        print("[DogFightEnv][lstm_log_analyzer][WARN]", warning)
    for error in errors:
        print("[DogFightEnv][lstm_log_analyzer][FAIL]", error)

    if errors:
        print("[DogFightEnv][lstm_log_analyzer] result=FAIL")
        return 1

    print("[DogFightEnv][lstm_log_analyzer] result=PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_path",
        nargs="?",
        default="DogFightEnv/MyTrainEnv/lstm_smoke_train.txt",
    )
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument(
        "--expect-q-debug",
        action="store_true",
        help="Require actor_critic recurrent Q debug records and validate their shapes.",
    )
    args = parser.parse_args()
    raise SystemExit(
        analyze_log(
            Path(args.log_path),
            args.max_seq_len,
            expect_q_debug=args.expect_q_debug,
        )
    )


if __name__ == "__main__":
    main()
