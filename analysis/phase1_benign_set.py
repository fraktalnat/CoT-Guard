"""Phase 1: the benign monitor-input file for TNR, built from existing no_side_task generations.

Collects the main-all cot_action monitor inputs that already exist for the set's problems (every
bigcodebench.<objective>.no_side_task folder holds a subset of the same unprompted Qwen3-14B generations),
keeps one record per distinct monitor prompt, and renumbers task_ids so gen/monitor.py sees no duplicates.
The output mirrors $COT_GLOBAL_DIRECTORY, so copy <out_root>/datasets onto the GPU box as is.

Run from the repo root:
    python -m analysis.phase1_benign_set
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config, read_jsonl
from utils import save_jsonl


def select_benign(samples: list[dict[str, Any]], problem_ids: set[str]) -> list[dict[str, Any]]:
    """Samples on the given problems, one per distinct monitor prompt, in input order."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for sample in samples:
        key = (sample["metadata"]["main_task_id"], sample["messages"][0]["content"])
        if key[0] in problem_ids and key not in seen:
            seen.add(key)
            out.append(sample)
    return out


def renumber(samples: list[dict[str, Any]], monitor_type: str) -> list[dict[str, Any]]:
    """task_id = <problem>_no_side_task:<index within problem>:<monitor_type>, sorted by problem."""
    counts: Counter[str] = Counter()
    out = []
    for sample in sorted(samples, key=lambda s: int(s["metadata"]["main_task_id"])):
        pid = sample["metadata"]["main_task_id"]
        out.append({**sample, "task_id": f"{pid}_no_side_task:{counts[pid]}:{monitor_type}"})
        counts[pid] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_benign_set.yaml")
    cfg = load_config(parser.parse_args().config)

    problem_ids = set(json.loads(Path(cfg["task_id_sets"]).read_text())[f"{cfg['id_set']}_SET"])
    input_name = f"main-all.{cfg['monitor_policy']}.{cfg['monitor_type']}.jsonl"
    samples: list[dict[str, Any]] = []
    for f in sorted(Path(cfg["monitor_inputs_root"]).expanduser().glob(f"{cfg['main_task']}.*.no_side_task/{cfg['generator']}--*/{input_name}")):
        samples += read_jsonl(f)
    records = renumber(select_benign(samples, problem_ids), cfg["monitor_type"])

    per_problem = Counter(r["metadata"]["main_task_id"] for r in records)
    missing = sorted(problem_ids - set(per_problem), key=int)
    print(f"{cfg['id_set']}_SET: {len(records)} benign records on {len(per_problem)}/{len(problem_ids)} problems")
    print("   per problem: " + ", ".join(f"{pid}:{n}" for pid, n in sorted(per_problem.items(), key=lambda kv: int(kv[0]))))
    print(f"   no benign record: {', '.join(missing) or 'none'}")

    folder = (
        Path(cfg["out_root"]) / "datasets" / "monitor"
        / f"{cfg['main_task']}.{cfg['benign_name']}.no_side_task" / f"{cfg['generator']}--{cfg['attack_rollouts']}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    save_jsonl(records, str(folder / input_name))
    print(f"   wrote {folder / input_name}")


if __name__ == "__main__":
    main()
