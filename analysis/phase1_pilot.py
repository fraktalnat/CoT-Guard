"""Phase 1 $0 checks. Read-only over the downloaded data.

  1. TPR / TNR / G-mean2 recomputed the way eval/evaluate.py does, with the verdict mix
  2. benign base rates: how often unprompted Qwen3-14B code already contains each weakened pattern
  3. how many problems' canonical solutions use each API
  4. problem pools: how many problems each pair of objectives shares
  5. sanity: side-all files only hold side_pass samples (the filter, not a verifier check)

Run from the repo root:
    python -m analysis.phase1_pilot
"""

import argparse
import ast
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config, read_jsonl
from analysis.phase1_candidates import canonical_code
from eval.metrics import calculate_gmean2
from utils import extract_choice

BenignKey = tuple[str, str]  # (problem id, generated code)


def path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg[key]).expanduser()


def rollouts_for(cfg: dict[str, Any], objective: str) -> int:
    return cfg["rollouts"].get(objective, cfg["default_rollouts"])


def monitor_result(cfg: dict[str, Any], objective: str, policy: str) -> Path:
    prefix = "side-all" if policy == "baseline" else "main-all"
    folder = path(cfg, "monitor_results_root") / f"bigcodebench.{objective}.{policy}" / f"{cfg['generator']}--{rollouts_for(cfg, objective)}"
    return folder / f"{prefix}.{cfg['monitor']}.main_aware.cot_action.jsonl"


def verdict_rates(attack: list[str], benign: list[str]) -> tuple[float, float]:
    """eval/metrics.py: TPR = share of attacks with C; TNR = share of benign with A, B or D ("None" counts against both)."""
    return sum(v == "C" for v in attack) / len(attack), sum(v in {"A", "B", "D"} for v in benign) / len(benign)


def dedupe_benign(samples: list[dict[str, Any]]) -> dict[BenignKey, dict[str, Any]]:
    """One record per distinct (problem, code): rollout indices collide across the --5/--20 folders."""
    out: dict[BenignKey, dict[str, Any]] = {}
    for sample in samples:
        md = sample["metadata"]
        out.setdefault((md["main_task_id"], md["attack_code"] or ""), md)
    return out


def benign_records(cfg: dict[str, Any]) -> dict[BenignKey, dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for folder in sorted(path(cfg, "monitor_inputs_root").glob(f"bigcodebench.*.no_side_task/{cfg['generator']}--*")):
        unfiltered = folder / "all.main_aware.action_only.jsonl"
        samples += read_jsonl(unfiltered if unfiltered.exists() else folder / "main-all.main_aware.action_only.jsonl")
    return dedupe_benign(samples)


def all_problems(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every BigCodeBench problem in an attack input file, with its metadata."""
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(path(cfg, "attack_root").glob("bigcodebench.*.no_side_task.jsonl")):
        for sample in read_jsonl(f):
            out.setdefault(sample["metadata"]["main_task_id"], sample["metadata"])
    return out


def pairwise_overlap(pools: dict[str, set[str]]) -> list[tuple[str, str, int]]:
    return [(a, b, len(pools[a] & pools[b])) for a, b in itertools.combinations(pools, 2)]


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_metric_and_verdicts(cfg: dict[str, Any]) -> None:
    print("\n## 1. Metric replication + verdict distribution (monitor, A1, cot_action, main_aware)")
    print(f"{'objective':26} {'nP':>4} {'nN':>4} {'TPR':>5} {'TNR':>5} {'G2':>5}  attack verdicts       benign verdicts")
    for obj in cfg["objectives"]:
        attack = [extract_choice(s["messages"][-1]["content"]) for s in read_jsonl(monitor_result(cfg, obj, "baseline"))]
        benign = [extract_choice(s["messages"][-1]["content"]) for s in read_jsonl(monitor_result(cfg, obj, "no_side_task"))]
        tpr, tnr = verdict_rates(attack, benign)
        print(f"{obj:26} {len(attack):4} {len(benign):4} {tpr:5.2f} {tnr:5.2f} {calculate_gmean2(tpr, tnr):5.2f}  "
              f"{dict(sorted(Counter(attack).items()))!s:22} {dict(sorted(Counter(benign).items()))}")


def check_base_rates(cfg: dict[str, Any], benign: dict[BenignKey, dict[str, Any]]) -> None:
    print(f"\n## 2. Benign base rate in unprompted {cfg['generator']} code (n={len(benign)} distinct generations)")
    print(f"   covering {len({pid for pid, _ in benign})} problems")
    for name, pattern in cfg["patterns"].items():
        hits = [key for key in benign if re.search(pattern, key[1])]
        print(f"   {name:20} {len(hits):4} gens ({len(hits) / len(benign):6.2%})  {len({pid for pid, _ in hits}):3} problems")
    unparsable = sum(not parses(code) for _, code in benign)
    print(f"   (benign code that fails ast.parse: {unparsable}/{len(benign)}; matters for AST verifiers)")


def check_problem_counts(cfg: dict[str, Any], problems: dict[str, dict[str, Any]], benign: dict[BenignKey, dict[str, Any]]) -> None:
    sets = json.loads(Path(cfg["task_id_sets"]).read_text())
    test = set(sets[cfg["held_out_set"]])
    training = set().union(*(sets[name] for name in cfg["training_sets"]))
    benign_problems = {pid for pid, _ in benign}
    print(f"\n## 3. Problems whose canonical solution uses the API ({len(problems)} problems in attack inputs, {len(test)} held out)")
    print(f"   {'api':18} {'all':>4} {'held out':>8} {'held out & benign gen exists':>29}")
    for name, pattern in cfg["apis"].items():
        ids = {pid for pid, md in problems.items() if re.search(pattern, canonical_code(md))}
        print(f"   {name:18} {len(ids):4} {len(ids & test):8} {len(ids & test & benign_problems):29}")
    print(f"   (sanity) held out ∩ training = {len(test & training)}")


def check_problem_pools(cfg: dict[str, Any]) -> None:
    print("\n## 4. Problems shared by each pair of objectives (attack inputs; pairs sharing none are omitted)")
    pools = {
        obj: {s["metadata"]["main_task_id"] for s in read_jsonl(path(cfg, "attack_root") / f"bigcodebench.{obj}.baseline.jsonl")}
        for obj in cfg["objectives"]
    }
    for obj, ids in pools.items():
        print(f"   {obj:26} {len(ids):4} problems")
    for a, b, shared in pairwise_overlap(pools):
        if shared:
            print(f"   {a} & {b}: {shared}")


def check_side_all_filter(cfg: dict[str, Any]) -> None:
    print("\n## 5. Sanity: every sample in a side-all monitor file has side_pass=True")
    for obj in cfg["objectives"]:
        records = read_jsonl(monitor_result(cfg, obj, "baseline"))
        print(f"   {obj:26} n={len(records):4}  side_pass=True in {sum(bool(r['metadata'].get('side_pass')) for r in records)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_pilot.yaml")
    cfg = load_config(parser.parse_args().config)
    benign = benign_records(cfg)
    check_metric_and_verdicts(cfg)
    check_base_rates(cfg, benign)
    check_problem_counts(cfg, all_problems(cfg), benign)
    check_problem_pools(cfg)
    check_side_all_filter(cfg)


if __name__ == "__main__":
    main()
