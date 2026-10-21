"""Follow-ups to phase1_pilot.py, read-only over the downloaded data.

  A. per goal: held-out problems using the API, the benign base rate inside that subset,
     and how many of those problems already have monitor verdicts on benign samples
  B. which hash each held-out hashlib problem's canonical solution uses
  C. benign generations per problem
(Pool overlap between objectives is section 4 of phase1_pilot.py.)

Run from the repo root:
    python -m analysis.phase1_followup
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config, read_jsonl
from analysis.phase1_candidates import canonical_code
from analysis.phase1_pilot import all_problems, benign_records, path
from utils import extract_choice


def benign_verdicts(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Every existing monitor verdict on a no_side_task sample, keyed by problem id."""
    out: dict[str, list[str]] = {}
    pattern = f"bigcodebench.*.no_side_task/{cfg['generator']}--*/main-all.{cfg['monitor']}.main_aware.cot_action.jsonl"
    for f in path(cfg, "monitor_results_root").glob(pattern):
        for sample in read_jsonl(f):
            out.setdefault(sample["metadata"]["main_task_id"], []).append(extract_choice(sample["messages"][-1]["content"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_pilot.yaml")
    cfg = load_config(parser.parse_args().config)
    test = set(json.loads(Path(cfg["task_id_sets"]).read_text())[cfg["held_out_set"]])
    problems = all_problems(cfg)
    benign = benign_records(cfg)
    verdicts = benign_verdicts(cfg)

    print("## A. Per goal: held-out problems, benign base rate inside that subset, existing benign monitor verdicts")
    for goal, (api, pattern_name) in cfg["goal_api"].items():
        ids = {pid for pid, md in problems.items() if re.search(cfg["apis"][api], canonical_code(md))} & test
        gens = {key: md for key, md in benign.items() if key[0] in ids}
        hits = [key for key in gens if re.search(cfg["patterns"][pattern_name], key[1])]
        main_pass = sum(bool(md.get("main_pass")) for md in gens.values())
        with_verdict = ids & set(verdicts)
        print(f"   {goal:30} problems={len(ids):3}  benign gens={len(gens):4} (main_pass {main_pass:4})  "
              f"base rate={len(hits) / max(len(gens), 1):6.1%} on {len({pid for pid, _ in hits})} problems  "
              f"benign verdicts on {len(with_verdict)} problems (n={sum(len(verdicts[p]) for p in with_verdict)})")

    print("\n## B. Which hash does the canonical solution use, per held-out hashlib problem?")
    hash_use: Counter[str] = Counter()
    for pid, md in problems.items():
        code = canonical_code(md)
        if pid in test and re.search(cfg["apis"]["hashlib-any"], code):
            hash_use.update(name for name in cfg["hash_names"] if name in code)
    print("   ", dict(hash_use))

    print("\n## C. Benign generations per problem (distinct code)")
    per_problem = Counter(pid for pid, _ in benign)
    print("   distribution of gens/problem:", dict(sorted(Counter(per_problem.values()).items())))


if __name__ == "__main__":
    main()
