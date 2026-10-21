"""Phase 1 step 2: the review sheet for the silent-errors hand-check.

The candidate scan drops any problem whose statement matches `raise|exception|error`, which is crude.
For each dropped problem that would otherwise enter the set (no benign hits, separable handler), this prints
the statement sentences that matched next to what each canonical handler catches and does, so a reader can decide:
keep the problem if silencing its handler does not contradict the statement, i.e. the statement does not
specify the behaviour that handler implements. Decisions go in `handcheck_keep` in phase1_problem_sets.yaml.

Run from the repo root, after analysis.phase1_candidates:
    python -m analysis.phase1_handcheck > analysis/outputs/phase1_handcheck.md
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config, read_jsonl
from analysis.phase1_problem_sets import has_separable_handler


def handler_summaries(code: str) -> list[str]:
    """One line per except handler: what it catches -> its first statement."""
    return [
        f"{ast.unparse(node.type) if node.type else '(bare)'} -> {ast.unparse(node.body[0]).splitlines()[0]}"
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.ExceptHandler)
    ]


def matching_sentences(statement: str, spec_words: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", statement)
    return [s.strip() for s in sentences if re.search(spec_words, s, re.IGNORECASE)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_problem_sets.yaml")
    parser.add_argument("--candidates-config", default="analysis/configs/phase1_candidates.yaml")
    args = parser.parse_args()
    cfg, cand_cfg = load_config(args.config), load_config(args.candidates_config)
    goal = cfg["goals"]["silent_errors"]["candidate_goal"]
    row = next(r for r in json.loads(Path(cfg["candidates_summary"]).read_text())["substitutive"] if r["goal"] == goal)
    spec_words = cand_cfg["substitutions"][goal]["spec_words"]

    statements: dict[str, str] = {}
    for f in sorted(Path(cand_cfg["attack_root"]).expanduser().glob("bigcodebench.*.no_side_task.jsonl")):
        for sample in read_jsonl(f):
            statements.setdefault(sample["metadata"]["main_task_id"], sample["metadata"]["original_prompt"])

    named = set(row["problem_ids"]) - set(row["unspecified_ids"]) - set(row["benign_hit_ids"])
    canonical_dir = Path(cfg["canonical_dir"])
    to_review = sorted((pid for pid in named if has_separable_handler((canonical_dir / f"{pid}.py").read_text())), key=int)
    print(f"# Silent-errors hand-check: {len(to_review)} problems\n")
    for pid in to_review:
        print(f"## {pid}\n**Statement:**")
        for sentence in matching_sentences(statements[pid], spec_words):
            print(f"- {sentence}")
        print("**Canonical handlers:**")
        for summary in handler_summaries((canonical_dir / f"{pid}.py").read_text()):
            print(f"- `{summary}`")
        print()


if __name__ == "__main__":
    main()
