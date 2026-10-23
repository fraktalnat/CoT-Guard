"""Phase 1: build the pre-registered problem sets for the TLS and silent-errors matched pairs.

Starts from the candidate scan (analysis/phase1_candidates.py) and applies the rules in docs/phase1-preregistration.md:
  every goal     statement does not name the API (or kept by the step-2 hand-check; dropped if the hand-check says so),
                 and no unprompted benign generation on the problem already trips the Bandit rule
  tls            the canonical solution makes exactly one requests call Bandit can see
  silent_errors  the canonical solution has an except handler, and its whole body is not one broad try
                 (silencing that handler would look exactly like the additive wrapper)
The pilot subset is a seeded sample of one goal's set. Exclusion rules added after the pilot (tls: http_only, where
verify=False is a no-op) are applied after it is drawn, so the pilot set does not change.

Run from the repo root, after analysis.phase1_candidates:
    python -m analysis.phase1_problem_sets
"""

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any, Callable

from analysis.miss_triage import load_config
from src.cot_red_handed.tasks.static_verifiers import body_try, find_task_func, is_broad, requests_calls


def single_requests_call(code: str) -> bool:
    return len(requests_calls(ast.parse(code))) == 1


def has_separable_handler(code: str) -> bool:
    tree = ast.parse(code)
    if not any(isinstance(node, ast.ExceptHandler) for node in ast.walk(tree)):
        return False
    func = find_task_func(tree)
    try_ = body_try(func) if func else None
    return not (try_ and any(is_broad(h) for h in try_.handlers))


def url_literals(code: str) -> list[str]:
    """String constants that start with a URL scheme; an f-string's leading text is its own Constant, so it counts too."""
    return [
        node.value for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(("http://", "https://"))
    ]


def http_only(code: str) -> bool:
    urls = url_literals(code)
    return bool(urls) and all(url.startswith("http://") for url in urls)


STRUCTURE_RULES: dict[str, Callable[[str], bool]] = {
    "tls": single_requests_call,
    "silent_errors": has_separable_handler,
}

EXCLUSION_RULES: dict[str, Callable[[str], bool]] = {
    "http_only": http_only,
}


def problem_set(
    row: dict[str, Any], handcheck_keep: list[str], handcheck_drop: list[str],
    canonical: dict[str, str], structure_ok: Callable[[str], bool],
) -> list[str]:
    candidates = (set(row["unspecified_ids"]) | set(handcheck_keep)) - set(handcheck_drop) - set(row["benign_hit_ids"])
    return sorted((pid for pid in candidates if structure_ok(canonical[pid])), key=int)


def pilot_sample(ids: list[str], size: int, seed: int) -> list[str]:
    return sorted(random.Random(seed).sample(ids, size), key=int)


def without(ids: list[str], canonical: dict[str, str], rules: list[Callable[[str], bool]]) -> list[str]:
    return [pid for pid in ids if not any(rule(canonical[pid]) for rule in rules)]


def build_sets(
    goals: dict[str, Any], rows: dict[str, dict[str, Any]], canonical: dict[str, str], pilot: dict[str, Any],
) -> dict[str, list[str]]:
    """The pilot is drawn before the exclusion rules, so adding a rule after the pilot ran leaves the pilot set as it was."""
    sets = {
        goal: problem_set(rows[spec["candidate_goal"]], spec["handcheck_keep"], spec["handcheck_drop"], canonical, STRUCTURE_RULES[goal])
        for goal, spec in goals.items()
    }
    sets[f"{pilot['goal']}_pilot"] = pilot_sample(sets[pilot["goal"]], pilot["size"], pilot["seed"])
    for goal, spec in goals.items():
        sets[goal] = without(sets[goal], canonical, [EXCLUSION_RULES[name] for name in spec["exclude"]])
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_problem_sets.yaml")
    cfg = load_config(parser.parse_args().config)

    rows = {r["goal"]: r for r in json.loads(Path(cfg["candidates_summary"]).read_text())["substitutive"]}
    canonical = {p.stem: p.read_text() for p in Path(cfg["canonical_dir"]).glob("*.py")}
    sets = build_sets(cfg["goals"], rows, canonical, cfg["pilot"])

    for name, ids in sets.items():
        print(f"{name:16} {len(ids):3}  {', '.join(ids)}")
    Path(cfg["out_path"]).write_text(json.dumps(sets, indent=2))


if __name__ == "__main__":
    main()
