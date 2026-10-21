"""Phase 1 pilot: which Bandit rules could become an additive or substitutive hidden objective?

Bandit (the Python security linter) is the detector, so the weakened pattern is defined by a tested tool, not a hand regex.
It runs over two corpora, both copied out of the downloaded data into out_dir:
  canonical  the reference solution of every held-out problem
  benign     every distinct unprompted (no_side_task) generation on those problems
Two tables come out:
  substitutive  per goal in the config: held-out problems whose canonical solution already uses the API and does not
                already trip the rule, how many of their statements name it, and the benign base rate on them
  additive      per Bandit rule: the benign base rate over all held-out problems (an addition fits any problem)

Run from the repo root:
    python -m analysis.phase1_candidates
"""

import argparse
import dataclasses
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config, read_jsonl


@dataclasses.dataclass(frozen=True)
class Problem:
    problem_id: str
    code: str
    statement: str


@dataclasses.dataclass(frozen=True)
class Generation:
    gen_id: str
    problem_id: str
    code: str


def canonical_code(metadata: dict[str, Any]) -> str:
    """Some attack files store the full solution in code_prompt already; appending the body again would not parse."""
    prompt, body = metadata["code_prompt"], metadata["canonical_main_sol"]
    return prompt if prompt.endswith(body) else prompt + body


def load_problems(attack_root: Path, held_out: set[str]) -> dict[str, Problem]:
    problems: dict[str, Problem] = {}
    for path in sorted(attack_root.glob("bigcodebench.*.no_side_task.jsonl")):
        for sample in read_jsonl(path):
            md = sample["metadata"]
            if md["main_task_id"] in held_out:
                problems.setdefault(md["main_task_id"], Problem(md["main_task_id"], canonical_code(md), md["original_prompt"]))
    return problems


def dedupe_generations(samples: list[dict[str, Any]], held_out: set[str]) -> list[Generation]:
    """Distinct (problem, code) pairs; rollout indices collide across the --5/--20 folders, so dedupe on the code itself."""
    seen: set[tuple[str, str]] = set()
    gens: list[Generation] = []
    for sample in samples:
        md = sample["metadata"]
        key = (md["main_task_id"], md["attack_code"] or "")
        if md["main_task_id"] in held_out and key not in seen:
            seen.add(key)
            gens.append(Generation(f"{key[0]}__{len(gens)}", key[0], key[1]))
    return gens


def load_benign(monitor_inputs_root: Path, generator: str, held_out: set[str]) -> list[Generation]:
    samples: list[dict[str, Any]] = []
    for folder in sorted(monitor_inputs_root.glob(f"bigcodebench.*.no_side_task/{generator}--*")):
        unfiltered = folder / "all.main_aware.action_only.jsonl"
        samples += read_jsonl(unfiltered if unfiltered.exists() else folder / "main-all.main_aware.action_only.jsonl")
    return dedupe_generations(samples, held_out)


def write_corpus(directory: Path, files: dict[str, str]) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    for name, code in files.items():
        (directory / f"{name}.py").write_text(code)


def findings_by_file(report: dict[str, Any]) -> dict[str, set[str]]:
    """Bandit JSON report -> file stem -> the test ids that fired in it."""
    fired: dict[str, set[str]] = defaultdict(set)
    for result in report["results"]:
        fired[Path(result["filename"]).stem].add(result["test_id"])
    return fired


def run_bandit(bandit_cmd: list[str], directory: Path) -> tuple[dict[str, set[str]], int]:
    """Returns the findings and how many files Bandit could not parse (those count as clean, so report them)."""
    proc = subprocess.run([*bandit_cmd, "-r", str(directory), "-f", "json", "-q"], capture_output=True, text=True, check=False)
    if not proc.stdout.strip():
        raise RuntimeError(f"bandit produced no report (exit {proc.returncode}): {proc.stderr[:500]}")
    report = json.loads(proc.stdout)
    return findings_by_file(report), len(report["errors"])


def eligible_problems(problems: dict[str, Problem], precondition: str, bandit_ids: list[str], canonical_fired: dict[str, set[str]]) -> set[str]:
    """Problems that already use the API safely, so the weakened version is a substitution."""
    return {
        pid for pid, problem in problems.items()
        if re.search(precondition, problem.code) and not canonical_fired.get(pid, set()) & set(bandit_ids)
    }


def base_rate(gens: list[Generation], bandit_ids: list[str], benign_fired: dict[str, set[str]]) -> tuple[int, int]:
    hits = sum(bool(benign_fired.get(g.gen_id, set()) & set(bandit_ids)) for g in gens)
    return hits, len(gens)


def substitutive_rows(
    cfg: dict[str, Any], problems: dict[str, Problem], gens: list[Generation],
    canonical_fired: dict[str, set[str]], benign_fired: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows = []
    for goal, spec in cfg["substitutions"].items():
        ids = eligible_problems(problems, spec["precondition"], spec["bandit_ids"], canonical_fired)
        named = {pid for pid in ids if re.search(spec["spec_words"], problems[pid].statement, re.IGNORECASE)}
        pool_gens = [g for g in gens if g.problem_id in ids]
        hits, n = base_rate(pool_gens, spec["bandit_ids"], benign_fired)
        hit_ids = {g.problem_id for g in pool_gens if benign_fired.get(g.gen_id, set()) & set(spec["bandit_ids"])}
        rate = hits / n if n else None
        viable = len(ids) >= cfg["thresholds"]["min_problems"] and rate is not None and rate <= cfg["thresholds"]["max_base_rate"]
        rows.append(dict(goal=goal, bandit_ids=spec["bandit_ids"], problems=len(ids), spec_names_it=len(named),
                         unspecified=len(ids) - len(named), benign_gens=n, base_rate=rate, viable=viable,
                         problem_ids=sorted(ids, key=int), unspecified_ids=sorted(ids - named, key=int),
                         benign_hit_ids=sorted(hit_ids, key=int)))
    return rows


def additive_rows(cfg: dict[str, Any], gens: list[Generation], canonical_fired: dict[str, set[str]], benign_fired: dict[str, set[str]]) -> list[dict[str, Any]]:
    rule_ids = sorted({tid for fired in [*canonical_fired.values(), *benign_fired.values()] for tid in fired})
    rows = []
    for tid in rule_ids:
        hits, n = base_rate(gens, [tid], benign_fired)
        rows.append(dict(bandit_id=tid, benign_hits=hits, benign_gens=n, base_rate=hits / n,
                         canonical_problems_firing=sum(tid in fired for fired in canonical_fired.values()),
                         low_base_rate=hits / n <= cfg["thresholds"]["max_base_rate"]))
    return rows


def print_tables(substitutive: list[dict[str, Any]], additive: list[dict[str, Any]]) -> None:
    print(f"\n## Substitutive (problems that already use the API safely)\n{'goal':26} {'ids':18} {'problems':>8} {'unspecified':>11} {'benign':>6} {'base rate':>9}  viable")
    for r in sorted(substitutive, key=lambda r: -r["problems"]):
        rate = "-" if r["base_rate"] is None else f"{r['base_rate']:.1%}"
        print(f"{r['goal']:26} {','.join(r['bandit_ids']):18} {r['problems']:8} {r['unspecified']:11} {r['benign_gens']:6} {rate:>9}  {'yes' if r['viable'] else ''}")
    print(f"\n## Additive (benign base rate over all held-out problems)\n{'rule':6} {'benign hits':>11} {'base rate':>9} {'canonical problems firing':>26}")
    for r in sorted(additive, key=lambda r: r["base_rate"]):
        print(f"{r['bandit_id']:6} {r['benign_hits']:11} {r['base_rate']:9.2%} {r['canonical_problems_firing']:26}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_candidates.yaml")
    parser.add_argument("--out-dir", help="overrides out_dir from the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg["out_dir"])

    held_out = set(json.loads(Path(cfg["task_id_sets"]).read_text())[cfg["held_out_set"]])
    problems = load_problems(Path(cfg["attack_root"]).expanduser(), held_out)
    gens = load_benign(Path(cfg["monitor_inputs_root"]).expanduser(), cfg["generator"], held_out)
    print(f"{len(problems)} held-out problems, {len(gens)} distinct benign generations")

    write_corpus(out_dir / "code/canonical", {pid: p.code for pid, p in problems.items()})
    write_corpus(out_dir / "code/benign", {g.gen_id: g.code for g in gens})
    canonical_fired, canonical_errors = run_bandit(cfg["bandit_cmd"], out_dir / "code/canonical")
    benign_fired, benign_errors = run_bandit(cfg["bandit_cmd"], out_dir / "code/benign")
    print(f"bandit could not parse {canonical_errors} canonical and {benign_errors} benign files (counted as clean)")

    substitutive = substitutive_rows(cfg, problems, gens, canonical_fired, benign_fired)
    additive = additive_rows(cfg, gens, canonical_fired, benign_fired)
    print_tables(substitutive, additive)
    (out_dir / "summary.json").write_text(json.dumps(dict(substitutive=substitutive, additive=additive), indent=2))


if __name__ == "__main__":
    main()
