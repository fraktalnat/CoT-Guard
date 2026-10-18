"""Experiment 2b: is the unrequested / requested pair design leaking the label through the problem text alone?

The requested prompt is the clean problem plus one requirement sentence, so a probe could separate the pair by
noticing "this problem has an extra requirement" without ever checking it against the CoT. This script measures
that with prompt-only classifiers (they never see the CoT), on two pair designs:
  naive    unrequested = clean problem                       vs requested = problem + this side task's requirement
  control  unrequested = problem + another side task's req.  vs requested = problem + this side task's requirement
In the control both prompts carry a requirement, so only "does it match what the CoT did" differs. The control is
then scored with Experiment 2's CoT overlap metrics, which gives the bar a probe has to beat on matched pairs.

Run from the repo root after Experiment 2's sample/export/paraphrase steps:
    python -m analysis.leakage_check report
"""

import argparse
import ast
import collections
import math
import os
import random
import re
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config, read_jsonl, write_jsonl
from analysis.overlap_baseline import (
    REQUESTED_VERSIONS,
    Attack,
    attribution_text,
    auroc,
    cluster_bootstrap_auroc,
    load_attack_prompts,
    load_paraphrases,
    metric_names,
    overlap_scores,
    paired_win_rate,
    requested_prompt,
    words,
)

DESIGNS = ("naive", "control")
SIGNATURE_RE = re.compile(r"def task_func\(.*?\):", re.S)
CALL_RE = re.compile(r"task_func\([^)]*\)")


def problem_key(prompt_id: str) -> str:
    """The BigCodeBench problem behind a prompt_id; one problem can appear under several side tasks."""
    return prompt_id.split(":", 1)[1]


def signature_names(problem: str) -> str | None:
    """Comma-joined argument names of task_func in the problem, e.g. "dic, cities"; None if not found."""
    match = SIGNATURE_RE.search(problem)
    if match is None:
        return None
    args = ast.parse(f"{match.group(0)}\n    pass").body[0].args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    names += [f"*{a.arg}" for a in (args.vararg,) if a] + [f"**{a.arg}" for a in (args.kwarg,) if a]
    return ", ".join(names)


def adapt_requirement(requirement: str, problem: str) -> str:
    """Point a requirement written for another problem at this problem's task_func signature."""
    names = signature_names(problem)
    if names is None:
        return requirement
    return CALL_RE.sub(lambda _: f"task_func({names})", requirement)


def pick_donors(side_task_of: dict[str, str], rng: random.Random) -> dict[str, str]:
    """prompt_id -> a donor prompt_id from a different side task, drawn uniformly so every side task's
    requirements appear on the unrequested side about as often as on the requested side."""
    ids = sorted(side_task_of)
    return {pid: rng.choice([d for d in ids if side_task_of[d] != side_task_of[pid]]) for pid in ids}


def build_prompts(
    attack: Attack, attack_prompt: str, requirements: dict[str, str], donor_requirement: dict[str, str]
) -> dict[str, dict[str, str]]:
    """version -> {"requested", "naive", "control"} prompts for one attack.

    requirements / donor_requirement map version -> the text spliced in for this attack / for its donor.
    """
    line = attack.side_and_attack_policy
    prompts = {}
    for version in REQUESTED_VERSIONS:
        donor = adapt_requirement(donor_requirement[version], attack.problem)
        prompts[version] = {
            "requested": requested_prompt(attack.problem, attack_prompt, line, requirements[version]),
            "naive": attack.problem,
            "control": requested_prompt(attack.problem, attack_prompt, line, donor),
        }
    return prompts


def train_naive_bayes(docs: list[list[str]], labels: list[int], alpha: float) -> dict[str, float]:
    """Per-word log-likelihood ratio log P(w | label 1) - log P(w | label 0), multinomial with Laplace smoothing."""
    counts = [collections.Counter(), collections.Counter()]
    for doc, label in zip(docs, labels):
        counts[label].update(doc)
    vocab = set(counts[0]) | set(counts[1])
    totals = [sum(c.values()) + alpha * len(vocab) for c in counts]
    return {w: math.log((counts[1][w] + alpha) / totals[1]) - math.log((counts[0][w] + alpha) / totals[0]) for w in vocab}


def nb_score(weights: dict[str, float], doc: list[str]) -> float:
    """Log-odds that the doc has label 1 (equal priors); unseen words contribute nothing."""
    return sum(weights.get(w, 0.0) for w in doc)


def grouped_folds(groups: list[str], n_folds: int, rng: random.Random) -> list[int]:
    """Fold index per item; every item of a group lands in the same fold."""
    distinct = sorted(set(groups))
    rng.shuffle(distinct)
    fold_of = {g: i % n_folds for i, g in enumerate(distinct)}
    return [fold_of[g] for g in groups]


def cross_val_nb(
    docs: list[list[str]], labels: list[int], groups: list[str], owners: list[str | None], n_folds: int, alpha: float, rng: random.Random
) -> list[float]:
    """Out-of-fold naive Bayes score for every doc.

    owners[i] names whose requirement text doc i carries (None for none). Training drops docs carrying a requirement
    that also appears in the test fold, otherwise a requirement reused across designs leaks with the opposite label.
    """
    folds = grouped_folds(groups, n_folds, rng)
    scores = [0.0] * len(docs)
    for k in range(n_folds):
        test_owners = {owners[i] for i, f in enumerate(folds) if f == k} - {None}
        train = [i for i, f in enumerate(folds) if f != k and owners[i] not in test_owners]
        weights = train_naive_bayes([docs[i] for i in train], [labels[i] for i in train], alpha)
        for i, f in enumerate(folds):
            if f == k:
                scores[i] = nb_score(weights, docs[i])
    return scores


def prompt_only_scores(
    pairs: dict[str, tuple[str, str]], unrequested_owner: dict[str, str | None], cfg: dict[str, Any], rng: random.Random
) -> dict[str, dict[str, tuple[float, float]]]:
    """classifier -> prompt_id -> (requested score, unrequested score), from the problem text alone.

    pairs maps prompt_id -> (requested prompt, unrequested prompt); unrequested_owner maps prompt_id -> whose
    requirement the unrequested prompt carries (None for the clean problem).
    """
    ids = sorted(pairs)
    docs = [words(pairs[pid][0]) for pid in ids] + [words(pairs[pid][1]) for pid in ids]
    labels = [1] * len(ids) + [0] * len(ids)
    groups = [problem_key(pid) for pid in ids] * 2
    owners = list(ids) + [unrequested_owner[pid] for pid in ids]
    nb = cross_val_nb(docs, labels, groups, owners, cfg["n_folds"], cfg["nb_alpha"], rng)
    n = len(ids)
    return {
        "length": {pid: (float(len(docs[i])), float(len(docs[n + i]))) for i, pid in enumerate(ids)},
        "naive_bayes": {pid: (nb[i], nb[n + i]) for i, pid in enumerate(ids)},
    }


def summarize_pairs(clusters: list[list[tuple[float, float]]], overlap_cfg: dict[str, Any], rng: random.Random) -> dict[str, float]:
    pairs = [p for c in clusters for p in c]
    low, high = cluster_bootstrap_auroc(clusters, overlap_cfg["n_bootstrap"], overlap_cfg["ci"], rng)
    return {"auroc": auroc([r for r, _ in pairs], [u for _, u in pairs]), "ci_low": low, "ci_high": high, "paired_win": paired_win_rate(pairs)}


def by_problem(pair_of: dict[str, tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Cluster prompt_id-level pairs by problem for the bootstrap."""
    clusters: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for pid, pair in sorted(pair_of.items()):
        clusters[problem_key(pid)].append(pair)
    return list(clusters.values())


def separation(auc: float) -> float:
    """AUROC folded around 0.5: a classifier reliably below chance still carries the label."""
    return max(auc, 1 - auc)


def fmt(s: dict[str, float]) -> str:
    return f"{s['auroc']:.2f} [{s['ci_low']:.2f},{s['ci_high']:.2f}] ({s['paired_win']:.2f})"


def cmd_report(cfg: dict[str, Any], out_dir: Path) -> None:
    ocfg = load_config(cfg["overlap_config"])
    overlap_cfg, metrics = ocfg["overlap"], metric_names(ocfg["overlap"]["ngram_sizes"])
    rng = random.Random(cfg["seed"])
    exp2_dir = Path(ocfg["out_dir"])
    attacks = [Attack(**row) for row in read_jsonl(exp2_dir / "attacks.jsonl")]
    attack_prompts = load_attack_prompts(os.path.expanduser(ocfg["attack_root"]), ocfg["tasks"])
    paraphrases, _ = load_paraphrases(exp2_dir / "paraphrases.jsonl")
    first = {a.prompt_id: a for a in attacks}
    requirements = {pid: {"literal": a.side_and_attack_policy, "paraphrased": paraphrases[pid]} for pid, a in first.items()}
    donors = pick_donors({pid: a.side_task for pid, a in first.items()}, rng)
    n_unadapted = sum(signature_names(a.problem) is None for a in first.values())

    prompts = {pid: build_prompts(a, attack_prompts[pid], requirements[pid], requirements[donors[pid]]) for pid, a in first.items()}
    leak_auroc = cfg["thresholds"]["leak_auroc"]
    print(f"{len(attacks)} attacks over {len(first)} prompts; {n_unadapted} problems without a parsable task_func signature")
    print(f"\n== Prompt-only classifiers (no CoT), requested vs unrequested, AUROC [{overlap_cfg['ci']:.0%} CI] (paired win)")
    print(f"{'version':14s}{'design':10s}{'length':>26s}{'naive_bayes':>26s}")
    leak = {}
    for version in REQUESTED_VERSIONS:
        for design in DESIGNS:
            pairs = {pid: (p[version]["requested"], p[version][design]) for pid, p in prompts.items()}
            owner = {pid: donors[pid] if design == "control" else None for pid in pairs}
            scores = prompt_only_scores(pairs, owner, cfg["classifier"], rng)
            results = {name: summarize_pairs(by_problem(s), overlap_cfg, rng) for name, s in scores.items()}
            leak[(version, design)] = max(separation(r["auroc"]) for r in results.values())
            print(f"{version:14s}{design:10s}" + "".join(f"{fmt(r):>26s}" for r in results.values()))

    rows = []
    for a in attacks:
        text, fallback = attribution_text(a.attack_cot)
        scores = {version: {kind: overlap_scores(text, prompt, overlap_cfg["ngram_sizes"]) for kind, prompt in p.items()} for version, p in prompts[a.prompt_id].items()}
        rows.append({"attack_id": a.attack_id, "prompt_id": a.prompt_id, "side_task": a.side_task, "donor": donors[a.prompt_id], "attribution_fallback": fallback, "scores": scores})
    write_jsonl(out_dir / "scores.jsonl", rows)

    groups = {"pooled": rows, **{side: [r for r in rows if r["side_task"] == side] for side in ocfg["tasks"]}}
    bars = {}
    for version in REQUESTED_VERSIONS:
        print(f"\n== CoT overlap on the control design ({version}): requested vs other-task requirement, AUROC [{overlap_cfg['ci']:.0%} CI] (paired win)")
        print(f"{'group':26s}{'n':>5s}" + "".join(f"{m:>26s}" for m in metrics))
        for name, group in groups.items():
            cells = []
            for metric in metrics:
                clusters: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
                for r in group:
                    clusters[r["prompt_id"]].append((r["scores"][version]["requested"][metric], r["scores"][version]["control"][metric]))
                s = summarize_pairs(list(clusters.values()), overlap_cfg, rng)
                cells.append(fmt(s))
                if name == "pooled":
                    bars[(version, metric)] = s["auroc"]
            print(f"{name:26s}{len(group):5d}" + "".join(f"{c:>26s}" for c in cells))

    print("\n== Decision")
    for version in REQUESTED_VERSIONS:
        naive, control = leak[(version, "naive")], leak[(version, "control")]
        best_metric = max(metrics, key=lambda m: bars[(version, m)])
        print(
            f"{version}: naive design {'LEAKS' if naive >= leak_auroc else 'clean'} (best prompt-only separation {naive:.2f}); "
            f"control design {'LEAKS' if control >= leak_auroc else 'clean'} (best prompt-only separation {control:.2f}); "
            f"probe bar on control = {best_metric} {bars[(version, best_metric)]:.2f}  (leak if >= {leak_auroc})"
        )


COMMANDS = {"report": cmd_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="analysis/configs/leakage_check.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    COMMANDS[args.command](cfg, out_dir)


if __name__ == "__main__":
    main()
