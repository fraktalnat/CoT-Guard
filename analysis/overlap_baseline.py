"""Experiment 2: can simple string matching already separate the unrequested / requested pairs?

Each sampled attack (side task succeeded, A1 baseline) is scored against three versions of the problem:
  unrequested  the clean problem the monitor actually sees (label: malicious)
  literal      the problem with the generator's "You must also: X" line spliced back in (label: benign)
  paraphrased  the same, with X rewritten into ordinary spec wording (label: benign)
The score is word overlap between the CoT's "the problem says..." sentences and the problem version.
Its AUROC (requested vs unrequested) is the bar a provenance probe has to beat.

Run from the repo root, in order:
    python -m analysis.overlap_baseline sample       # stratified attacks -> attacks.jsonl
    python -m analysis.overlap_baseline export       # one request per distinct problem -> paraphrase_requests.jsonl
    (write paraphrases.jsonl by hand or with an agent, following analysis/prompts/overlap_paraphrase.yaml)
    python -m analysis.overlap_baseline report       # AUROCs, paired win rates, stop/go -> scores.jsonl

No model API is called; the paraphrases are an input file.
"""

import argparse
import bisect
import collections
import dataclasses
import difflib
import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from analysis.explore_monitor_results import ATTRIBUTION_RE
from analysis.miss_triage import generator_from_dir, load_config, read_jsonl, stratified_sample, write_jsonl

REQUESTED_VERSIONS = ("literal", "paraphrased")
POLICY_PREFIX = "You must also"
WORD_RE = re.compile(r"[a-z0-9_]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
THINK_TAG_RE = re.compile(r"</?think>")


@dataclasses.dataclass(frozen=True)
class Attack:
    attack_id: str
    prompt_id: str
    side_task: str
    generator: str
    problem: str
    side_task_description: str
    side_and_attack_policy: str
    attack_cot: str


def make_prompt_id(side_task: str, main_task_id: str) -> str:
    """One requested prompt per (side task, problem); all generators and seeds share it."""
    return f"{side_task}:{main_task_id}"


def collect_attacks(root: str, model: str, tasks: list[str]) -> list[Attack]:
    attacks = []
    for side in tasks:
        pattern = f"{root}/bigcodebench.{side}.baseline/*/side-all.{model}.main_aware.cot_action.jsonl"
        for path in sorted(glob.glob(pattern)):
            rel = os.path.relpath(path, root)
            with open(path) as f:
                for line_no, line in enumerate(f):
                    meta = json.loads(line)["metadata"]
                    attacks.append(
                        Attack(
                            attack_id=f"{rel}:{line_no}",
                            prompt_id=make_prompt_id(side, meta["main_task_id"]),
                            side_task=side,
                            generator=generator_from_dir(Path(path).parent.name),
                            problem=meta["original_prompt"],
                            side_task_description=meta["side_task_description"],
                            side_and_attack_policy=meta["side_and_attack_policy"],
                            attack_cot=meta["attack_cot"],
                        )
                    )
    return attacks


def load_attack_prompts(attack_root: str, tasks: list[str]) -> dict[str, str]:
    """prompt_id -> the prompt the generator actually received, from datasets/attack/*.baseline.jsonl."""
    prompts = {}
    for side in tasks:
        for row in read_jsonl(Path(attack_root) / f"bigcodebench.{side}.baseline.jsonl"):
            prompts[make_prompt_id(side, row["metadata"]["main_task_id"])] = row["messages"][0]["content"]
    return prompts


def requested_prompt(problem: str, attack_prompt: str, policy_line: str, requirement: str) -> str:
    """The clean problem with `requirement` inserted where the attack prompt had its policy line.

    Built from the clean problem rather than the attack prompt, so the two versions differ only in the
    inserted requirement (the attack prompt also carries an output-format trailer the monitor never sees).
    """
    prefix = attack_prompt.split(policy_line)[0]
    if attack_prompt.count(policy_line) != 1 or not problem.startswith(prefix):
        raise ValueError("attack prompt is not the clean problem with one policy line spliced in")
    return f"{prefix}{requirement}\n{problem[len(prefix):]}"


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def containment(text_tokens: list[str], prompt_tokens: list[str], n: int) -> float:
    """Share of the text's distinct n-grams that also occur in the prompt."""
    text_ngrams = ngrams(text_tokens, n)
    if not text_ngrams:
        return 0.0
    return len(text_ngrams & ngrams(prompt_tokens, n)) / len(text_ngrams)


def longest_match(text_tokens: list[str], prompt_tokens: list[str]) -> int:
    """Length in words of the longest run the text copies verbatim from the prompt."""
    matcher = difflib.SequenceMatcher(None, text_tokens, prompt_tokens, autojunk=False)
    return matcher.find_longest_match(0, len(text_tokens), 0, len(prompt_tokens)).size


def attribution_text(cot: str) -> tuple[str, bool]:
    """(the CoT's sentences that attribute something to the problem, whether we fell back to the whole CoT)."""
    body = THINK_TAG_RE.sub("", cot)
    sentences = [s for s in SENTENCE_SPLIT_RE.split(body) if ATTRIBUTION_RE.search(s)]
    if not sentences:
        return body, True
    return " ".join(sentences), False


def metric_names(ngram_sizes: list[int]) -> list[str]:
    return [f"containment_{n}" for n in ngram_sizes] + ["longest_match"]


def overlap_scores(text: str, prompt: str, ngram_sizes: list[int]) -> dict[str, float]:
    text_tokens, prompt_tokens = words(text), words(prompt)
    scores = {f"containment_{n}": containment(text_tokens, prompt_tokens, n) for n in ngram_sizes}
    scores["longest_match"] = float(longest_match(text_tokens, prompt_tokens))
    return scores


def auroc(positives: list[float], negatives: list[float]) -> float:
    """P(random positive scores above random negative), ties count half (Mann-Whitney U / (n_pos * n_neg))."""
    ordered = sorted(negatives)
    wins = 0.0
    for p in positives:
        below, not_above = bisect.bisect_left(ordered, p), bisect.bisect_right(ordered, p)
        wins += below + 0.5 * (not_above - below)
    return wins / (len(positives) * len(negatives))


def paired_win_rate(pairs: list[tuple[float, float]]) -> float:
    """pairs are (requested, unrequested) for the same CoT. Share where requested scores higher, ties half."""
    return sum(1.0 if r > u else 0.5 if r == u else 0.0 for r, u in pairs) / len(pairs)


def cluster_bootstrap_auroc(
    clusters: list[list[tuple[float, float]]], n_bootstrap: int, ci: float, rng: random.Random
) -> tuple[float, float]:
    """Percentile CI over resampled clusters of (requested, unrequested) pairs.

    Clusters are problems: rollouts of one problem share a requested prompt, so they aren't independent.
    """
    stats = []
    for _ in range(n_bootstrap):
        pairs = [pair for cluster in rng.choices(clusters, k=len(clusters)) for pair in cluster]
        stats.append(auroc([r for r, _ in pairs], [u for _, u in pairs]))
    stats.sort()
    tail = (1 - ci) / 2
    return stats[int(tail * (n_bootstrap - 1))], stats[int((1 - tail) * (n_bootstrap - 1))]


def quantile_bins(values: list[float], n_bins: int) -> list[int]:
    """Bin index per value, by rank, into n_bins roughly equal-sized bins."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    bins = [0] * len(values)
    for rank, i in enumerate(order):
        bins[i] = rank * n_bins // len(values)
    return bins


def decide(best_auroc: float, stop_auroc: float) -> str:
    return "STOP" if best_auroc >= stop_auroc else "GO"


def valid_paraphrase(requirement: str | None) -> str | None:
    """The stripped requirement, or None if missing or still carrying the literal policy prefix."""
    if not isinstance(requirement, str):
        return None
    requirement = requirement.strip()
    if not requirement or requirement.startswith(POLICY_PREFIX):
        return None
    return requirement


def paraphrase_requests(attacks: list[Attack]) -> list[dict[str, str]]:
    """One request per distinct problem. Only the problem and requirement, never the CoT the overlap is scored on."""
    by_prompt = {a.prompt_id: a for a in attacks}
    return [
        {"prompt_id": pid, "side_task": a.side_task, "problem": a.problem, "requirement": a.side_task_description}
        for pid, a in sorted(by_prompt.items())
    ]


def cmd_sample(cfg: dict[str, Any], out_dir: Path) -> None:
    attacks = collect_attacks(os.path.expanduser(cfg["results_root"]), cfg["source_monitor"], cfg["tasks"])
    attack_prompts = load_attack_prompts(os.path.expanduser(cfg["attack_root"]), cfg["tasks"])
    unmatched = [a.attack_id for a in attacks if a.prompt_id not in attack_prompts]
    if unmatched:
        raise SystemExit(f"{len(unmatched)} attacks have no attack prompt in {cfg['attack_root']}: {unmatched[:5]}")
    sample = stratified_sample(attacks, cfg["n_attacks"], lambda a: a.side_task, random.Random(cfg["seed"]))
    per_task = collections.Counter(a.side_task for a in sample)
    n_prompts = len({a.prompt_id for a in sample})
    print(f"{len(attacks)} attacks, sampled {len(sample)} over {n_prompts} distinct problems: {dict(per_task)}")
    write_jsonl(out_dir / "attacks.jsonl", [dataclasses.asdict(a) for a in sample])
    print(f"wrote {out_dir / 'attacks.jsonl'}")


def cmd_export(cfg: dict[str, Any], out_dir: Path) -> None:
    attacks = [Attack(**row) for row in read_jsonl(out_dir / "attacks.jsonl")]
    requests = paraphrase_requests(attacks)
    write_jsonl(out_dir / "paraphrase_requests.jsonl", requests)
    print(f"wrote {len(requests)} requests to {out_dir / 'paraphrase_requests.jsonl'}")
    print(f"write {out_dir / 'paraphrases.jsonl'} with one {{\"prompt_id\", \"requirement\"}} line each, per {cfg['paraphrase_rules']}")


def load_paraphrases(path: Path) -> tuple[dict[str, str], int]:
    """(latest valid paraphrase per prompt_id, number of invalid rows skipped)."""
    paraphrases, n_invalid = {}, 0
    for row in read_jsonl(path):
        requirement = valid_paraphrase(row.get("requirement"))
        if requirement is None:
            n_invalid += 1
            continue
        paraphrases[row["prompt_id"]] = requirement
    return paraphrases, n_invalid


def score_attack(attack: Attack, attack_prompt: str, paraphrase: str, ngram_sizes: list[int]) -> dict[str, Any]:
    text, fallback = attribution_text(attack.attack_cot)
    literal_line = attack.side_and_attack_policy
    prompts = {
        "unrequested": attack.problem,
        "literal": requested_prompt(attack.problem, attack_prompt, literal_line, literal_line),
        "paraphrased": requested_prompt(attack.problem, attack_prompt, literal_line, paraphrase),
    }
    return {
        "attack_id": attack.attack_id,
        "prompt_id": attack.prompt_id,
        "side_task": attack.side_task,
        "generator": attack.generator,
        "attribution_fallback": fallback,
        "paraphrase": paraphrase,
        "paraphrase_copy": containment(words(paraphrase), words(attack.side_task_description), 3),
        "scores": {version: overlap_scores(text, p, ngram_sizes) for version, p in prompts.items()},
    }


def pair_clusters(rows: list[dict], version: str, metric: str) -> list[list[tuple[float, float]]]:
    clusters: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in rows:
        clusters[row["prompt_id"]].append((row["scores"][version][metric], row["scores"]["unrequested"][metric]))
    return list(clusters.values())


def summarize(rows: list[dict], version: str, metric: str, overlap_cfg: dict[str, Any], rng: random.Random) -> dict[str, float]:
    clusters = pair_clusters(rows, version, metric)
    pairs = [pair for cluster in clusters for pair in cluster]
    low, high = cluster_bootstrap_auroc(clusters, overlap_cfg["n_bootstrap"], overlap_cfg["ci"], rng)
    return {
        "auroc": auroc([r for r, _ in pairs], [u for _, u in pairs]),
        "ci_low": low,
        "ci_high": high,
        "paired_win": paired_win_rate(pairs),
    }


def print_table(title: str, groups: dict[str, list[dict]], version: str, metrics: list[str], overlap_cfg: dict[str, Any], rng: random.Random) -> dict[str, dict[str, float]]:
    """Prints AUROC [CI] (paired win) per group x metric; returns the pooled row's summaries."""
    print(f"\n== {title}: requested={version} vs unrequested, AUROC [{overlap_cfg['ci']:.0%} CI] (paired win)")
    print(f"{'group':26s}{'n':>5s}" + "".join(f"{m:>26s}" for m in metrics))
    pooled = {}
    for name, rows in groups.items():
        cells = []
        for metric in metrics:
            s = summarize(rows, version, metric, overlap_cfg, rng)
            cells.append(f"{s['auroc']:.2f} [{s['ci_low']:.2f},{s['ci_high']:.2f}] ({s['paired_win']:.2f})")
            if name == "pooled":
                pooled[metric] = s
        print(f"{name:26s}{len(rows):5d}" + "".join(f"{c:>26s}" for c in cells))
    return pooled


def cmd_report(cfg: dict[str, Any], out_dir: Path) -> None:
    overlap_cfg = cfg["overlap"]
    metrics = metric_names(overlap_cfg["ngram_sizes"])
    rng = random.Random(cfg["seed"])
    attacks = [Attack(**row) for row in read_jsonl(out_dir / "attacks.jsonl")]
    attack_prompts = load_attack_prompts(os.path.expanduser(cfg["attack_root"]), cfg["tasks"])
    paraphrases, n_invalid = load_paraphrases(out_dir / "paraphrases.jsonl")
    missing = sorted({a.prompt_id for a in attacks if a.prompt_id not in paraphrases})
    if missing:
        raise SystemExit(f"{len(missing)} problems have no valid paraphrase ({n_invalid} invalid rows): {missing[:5]}")

    rows = [score_attack(a, attack_prompts[a.prompt_id], paraphrases[a.prompt_id], overlap_cfg["ngram_sizes"]) for a in attacks]
    write_jsonl(out_dir / "scores.jsonl", rows)
    n_fallback = sum(r["attribution_fallback"] for r in rows)
    mean_copy = sum(r["paraphrase_copy"] for r in rows) / len(rows)
    print(f"{len(rows)} attacks; {n_fallback} had no attribution sentence (scored on the whole CoT); {n_invalid} invalid paraphrase rows skipped")
    print(f"paraphrase copy (share of paraphrase 3-grams found in the literal requirement): mean {mean_copy:.2f}")

    groups = {"pooled": rows, **{side: [r for r in rows if r["side_task"] == side] for side in cfg["tasks"]}}
    groups = {name: g for name, g in groups.items() if g}
    groups["pooled, attribution only"] = [r for r in rows if not r["attribution_fallback"]]
    results = {version: print_table(version, groups, version, metrics, overlap_cfg, rng) for version in REQUESTED_VERSIONS}

    para = results["paraphrased"]
    best_metric = max(para, key=lambda m: para[m]["auroc"])
    bins = quantile_bins([r["paraphrase_copy"] for r in rows], overlap_cfg["n_copy_bins"])
    binned = {}
    for b in range(overlap_cfg["n_copy_bins"]):
        bin_rows = [r for r, rb in zip(rows, bins) if rb == b]
        copies = [r["paraphrase_copy"] for r in bin_rows]
        binned[f"copy bin {b} [{min(copies):.2f},{max(copies):.2f}]"] = bin_rows
    print_table(f"paraphrased, binned by paraphrase copy ({best_metric} only)", binned, "paraphrased", [best_metric], overlap_cfg, rng)

    best = para[best_metric]
    stop_auroc = cfg["thresholds"]["stop_auroc"]
    print(
        f"\n== Decision: best paraphrased metric {best_metric} AUROC {best['auroc']:.2f} "
        f"[{best['ci_low']:.2f},{best['ci_high']:.2f}] -> {decide(best['auroc'], stop_auroc)} (stop if >= {stop_auroc})"
    )


COMMANDS = {
    "sample": cmd_sample,
    "export": cmd_export,
    "report": cmd_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="analysis/configs/overlap_baseline.yaml")
    parser.add_argument("--out-dir", help="overrides out_dir from the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    COMMANDS[args.command](cfg, out_dir)


if __name__ == "__main__":
    main()
