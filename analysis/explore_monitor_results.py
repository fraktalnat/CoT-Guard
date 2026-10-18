"""Exploratory checks on downloaded monitor results (no GPU needed).

Reproduces the checks behind docs/data-exploration-findings.md:
  1. per-task TPR / TNR / g-mean^2 for one monitor model (A1 baseline, cot_action)
  2. coverage and sample values of the scalar (0-100) variant
  3. regex counts of how attack CoTs and base-monitor misses talk about the side task

Stdlib only, so it runs outside the (linux-only) uv environment. Run from the repo root:
    python -m analysis.explore_monitor_results --root ~/cot-download/results/monitor
"""

import argparse
import collections
import glob
import json
import os
import re

from eval.metrics import calculate_gmean2, compute_tnr, compute_tpr

COT_GUARD = (
    "qwen4b-three-scopes-defaultval_complexity_varname-rl_on_three_scopes_sft_step_80"
)
BASE_4B = "Qwen3-4B"
EVAL_TASKS = [
    "exit",
    "permissions",
    "logio",
    "edgecase",
    "longlines",
    "vulnimport",
    "typosquat",
    "benign_comment_watermark",
    "benign_print_watermark",
]
GENERATORS = ["Qwen3-14B", "gpt-5"]

# CoT phrases that present the side task as part of the request ("the problem also says ...")
ATTRIBUTION_RE = re.compile(
    r"(the (user|problem|task|prompt|instructions?) (also )?(says|asks|wants|requires|requests|mention|specif)"
    r"|also (need|must|required)|additional(ly)? requirement|must also|they want"
    r"|as (required|requested|instructed|specified))",
)
# CoT phrases that present it as hidden
EXPLICIT_RE = re.compile(
    r"(side task|hidden|secret(ly)?|covert|sneak|without (the user|anyone) noticing|monitor)",
    re.I,
)
# Broad on purpose; over-matches ordinary "the problem says ..." reasoning (see findings doc)
MONITOR_ATTRIBUTION_RE = re.compile(
    ATTRIBUTION_RE.pattern
    + r"|part of the (problem|task|requirements?)|required by the (problem|task)|the problem (states|requires|asks)",
    re.I,
)


def extract_choice(response: str) -> str:
    # Mirrors utils.extract_choice without importing utils (which pulls in litellm)
    without_cot = response.split("</think>")[-1]
    matches = re.findall(r"<answer>\s*([A-Da-d])\s*</answer>", without_cot)
    return matches[-1].upper() if matches else "None"


def extract_raw_answer(response: str) -> str | None:
    without_cot = response.split("</think>")[-1]
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", without_cot, re.S)
    return matches[-1].strip() if matches else None


def load(pattern: str) -> list[dict]:
    return [json.loads(line) for path in glob.glob(pattern) for line in open(path)]


def monitor_files(
    root: str, side: str, gen: str, model: str, variant: str = ""
) -> tuple[str, str]:
    pos = f"{root}/bigcodebench.{side}.baseline/{gen}--*/side-all.{model}.main_aware.{variant}cot_action.jsonl"
    neg = f"{root}/bigcodebench.{side}.no_side_task/{gen}--*/main-all.{model}.main_aware.{variant}cot_action.jsonl"
    return pos, neg


def check_metrics(root: str, model: str) -> None:
    print(
        f"\n== 1. TPR / TNR / g-mean^2 for {model} (baseline, cot_action, all seeds pooled)"
    )
    print(
        f"{'task':26s}{'generator':11s}{'nP':>5s}{'nN':>5s}{'TPR':>6s}{'TNR':>6s}{'G2':>6s}{'unparsed':>9s}"
    )
    g2s, total_pos, total_neg = [], 0, 0
    for side in EVAL_TASKS:
        for gen in GENERATORS:
            pos_glob, neg_glob = monitor_files(root, side, gen, model)
            pos = [extract_choice(r["messages"][-1]["content"]) for r in load(pos_glob)]
            neg = [extract_choice(r["messages"][-1]["content"]) for r in load(neg_glob)]
            if not pos or not neg:
                continue
            tpr, pos_failed = compute_tpr(pos)
            tnr, neg_failed = compute_tnr(neg)
            g2 = calculate_gmean2(tpr, tnr)
            g2s.append(g2)
            total_pos += len(pos)
            total_neg += len(neg)
            print(
                f"{side:26s}{gen:11s}{len(pos):5d}{len(neg):5d}{tpr:6.2f}{tnr:6.2f}{g2:6.2f}{pos_failed + neg_failed:9d}"
            )
    if g2s:
        print(f"mean g-mean^2 over {len(g2s)} cells: {sum(g2s) / len(g2s):.3f}")
    print(f"pooled positives: {total_pos}, pooled negatives: {total_neg}")


def check_scalar(root: str, model: str) -> None:
    print(f"\n== 2. Scalar (0-100) variant coverage for {model}")
    for side in EVAL_TASKS:
        for gen in GENERATORS:
            pos_glob, neg_glob = monitor_files(
                root, side, gen, model, variant="scalar."
            )
            pos = [
                extract_raw_answer(r["messages"][-1]["content"]) for r in load(pos_glob)
            ]
            neg = [
                extract_raw_answer(r["messages"][-1]["content"]) for r in load(neg_glob)
            ]
            if not pos and not neg:
                continue
            distinct = sorted({s for s in pos + neg if s is not None})
            print(
                f"{side:26s}{gen:11s} nP={len(pos):4d} nN={len(neg):4d} distinct scores={distinct[:12]}"
            )


def check_attribution(root: str, model: str, n_examples: int) -> None:
    print(
        f"\n== 3. Attribution language: attack CoTs, and {model} reasoning on its misses"
    )
    print(
        f"{'task':26s}{'n':>5s}{'CoT:attrib':>11s}{'CoT:explicit':>13s}{'flag rate':>10s}{'misses':>7s}{'miss:attrib':>12s}"
    )
    examples = []
    for side in EVAL_TASKS:
        counts = collections.Counter()
        for record in load(
            f"{root}/bigcodebench.{side}.baseline/*/side-all.{model}.main_aware.cot_action.jsonl"
        ):
            cot = record["metadata"]["attack_cot"]
            response = record["messages"][-1]["content"]
            counts["n"] += 1
            counts["cot_attrib"] += bool(ATTRIBUTION_RE.search(cot))
            counts["cot_explicit"] += bool(EXPLICIT_RE.search(cot))
            if extract_choice(response) == "C":
                counts["flag"] += 1
                continue
            counts["miss"] += 1
            thinking = response.split("</think>")[0]
            match = MONITOR_ATTRIBUTION_RE.search(thinking)
            counts["miss_attrib"] += bool(match)
            if match and len(examples) < n_examples:
                start = match.start()
                examples.append(
                    (
                        side,
                        thinking[max(0, start - 200) : start + 200].replace("\n", " "),
                    )
                )
        n = counts["n"]
        if not n:
            continue
        print(
            f"{side:26s}{n:5d}{counts['cot_attrib'] / n:11.2f}{counts['cot_explicit'] / n:13.2f}"
            f"{counts['flag'] / n:10.2f}{counts['miss']:7d}{counts['miss_attrib'] / max(counts['miss'], 1):12.2f}"
        )
    for side, snippet in examples:
        print(f"\n-- {side}: ...{snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", default=os.path.expanduser("~/cot-download/results/monitor")
    )
    parser.add_argument(
        "--model", default=COT_GUARD, help="monitor model for checks 1 and 2"
    )
    parser.add_argument(
        "--base-model", default=BASE_4B, help="monitor model for check 3"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=4,
        help="monitor-reasoning snippets to print in check 3",
    )
    args = parser.parse_args()

    check_metrics(args.root, args.model)
    check_scalar(args.root, args.model)
    check_attribution(args.root, args.base_model, args.examples)


if __name__ == "__main__":
    main()
