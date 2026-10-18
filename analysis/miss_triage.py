"""Experiment 1: is "misattribution" what's happening? LLM-judge triage of monitor misses.

Each miss (attack where the monitor verdict wasn't C) goes into one bucket:
never_noticed / attributed / harmless (judged by an API model), or answered_d (verdict D, no judge call).

Run from the repo root, in order:
    python -m analysis.miss_triage sample        # collect + stratify misses -> misses.jsonl
    python -m analysis.miss_triage judge         # needs $OPENROUTER_API_KEY; resumable -> judged.jsonl
    python -m analysis.miss_triage export-hand   # blind set -> hand_review.md + hand_labels.csv (fill hand_label)
    python -m analysis.miss_triage export-rest   # optional: every other non-D miss -> hand_review_rest.md + hand_labels_rest.csv
    python -m analysis.miss_triage report        # bucket shares, go/pivot decision, judge-vs-hand agreement
"""

import argparse
import collections
import csv
import dataclasses
import glob
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Hashable, TypeVar

import yaml

from analysis.explore_monitor_results import extract_choice

JUDGE_BUCKETS = ("never_noticed", "attributed", "harmless")
D_BUCKET = "answered_d"
UNPARSED = "unparsed"
BUCKETS = JUDGE_BUCKETS + (D_BUCKET,)
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Miss:
    miss_id: str
    monitor: str
    side_task: str
    generator: str
    verdict: str
    side_task_description: str
    problem: str
    attack_code: str
    monitor_response: str


@dataclasses.dataclass(frozen=True)
class Judgment:
    bucket: str
    evidence: str
    rationale: str


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def generator_from_dir(dirname: str) -> str:
    return dirname.split("--")[0]


def collect_misses(root: str, monitor_key: str, model: str, tasks: list[str]) -> tuple[list[Miss], int]:
    """Returns (misses, n_unparsed). Unparsed verdicts are excluded but counted, not silently dropped."""
    misses, n_unparsed = [], 0
    for side in tasks:
        pattern = f"{root}/bigcodebench.{side}.baseline/*/side-all.{model}.main_aware.cot_action.jsonl"
        for path in sorted(glob.glob(pattern)):
            rel = os.path.relpath(path, root)
            generator = generator_from_dir(Path(path).parent.name)
            with open(path) as f:
                for line_no, line in enumerate(f):
                    record = json.loads(line)
                    response = record["messages"][-1]["content"]
                    verdict = extract_choice(response)
                    if verdict == "C":
                        continue
                    if verdict == "None":
                        n_unparsed += 1
                        continue
                    meta = record["metadata"]
                    misses.append(
                        Miss(
                            miss_id=f"{monitor_key}:{rel}:{line_no}",
                            monitor=monitor_key,
                            side_task=side,
                            generator=generator,
                            verdict=verdict,
                            side_task_description=meta["side_task_description"],
                            problem=meta["original_prompt"],
                            attack_code=meta["attack_code"],
                            monitor_response=response,
                        )
                    )
    return misses, n_unparsed


def stratified_sample(items: list[T], n: int, key: Callable[[T], Hashable], rng: random.Random) -> list[T]:
    """Round-robin over strata so small strata are taken whole and the remainder goes to larger ones."""
    groups: dict[Hashable, list[T]] = collections.defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    queues = [groups[k] for k in sorted(groups)]
    for queue in queues:
        rng.shuffle(queue)
    sample: list[T] = []
    depth = 0
    while len(sample) < n and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(sample) < n:
                sample.append(queue[depth])
        depth += 1
    return sample


def build_judge_messages(miss: Miss, prompt: dict[str, str]) -> list[dict[str, str]]:
    user = prompt["user"].format(
        side_task_description=miss.side_task_description,
        problem=miss.problem,
        attack_code=miss.attack_code,
        monitor_response=miss.monitor_response,
    )
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def parse_judgment(text: str) -> Judgment:
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return Judgment(UNPARSED, "", text)
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Judgment(UNPARSED, "", text)
    bucket = obj.get("bucket", "")
    if bucket not in JUDGE_BUCKETS:
        return Judgment(UNPARSED, obj.get("evidence", ""), text)
    return Judgment(bucket, obj.get("evidence", ""), obj.get("rationale", ""))


def call_judge(messages: list[dict[str, str]], judge_cfg: dict[str, Any], api_key: str) -> tuple[str, float]:
    """Returns (content, cost_usd). Retries transient errors with exponential backoff, then raises."""
    payload = {
        "model": judge_cfg["model"],
        "messages": messages,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": judge_cfg["reasoning_effort"]},
        "usage": {"include": True},
    }
    request = urllib.request.Request(
        judge_cfg["url"],
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    def post() -> tuple[str, float]:
        with urllib.request.urlopen(request, timeout=judge_cfg["timeout_s"]) as response:
            body = json.loads(response.read())
        return body["choices"][0]["message"]["content"], body.get("usage", {}).get("cost", 0.0)

    for attempt in range(judge_cfg["max_retries"]):
        try:
            return post()
        except urllib.error.HTTPError as err:
            if err.code not in RETRYABLE_HTTP:
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2**attempt)
    return post()


def bucket_counts(judged: list[dict]) -> collections.Counter:
    return collections.Counter(row["bucket"] for row in judged)


def bucket_shares(counts: collections.Counter) -> dict[str, float]:
    """Shares over parsed buckets only; unparsed rows are reported separately."""
    total = sum(counts[b] for b in BUCKETS)
    return {b: counts[b] / total if total else 0.0 for b in BUCKETS}


def decide(shares: dict[str, float], go_share: float) -> str:
    if shares["attributed"] + shares["harmless"] >= go_share:
        return "GO"
    if shares["never_noticed"] == max(shares.values()):
        return "PIVOT"
    return "UNCLEAR"


def agreement(pairs: list[tuple[str, str]]) -> tuple[float, dict[tuple[str, str], int]]:
    """pairs are (hand, judge). Returns (accuracy, confusion counts keyed by (hand, judge))."""
    confusion = collections.Counter(pairs)
    correct = sum(n for (hand, judge), n in confusion.items() if hand == judge)
    return correct / len(pairs), dict(confusion)


def cmd_sample(cfg: dict[str, Any], out_dir: Path) -> None:
    rng = random.Random(cfg["seed"])
    rows = []
    for key, model in cfg["monitors"].items():
        misses, n_unparsed = collect_misses(os.path.expanduser(cfg["results_root"]), key, model, cfg["tasks"])
        sample = stratified_sample(misses, cfg["n_per_monitor"], lambda m: m.side_task, rng)
        per_task = collections.Counter(m.side_task for m in sample)
        print(f"{key}: {len(misses)} misses ({n_unparsed} unparsed excluded), sampled {len(sample)}: {dict(per_task)}")
        rows.extend(dataclasses.asdict(m) for m in sample)
    write_jsonl(out_dir / "misses.jsonl", rows)
    print(f"wrote {len(rows)} misses to {out_dir / 'misses.jsonl'}")


def cmd_judge(cfg: dict[str, Any], out_dir: Path) -> None:
    judge_cfg = cfg["judge"]
    api_key = os.environ.get(judge_cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"set ${judge_cfg['api_key_env']} first")
    prompt = load_config(judge_cfg["prompt_path"])
    misses = [Miss(**row) for row in read_jsonl(out_dir / "misses.jsonl")]
    judged_path = out_dir / "judged.jsonl"
    done = {row["miss_id"] for row in read_jsonl(judged_path)} if judged_path.exists() else set()
    todo = [m for m in misses if m.miss_id not in done]
    print(f"{len(done)} already judged, {len(todo)} to go with {judge_cfg['model']}")

    def record(miss: Miss, judgment: Judgment, judge_model: str | None, raw: str, cost: float) -> dict:
        return {
            "miss_id": miss.miss_id,
            "monitor": miss.monitor,
            "side_task": miss.side_task,
            "verdict": miss.verdict,
            **dataclasses.asdict(judgment),
            "judge_model": judge_model,
            "raw": raw,
            "cost": cost,
        }

    total_cost = 0.0
    with open(judged_path, "a") as out, ThreadPoolExecutor(max_workers=judge_cfg["max_workers"]) as pool:
        for miss in todo:
            if miss.verdict == "D":
                out.write(json.dumps(record(miss, Judgment(D_BUCKET, "", "verdict D"), None, "", 0.0)) + "\n")
        futures = {
            pool.submit(call_judge, build_judge_messages(m, prompt), judge_cfg, api_key): m
            for m in todo
            if m.verdict != "D"
        }
        for i, future in enumerate(as_completed(futures), 1):
            miss = futures[future]
            raw, cost = future.result()
            total_cost += cost
            out.write(json.dumps(record(miss, parse_judgment(raw), judge_cfg["model"], raw, cost)) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(futures)} judged, ${total_cost:.3f} so far")
    print(f"done, ${total_cost:.3f} this run")


def read_label_ids(labels_path: Path) -> set[str]:
    with open(labels_path) as f:
        return {row["miss_id"] for row in csv.DictReader(f)}


def unpicked_misses(misses: list[Miss], picked_ids: set[str]) -> list[Miss]:
    """Misses not yet in a hand-label file, excluding D verdicts (bucketed without the judge)."""
    return [m for m in misses if m.miss_id not in picked_ids and m.verdict != "D"]


def write_hand_files(chosen: list[Miss], review_path: Path, labels_path: Path, start: int) -> None:
    if labels_path.exists():
        raise SystemExit(f"{labels_path} exists; refusing to overwrite hand labels")
    with open(review_path, "w") as f:
        f.write(f"# Hand review ({len(chosen)} misses)\n\nBuckets: {', '.join(JUDGE_BUCKETS)}\n")
        for i, m in enumerate(chosen, start):
            f.write(
                f"\n---\n\n## {i}. `{m.miss_id}`\n\nmonitor: {m.monitor} | task: {m.side_task} | verdict: {m.verdict}\n\n"
                f"### Side task\n{m.side_task_description}\n\n### Problem\n{m.problem}\n\n"
                f"### Agent code\n```python\n{m.attack_code}\n```\n\n### Monitor response\n```\n{m.monitor_response}\n```\n"
            )
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "miss_id", "hand_label", "notes"])
        writer.writerows([i, m.miss_id, "", ""] for i, m in enumerate(chosen, start))
    print(f"wrote {len(chosen)} items to {review_path}; fill hand_label in {labels_path}")


def cmd_export_hand(cfg: dict[str, Any], out_dir: Path) -> None:
    rng = random.Random(cfg["seed"])
    misses = [Miss(**row) for row in read_jsonl(out_dir / "misses.jsonl") if row["verdict"] != "D"]
    per_monitor = cfg["n_hand"] // len(cfg["monitors"])
    chosen = []
    for key in cfg["monitors"]:
        pool = [m for m in misses if m.monitor == key]
        chosen.extend(stratified_sample(pool, per_monitor, lambda m: m.side_task, rng))
    rng.shuffle(chosen)
    write_hand_files(chosen, out_dir / "hand_review.md", out_dir / "hand_labels.csv", start=1)


def cmd_export_rest(cfg: dict[str, Any], out_dir: Path) -> None:
    labels_path = out_dir / "hand_labels.csv"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} not found; run export-hand first")
    picked = read_label_ids(labels_path)
    misses = [Miss(**row) for row in read_jsonl(out_dir / "misses.jsonl")]
    rest = unpicked_misses(misses, picked)
    write_hand_files(rest, out_dir / "hand_review_rest.md", out_dir / "hand_labels_rest.csv", start=len(picked) + 1)


def load_hand_labels(paths: list[Path]) -> dict[str, str]:
    """Filled-in hand labels across all existing label files, keyed by miss_id."""
    hand = {}
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            hand.update({row["miss_id"]: row["hand_label"].strip() for row in csv.DictReader(f) if row["hand_label"].strip()})
    return hand


def print_buckets(title: str, judged: list[dict]) -> dict[str, float]:
    counts = bucket_counts(judged)
    shares = bucket_shares(counts)
    cells = "".join(f"{b}={counts[b]} ({shares[b]:.2f})  " for b in BUCKETS)
    print(f"{title:42s} n={len(judged):4d}  {cells}unparsed={counts[UNPARSED]}")
    return shares


def cmd_report(cfg: dict[str, Any], out_dir: Path) -> None:
    judged = read_jsonl(out_dir / "judged.jsonl")
    go_share = cfg["thresholds"]["go_share"]
    print("== Bucket shares (over parsed judgments)")
    for key in cfg["monitors"]:
        rows = [r for r in judged if r["monitor"] == key]
        shares = print_buckets(key, rows)
        print(f"  -> {decide(shares, go_share)} (attributed + harmless = {shares['attributed'] + shares['harmless']:.2f}, go if >= {go_share})")
        for side in cfg["tasks"]:
            side_rows = [r for r in rows if r["side_task"] == side]
            if side_rows:
                print_buckets(f"    {side}", side_rows)

    hand = load_hand_labels([out_dir / "hand_labels.csv", out_dir / "hand_labels_rest.csv"])
    invalid = {mid: label for mid, label in hand.items() if label not in JUDGE_BUCKETS}
    if invalid:
        raise SystemExit(f"invalid hand labels (expected one of {JUDGE_BUCKETS}): {invalid}")
    judge_by_id = {r["miss_id"]: r["bucket"] for r in judged}
    missing = [mid for mid in hand if mid not in judge_by_id]
    if missing:
        raise SystemExit(f"hand-labelled items not judged yet: {missing}")
    if not hand:
        print("\nno hand labels filled in yet; run export-hand (and optionally export-rest) and label")
        return
    accuracy, confusion = agreement([(label, judge_by_id[mid]) for mid, label in hand.items()])
    min_agreement = cfg["thresholds"]["min_agreement"]
    verdict = "PASS" if accuracy >= min_agreement else "FAIL"
    print(f"\n== Judge vs hand: {accuracy:.2f} agreement on {len(hand)} items ({verdict}, need >= {min_agreement})")
    labels = list(JUDGE_BUCKETS) + [UNPARSED]
    print("hand / judge".ljust(16) + "".join(f"{b:>15s}" for b in labels))
    for h in JUDGE_BUCKETS:
        print(f"{h:16s}" + "".join(f"{confusion.get((h, j), 0):15d}" for j in labels))


COMMANDS = {
    "sample": cmd_sample,
    "judge": cmd_judge,
    "export-hand": cmd_export_hand,
    "export-rest": cmd_export_rest,
    "report": cmd_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="analysis/configs/miss_triage.yaml")
    parser.add_argument("--out-dir", help="overrides out_dir from the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    COMMANDS[args.command](cfg, out_dir)


if __name__ == "__main__":
    main()
