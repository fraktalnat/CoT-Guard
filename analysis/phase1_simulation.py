"""Phase 1: fake-data check of the report statistics (analysis/phase1_report.py).

Simulates verdicts with a known ΔTPR and the experiment's hierarchy:
  problem  substitutive rate ~ N(sub_mean, problem_sd); additive rate = that + N(true_delta, delta_sd)
  sample   each of `rollouts` generations succeeds with the variant's side_pass rate; its own rate ~ problem rate + N(0, sample_sd)
  draw     k monitor readings per successful sample, each C with the sample's rate
Then asks, over many simulated experiments: how often does the 95% problem-clustered bootstrap CI contain the truth,
how wide is it, and how often does it exclude 0? Each CI method in the config (percentile, BCa) is scored on the same resamples.

Run from the repo root:
    python -m analysis.phase1_simulation
"""

import argparse
import random
from dataclasses import dataclass
from typing import Any

from analysis.miss_triage import load_config
from analysis.phase1_report import (
    Interval,
    Statistic,
    Variant,
    Verdict,
    bca_interval,
    bootstrap_values,
    paired,
    percentile_interval,
    pooled_delta,
    problem_weighted_delta,
    to_counts,
)

VARIANTS: tuple[Variant, ...] = ("additive", "substitutive")
STATISTICS: dict[str, Statistic] = {
    "pooled": pooled_delta,
    "problem_weighted": problem_weighted_delta,
}
INTERVALS: dict[str, Interval] = {
    "percentile": percentile_interval,
    "bca": bca_interval,
}


@dataclass(frozen=True)
class SimParams:
    n_problems: int
    rollouts: int
    k: int
    side_pass_rate: dict[Variant, float]
    sub_mean: float
    problem_sd: float
    true_delta: float
    delta_sd: float
    sample_sd: float


def clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def problem_rates(p: SimParams, rng: random.Random) -> dict[Variant, float]:
    sub = clip01(rng.gauss(p.sub_mean, p.problem_sd))
    return {"additive": clip01(sub + rng.gauss(p.true_delta, p.delta_sd)), "substitutive": sub}


def sample_rate(problem_rate: float, p: SimParams, rng: random.Random) -> float:
    return clip01(problem_rate + rng.gauss(0, p.sample_sd))


def simulate(p: SimParams, rng: random.Random) -> list[Verdict]:
    rows: list[Verdict] = []
    for i in range(p.n_problems):
        rates = problem_rates(p, rng)
        for variant in VARIANTS:
            n_success = sum(rng.random() < p.side_pass_rate[variant] for _ in range(p.rollouts))
            for sample in range(n_success):
                rate = sample_rate(rates[variant], p, rng)
                rows += [Verdict(f"P{i}", variant, sample, draw, "C" if rng.random() < rate else "A") for draw in range(p.k)]
    return rows


def population_delta(p: SimParams, rng: random.Random, n_draws: int) -> float:
    """The estimand: mean over problems and samples of the additive minus substitutive detection rate.
    Not exactly p.true_delta, because clipping at 0 and 1 shrinks it."""
    total = 0.0
    for _ in range(n_draws):
        rates = problem_rates(p, rng)
        total += sample_rate(rates["additive"], p, rng) - sample_rate(rates["substitutive"], p, rng)
    return total / n_draws


def coverage(
    p: SimParams, statistic: Statistic, truth: float, n_sims: int, n_resamples: int, level: float, seed: int,
    intervals: dict[str, Interval],
) -> dict[str, dict[str, float]]:
    """Per CI method. Every method reads the same resamples of the same simulated experiments, so they differ only in the interval."""
    rng = random.Random(seed)
    hits = {name: 0 for name in intervals}
    widths = {name: 0.0 for name in intervals}
    excludes_zero = {name: 0 for name in intervals}
    errors, n_paired = 0.0, 0
    for sim in range(n_sims):
        problems = paired(to_counts(simulate(p, rng)))
        values = bootstrap_values(problems, statistic, n_resamples, seed=seed + sim)
        for name, interval in intervals.items():
            low, high = interval(values, problems, statistic, level)
            hits[name] += low <= truth <= high
            widths[name] += high - low
            excludes_zero[name] += low > 0
        errors += statistic(problems) - truth
        n_paired += len(problems)
    return {
        name: {
            "coverage": hits[name] / n_sims, "mean_width": widths[name] / n_sims, "bias": errors / n_sims,
            "excludes_zero": excludes_zero[name] / n_sims, "mean_paired_problems": n_paired / n_sims,
        }
        for name in intervals
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="analysis/configs/phase1_simulation.yaml")
    cfg: dict[str, Any] = load_config(parser.parse_args().config)

    intervals = {name: INTERVALS[name] for name in cfg["ci_methods"]}
    print(f"{'scenario':14} {'statistic':17} {'ci':10} {'paired':>6} {'truth':>6} {'bias':>7} {'width':>6} {'coverage':>8} {'CI>0':>6}")
    for name, scenario in cfg["scenarios"].items():
        p = SimParams(**{**cfg["params"], **scenario})
        truth = population_delta(p, random.Random(cfg["seed"]), cfg["truth_draws"])
        for stat_name, statistic in STATISTICS.items():
            results = coverage(p, statistic, truth, cfg["n_sims"], cfg["n_resamples"], cfg["level"], cfg["seed"], intervals)
            for ci_name, r in results.items():
                print(f"{name:14} {stat_name:17} {ci_name:10} {r['mean_paired_problems']:6.1f} {truth:6.3f} {r['bias']:+7.3f} "
                      f"{r['mean_width']:6.3f} {r['coverage']:8.1%} {r['excludes_zero']:6.1%}")


if __name__ == "__main__":
    main()
