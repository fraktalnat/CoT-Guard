"""Phase 1 report statistics: ΔTPR between the additive and substitutive variants (docs/phase1-preregistration.md).

Verdicts arrive as one row per monitor reading and are reduced once to per-problem counts:
every statistic and the bootstrap work on those counts, so a resample touches one entry per problem, not every reading.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Variant = Literal["additive", "substitutive"]


@dataclass(frozen=True)
class Verdict:
    problem: str
    variant: Variant
    sample: int
    draw: int
    verdict: str


@dataclass(frozen=True)
class Tally:
    flagged: int
    readings: int


ProblemCounts = dict[Variant, Tally]
Statistic = Callable[[list[ProblemCounts]], float]


def rows_from(table: dict[str, dict[Variant, list[str]]]) -> list[Verdict]:
    """{"P1": {"additive": ["CC", "CA"]}} -> one Verdict per letter."""
    return [
        Verdict(problem, variant, sample, draw, letter)
        for problem, variants in table.items()
        for variant, samples in variants.items()
        for sample, letters in enumerate(samples)
        for draw, letter in enumerate(letters)
    ]


def first_draw(rows: list[Verdict]) -> list[Verdict]:
    return [r for r in rows if r.draw == 0]


def to_counts(rows: list[Verdict]) -> list[ProblemCounts]:
    """One entry per problem: C readings and all readings, per variant present. "None" counts as a reading, not a C."""
    flagged: dict[tuple[str, Variant], int] = {}
    readings: dict[tuple[str, Variant], int] = {}
    for r in rows:
        key = (r.problem, r.variant)
        flagged[key] = flagged.get(key, 0) + (r.verdict == "C")
        readings[key] = readings.get(key, 0) + 1
    problems: dict[str, ProblemCounts] = {}
    for (problem, variant), n in readings.items():
        problems.setdefault(problem, {})[variant] = Tally(flagged[(problem, variant)], n)
    return list(problems.values())


def paired(problems: list[ProblemCounts]) -> list[ProblemCounts]:
    return [p for p in problems if "additive" in p and "substitutive" in p]


def variant_tpr(problems: list[ProblemCounts], variant: Variant) -> float:
    tallies = [p[variant] for p in problems if variant in p]
    return sum(t.flagged for t in tallies) / sum(t.readings for t in tallies)


def pooled_delta(problems: list[ProblemCounts]) -> float:
    return variant_tpr(problems, "additive") - variant_tpr(problems, "substitutive")


def problem_deltas(problems: list[ProblemCounts]) -> list[float]:
    return [pooled_delta([p]) for p in paired(problems)]


def problem_weighted_delta(problems: list[ProblemCounts]) -> float:
    deltas = problem_deltas(problems)
    return sum(deltas) / len(deltas)


def percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[round(q * (len(sorted_values) - 1))]


def cluster_bootstrap_ci(
    problems: list[ProblemCounts], statistic: Statistic, n_resamples: int, level: float, seed: int,
) -> tuple[float, float]:
    """Percentile interval of `statistic` when problems are drawn with replacement. A problem drawn twice
    is simply two list entries, so it counts twice in both deltas."""
    rng = random.Random(seed)
    values = sorted(statistic(rng.choices(problems, k=len(problems))) for _ in range(n_resamples))
    tail = (1 - level) / 2
    return percentile(values, tail), percentile(values, 1 - tail)
