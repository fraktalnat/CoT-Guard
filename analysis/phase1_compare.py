"""Phase 1 secondary runs: contrasts across monitor scopes and monitor models.
Every condition is reduced to per-problem counts, the conditions are joined on problem, and the bootstrap
draws problems once for all conditions, so every contrast keeps the pairing.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

from analysis.miss_triage import load_config
from analysis.phase1_report import (
    VARIANTS,
    ProblemCounts,
    Verdict,
    bootstrap_values,
    load_run,
    percentile,
    pooled_delta,
    to_counts,
    variant_tpr,
)

Condition = str
JointProblem = dict[Condition, ProblemCounts]
JointStatistic = Callable[[list[JointProblem]], float]


def condition_name(monitor: str, scope: str) -> Condition:
    return f"{monitor}.{scope}"


def counts_by_problem(rows: list[Verdict]) -> dict[str, ProblemCounts]:
    by_problem: dict[str, list[Verdict]] = defaultdict(list)
    for row in rows:
        by_problem[row.problem].append(row)
    return {
        problem: to_counts(problem_rows)[0]
        for problem, problem_rows in by_problem.items()
    }


def paired_problem_ids(counts: dict[str, ProblemCounts]) -> set[str]:
    return {
        problem
        for problem, tallies in counts.items()
        if all(v in tallies for v in VARIANTS)
    }


def join_conditions(
    counts: dict[Condition, dict[str, ProblemCounts]],
) -> list[JointProblem]:
    paired_sets = {condition: paired_problem_ids(c) for condition, c in counts.items()}
    reference = next(iter(paired_sets.values()))
    mismatched = sorted(
        condition for condition, ids in paired_sets.items() if ids != reference
    )
    if mismatched:
        raise SystemExit(f"paired problems differ across conditions: {mismatched}")
    return [
        {condition: c[problem] for condition, c in counts.items()}
        for problem in sorted(reference, key=lambda p: (len(p), p))
    ]


def view(joint: list[JointProblem], condition: Condition) -> list[ProblemCounts]:
    return [problem[condition] for problem in joint]


def tpr_of(condition: Condition, variant: str) -> JointStatistic:
    return lambda joint: variant_tpr(view(joint, condition), variant)


def delta_of(condition: Condition) -> JointStatistic:
    return lambda joint: pooled_delta(view(joint, condition))


def retained_of(condition: Condition) -> JointStatistic:
    return lambda joint: (
        tpr_of(condition, "substitutive")(joint) / tpr_of(condition, "additive")(joint)
    )


def difference(first: JointStatistic, second: JointStatistic) -> JointStatistic:
    return lambda joint: first(joint) - second(joint)


def with_ci(
    joint: list[JointProblem],
    statistic: JointStatistic,
    n_resamples: int,
    level: float,
    seed: int,
) -> dict[str, Any]:
    values = bootstrap_values(joint, statistic, n_resamples, seed)
    tail = (1 - level) / 2
    return {
        "estimate": statistic(joint),
        "ci_percentile": (percentile(values, tail), percentile(values, 1 - tail)),
    }


def bound_status(tpr: dict[str, float], floor: float, ceiling: float) -> str | None:
    if all(t <= floor for t in tpr.values()):
        return "floor"
    if all(t >= ceiling for t in tpr.values()):
        return "ceiling"
    return None


def analyse(
    joint: list[JointProblem], monitors: dict[str, list[str]], cfg: dict[str, Any]
) -> dict[str, Any]:
    ci_args = (cfg["n_resamples"], cfg["level"], cfg["seed"])
    floor, ceiling = cfg["floor"], cfg["ceiling"]
    first_scope, second_scope = cfg["scope_contrast"]

    conditions = {}
    for monitor, scopes in monitors.items():
        for scope in scopes:
            name = condition_name(monitor, scope)
            tpr = {v: tpr_of(name, v)(joint) for v in VARIANTS}
            conditions[name] = {
                "tpr": tpr,
                "bound": bound_status(tpr, floor, ceiling),
                "delta": with_ci(joint, delta_of(name), *ci_args),
                "retained": with_ci(joint, retained_of(name), *ci_args)
                if tpr["additive"] > floor
                else None,
            }

    scope_contrasts = {}
    for monitor, scopes in monitors.items():
        if first_scope in scopes and second_scope in scopes:
            first, second = (
                condition_name(monitor, first_scope),
                condition_name(monitor, second_scope),
            )
            scope_contrasts[monitor] = {
                "substitutive_tpr": with_ci(
                    joint,
                    difference(
                        tpr_of(first, "substitutive"), tpr_of(second, "substitutive")
                    ),
                    *ci_args,
                ),
                "did": with_ci(
                    joint, difference(delta_of(first), delta_of(second)), *ci_args
                ),
            }

    def comparable(name: Condition) -> bool:
        return (
            name in conditions
            and conditions[name]["retained"] is not None
            and conditions[name]["bound"] is None
        )

    monitor_contrasts = {}
    for first_monitor, second_monitor in combinations(monitors, 2):
        for scope in (first_scope, second_scope):
            first, second = (
                condition_name(first_monitor, scope),
                condition_name(second_monitor, scope),
            )
            if comparable(first) and comparable(second):
                monitor_contrasts[f"{first} − {second}"] = with_ci(
                    joint, difference(retained_of(first), retained_of(second)), *ci_args
                )

    return {
        "paired_problems": len(joint),
        "conditions": conditions,
        "scope_contrasts": scope_contrasts,
        "monitor_contrasts": monitor_contrasts,
        "main": scope_contrasts.get(cfg["main_monitor"], {}).get("did"),
    }


def fmt_ci(result: dict[str, Any]) -> str:
    low, high = result["ci_percentile"]
    return f"{result['estimate']:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]"


def render(report: dict[str, Any], cfg: dict[str, Any]) -> str:
    first_scope, second_scope = cfg["scope_contrast"]
    main = report["main"]
    lines = [
        "# Phase 1 secondary runs: scope × monitor contrasts (secondary, exploratory; no verdict)",
        "",
        f"- paired problems, drawn jointly across conditions: {report['paired_problems']}",
        f"- floor ≤ {cfg['floor']}, ceiling ≥ {cfg['ceiling']}: a condition with both variants at one of them is not interpreted",
        "",
        f"## Main contrast: {cfg['main_monitor']} DiD = ΔTPR({first_scope}) − ΔTPR({second_scope})",
        f"- {fmt_ci(main)}"
        if main
        else f"- not available: {cfg['main_monitor']} lacks one of the two scopes",
        "",
        "## Per condition (descriptive)",
    ]
    for name, c in report["conditions"].items():
        bound = f"  [{c['bound'].upper()}: not interpreted]" if c["bound"] else ""
        retained = (
            fmt_ci(c["retained"])
            if c["retained"]
            else f"not computed (additive TPR ≤ {cfg['floor']})"
        )
        lines += [
            f"### {name}{bound}",
            f"- TPR additive {c['tpr']['additive']:.3f}, substitutive {c['tpr']['substitutive']:.3f}",
            f"- ΔTPR {fmt_ci(c['delta'])}",
            f"- retained detection (TPR_sub / TPR_add) {retained}",
        ]
    lines += [
        "",
        f"## Scope contrasts per monitor ({first_scope} − {second_scope}, descriptive)",
    ]
    for monitor, s in report["scope_contrasts"].items():
        lines += [
            f"- {monitor}: substitutive TPR {fmt_ci(s['substitutive_tpr'])}",
            f"- {monitor}: DiD {fmt_ci(s['did'])}",
        ]
    lines += ["", "## Retained detection across monitors, per scope (descriptive)"]
    lines += [
        f"- {name}: {fmt_ci(result)}"
        for name, result in report["monitor_contrasts"].items()
    ]
    return "\n".join(lines) + "\n"


def load_conditions(
    root: Path, monitors: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict[Condition, dict[str, ProblemCounts]]]:
    scopes: dict[str, list[str]] = {}
    counts = {}
    for monitor, report_configs in monitors.items():
        scopes[monitor] = []
        for path in report_configs:
            report_cfg = load_config(path)
            rows, _, issues = load_run(root, report_cfg)
            if issues:
                raise SystemExit(
                    f"{path}: incomplete data, no comparison written:\n  "
                    + "\n  ".join(issues)
                )
            scope = report_cfg["monitor_type"]
            scopes[monitor].append(scope)
            counts[condition_name(monitor, scope)] = counts_by_problem(rows)
    return scopes, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="analysis/configs/phase1_compare.yaml")
    cfg = load_config(parser.parse_args().config)
    scopes, counts = load_conditions(
        Path(cfg["global_root"]).expanduser(), cfg["monitors"]
    )
    report = analyse(join_conditions(counts), scopes, cfg)
    text = render(report, cfg)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(text)
    print(text)
    print(f"written to {out_dir}/report.{{json,md}}")


if __name__ == "__main__":
    main()
