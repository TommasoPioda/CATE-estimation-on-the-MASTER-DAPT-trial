#!/usr/bin/env python3
"""Generate and verify every table in Chapter 8 from saved experiment results.

Usage, from the repository root::

    python3 online-learning/scripts/generate_chapter8_tables.py
    python3 online-learning/scripts/generate_chapter8_tables.py --check

The first command writes six LaTeX ``tabular`` fragments and a machine-readable
CSV audit trail.  ``--check`` recomputes everything and fails if the committed
outputs are stale.  No causal forests are refitted: the inputs are the saved
per-run Parquet artifacts produced by the mechanism scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chapter8_table_utils import (
    PairedSummary,
    format_diff,
    format_p,
    format_rate,
    format_sem,
    paired_columns_summary,
    paired_groups_summary,
    render_tabular,
    require_columns,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "online-learning" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "markdown_docs" / "thesis" / "generated_tables"
AUDIT_NAME = "chapter8_table_audit.csv"
SCORE_CONVENTION = (
    "conflict=bleed-isch;net_benefit=bleed+isch;"
    "angle_net_benefit=+45;angle_conflict=-45"
)

M1_SOURCE = RESULTS_DIR / "results_policy_comparison_full.parquet"
M2_SOURCE = RESULTS_DIR / "results_mechanism2_sample_select.parquet"
M3_SOURCE = RESULTS_DIR / "results_mechanism3_duel_policies.parquet"
M45_SOURCE = RESULTS_DIR / "results_mechanism4_5_bandit_duel.parquet"

M1_CHECKPOINT = 3500
ISCH_WEIGHTS = {"death": 0.2, "mi": 0.4, "stroke": 0.4}

M1_ORDER = [
    ("conflict", "Conflict"),
    ("net_benefit", "Net-benefit"),
    ("angle_net_benefit", "Angle net-benefit"),
    ("angle_conflict", "Angle conflict"),
    ("random", "Random"),
    ("-isch", "$-$isch"),
    ("+isch", "$+$isch"),
]
M2_ORDER = [
    ("conflict", "Conflict"),
    ("net_benefit", "Net-benefit"),
    ("angle_net_benefit", "Angle net-benefit"),
    ("angle_conflict", "Angle conflict"),
    ("random", "Random"),
    ("minus_isch", "$-$isch"),
    ("isch", "$+$isch"),
]
M3_ORDER = [
    ("conflict", "Conflict"),
    ("net_benefit", "Net-benefit"),
    ("angle_net_benefit", "Angle net-benefit"),
    ("angle_conflict", "Angle conflict"),
    ("-isch", "$-$isch"),
    ("+isch", "$+$isch"),
    ("random", "Random"),
]


def _read(source: Path, required: set[str]) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_parquet(source)
    require_columns(data, required, source)

    if "score_convention" in data.columns:
        conventions = set(data["score_convention"].dropna().astype(str))
        if conventions != {SCORE_CONVENTION}:
            raise ValueError(f"Unexpected score convention in {source}: {conventions}")
        data.attrs["legacy_score_convention"] = False
        return data

    # Legacy artifacts were generated when the scalar identifiers were attached to the
    # opposite formulas: `conflict` stored bleed+isch and `net_benefit` stored bleed-isch.
    # Swap identifiers only; numerical values are never changed. New producers write the
    # explicit `score_convention` column above and therefore bypass this migration.
    data = data.copy()
    for column in ("policy", "variant", "regime"):
        if column not in data.columns:
            continue
        data[column] = data[column].replace(
            {"conflict": "__legacy_win_win__", "net_benefit": "conflict"}
        ).replace({"__legacy_win_win__": "net_benefit"})
    data.attrs["legacy_score_convention"] = True
    print(f"normalised legacy scalar labels: {source.relative_to(REPO_ROOT)}")
    return data


def _table_order(
    order: list[tuple[str, str]], data: pd.DataFrame
) -> list[tuple[str, str]]:
    """Return labels that do not present legacy +45 output as Angle conflict.

    Untagged artifacts predate the directional split: both Angle identifiers invoked
    the same +45-degree scorer. The valid legacy estimate is therefore reported once,
    as Angle net-benefit. The current -45-degree Angle conflict estimate is unavailable
    until that mechanism is rerun. Tagged future artifacts report both policies.
    """
    if not data.attrs.get("legacy_score_convention", False):
        return order
    return [
        (key, r"Angle NB (legacy)" if key == "angle_net_benefit" else label)
        for key, label in order
        if key != "angle_conflict"
    ]


def _require_run_count(data: pd.DataFrame, by: list[str], expected: int, context: str) -> None:
    counts = data.groupby(by, observed=True)["run"].nunique()
    if counts.empty or not (counts == expected).all():
        raise ValueError(
            f"{context} expected {expected} runs for every {by}, got {counts.to_dict()}"
        )


def _record(
    audit: list[dict[str, object]],
    *,
    source: Path,
    table: str,
    policy: str,
    endpoint: str,
    group_a: str,
    group_b: str,
    summary: PairedSummary,
) -> None:
    audit.append(
        {
            "source": str(source.relative_to(REPO_ROOT)),
            "table": table,
            "policy": policy,
            "endpoint": endpoint,
            "group_a": group_a,
            "group_b": group_b,
            **summary.audit_dict(),
        }
    )


def _standard_row(label: str, endpoint: str, result: PairedSummary) -> str:
    return (
        f"{label} & {endpoint} & {format_rate(result.mean_a_pct)} & "
        f"{format_rate(result.mean_b_pct)} & {format_diff(result.diff_pp)} & "
        f"{format_sem(result.sem_diff_pp)} & {format_p(result.p_value)} \\\\"
    )


def _row_without_endpoint(label: str, result: PairedSummary) -> str:
    return (
        f"{label} & {format_rate(result.mean_a_pct)} & "
        f"{format_rate(result.mean_b_pct)} & {format_diff(result.diff_pp)} & "
        f"{format_sem(result.sem_diff_pp)} & {format_p(result.p_value)} \\\\"
    )


def build_mechanism1(audit: list[dict[str, object]]) -> dict[str, str]:
    required = {
        "run", "policy", "n", "new_n", "left_n", "new_isch_n", "left_isch_n",
        "new_bleed_n", "left_bleed_n",
    }
    data = _read(M1_SOURCE, required)
    table_order = _table_order(M1_ORDER, data)
    final = data.loc[data["n"] == M1_CHECKPOINT].copy()
    if final.empty:
        raise ValueError(f"{M1_SOURCE} has no n={M1_CHECKPOINT} checkpoint")
    if set(final["policy"]) != {key for key, _ in M1_ORDER}:
        raise ValueError("Mechanism 1 policy set does not match the Chapter 8 table")
    if set(final["new_n"]) != {2500} or set(final["left_n"]) != {1079}:
        raise ValueError("Unexpected Mechanism 1 selected/left-out group sizes")
    _require_run_count(final, ["policy"], 100, "Mechanism 1")

    comparison_files = {
        "isch": "ch8_mechanism1_ischaemic.tex",
        "bleed": "ch8_mechanism1_bleeding.tex",
    }
    comparison_outputs: dict[str, str] = {}
    endpoint_names = {"isch": "Ischaemic", "bleed": "Bleeding"}

    for endpoint, filename in comparison_files.items():
        selected_col = f"selected_{endpoint}_rate"
        left_col = f"left_{endpoint}_rate"
        final[selected_col] = final[f"new_{endpoint}_n"] / final["new_n"]
        final[left_col] = final[f"left_{endpoint}_n"] / final["left_n"]
        rows: list[str] = []
        for key, label in table_order:
            subset = final.loc[final["policy"] == key]
            result = paired_columns_summary(
                subset, run_col="run", a_col=selected_col, b_col=left_col
            )
            rows.append(
                f"{label} & {format_rate(result.mean_a_pct)} & "
                f"{format_rate(result.mean_b_pct)} & {format_diff(result.diff_pp)} & "
                f"{format_sem(result.sem_diff_pp)} & {format_p(result.p_value)} \\\\"
            )
            _record(
                audit,
                source=M1_SOURCE,
                table=filename.removesuffix(".tex"),
                policy=key,
                endpoint=endpoint_names[endpoint],
                group_a="actively selected (seed excluded)",
                group_b="left out",
                summary=result,
            )
        comparison_outputs[filename] = render_tabular(
            alignment="lrrrrr",
            header_lines=[
                f"Policy & Selected {endpoint_names[endpoint].lower()} (\\%) & Left-out (\\%) &",
                "Diff. (pp) & SEM (pp) & $p$ \\\\",
            ],
            rows=rows,
        )

    rows = []
    for key, label in [item for item in table_order if item[0] != "random"]:
        values: dict[str, PairedSummary] = {}
        for endpoint in ("isch", "bleed"):
            rate_col = f"selected_{endpoint}_rate"
            wide = final.pivot(index="run", columns="policy", values=rate_col).reset_index()
            result = paired_columns_summary(
                wide, run_col="run", a_col=key, b_col="random"
            )
            values[endpoint] = result
            _record(
                audit,
                source=M1_SOURCE,
                table="ch8_mechanism1_vs_random",
                policy=key,
                endpoint=endpoint_names[endpoint],
                group_a="actively selected (seed excluded)",
                group_b="matched random policy",
                summary=result,
            )
        rows.append(
            f"{label} & {format_diff(values['isch'].diff_pp)} & "
            f"{format_p(values['isch'].p_value)} & "
            f"{format_diff(values['bleed'].diff_pp)} & "
            f"{format_p(values['bleed'].p_value)} \\\\"
        )

    comparison_outputs["ch8_mechanism1_vs_random.tex"] = render_tabular(
        alignment="lrr|rr",
        header_lines=[
            "Policy & \\multicolumn{2}{c|}{Ischaemic} & \\multicolumn{2}{c}{Bleeding} \\\\ ",
            "& $\\Delta$ (pp) & $p$ & $\\Delta$ (pp) & $p$ \\\\ ",
        ],
        rows=rows,
    )
    return comparison_outputs


def build_mechanism2(audit: list[dict[str, object]]) -> dict[str, str]:
    required = {
        "variant", "run", "group", "n", "bleed_rate", "isch_weighted_rate"
    }
    data = _read(M2_SOURCE, required)
    table_order = _table_order(M2_ORDER, data)
    if set(data["variant"]) != {key for key, _ in M2_ORDER}:
        raise ValueError("Mechanism 2 policy set does not match the Chapter 8 table")
    _require_run_count(data, ["variant", "group"], 100, "Mechanism 2")
    size_sets = data.groupby("group", observed=True)["n"].apply(lambda values: set(values))
    if size_sets.to_dict() != {"discarded": {1790}, "included": {1789}}:
        raise ValueError(f"Unexpected Mechanism 2 group sizes: {size_sets.to_dict()}")

    rows = []
    for key, label in table_order:
        subset = data.loc[data["variant"] == key]
        for endpoint, value_col in (
            ("Bleeding", "bleed_rate"),
            ("Ischaemic", "isch_weighted_rate"),
        ):
            result = paired_groups_summary(
                subset,
                run_col="run",
                group_col="group",
                value_col=value_col,
                group_a="included",
                group_b="discarded",
            )
            rows.append(_standard_row(label, endpoint, result))
            _record(
                audit,
                source=M2_SOURCE,
                table="ch8_mechanism2",
                policy=key,
                endpoint=endpoint,
                group_a="included",
                group_b="discarded",
                summary=result,
            )
    return {
        "ch8_mechanism2.tex": render_tabular(
            alignment="llrrrrr",
            header_lines=[
                "Policy & Endpoint & Included (\\%) & Discarded (\\%) & Diff. (pp) & SEM (pp) & $p$ \\\\"
            ],
            rows=rows,
        )
    }


def _mechanism3_wide() -> pd.DataFrame:
    required = {"policy", "run", "group", "endpoint", "n", "rate"}
    data = _read(M3_SOURCE, required)
    index = ["policy", "run", "group", "n"]
    wide = data.pivot(index=index, columns="endpoint", values="rate").reset_index()
    for endpoint in ("bleed", *ISCH_WEIGHTS):
        if endpoint not in wide:
            raise ValueError(f"Mechanism 3 is missing endpoint {endpoint!r}")
    wide["Bleeding"] = wide["bleed"]
    wide["Ischaemic"] = sum(
        weight * wide[endpoint] for endpoint, weight in ISCH_WEIGHTS.items()
    )
    wide.attrs["legacy_score_convention"] = data.attrs.get(
        "legacy_score_convention", False
    )

    return wide


def build_mechanism3(audit: list[dict[str, object]]) -> dict[str, str]:
    data = _mechanism3_wide()
    table_order = _table_order(M3_ORDER, data)
    if set(data["policy"]) != {key for key, _ in M3_ORDER}:
        raise ValueError("Mechanism 3 policy set does not match the Chapter 8 table")
    _require_run_count(data, ["policy", "group"], 100, "Mechanism 3")
    if set(data["n"]) != {1789}:
        raise ValueError("Unexpected Mechanism 3 group size")
    
    bleeding_rows = []
    ischaemic_rows = []

    for key, label in table_order:
        subset = data.loc[data["policy"] == key]
        for endpoint in ("Bleeding", "Ischaemic"):
            result = paired_groups_summary(
                subset,
                run_col="run",
                group_col="group",
                value_col=endpoint,
                group_a="included",
                group_b="excluded",
            )
            
            row = _row_without_endpoint(label, result)
            
            if endpoint == "Bleeding":
                bleeding_rows.append(row)
            else:
                ischaemic_rows.append(row)

            _record(
                audit,
                source=M3_SOURCE,
                table="ch8_mechanism3",
                policy=key,
                endpoint=endpoint,
                group_a="included",
                group_b="excluded",
                summary=result,
            )

    header = [
        "Duel rule & Included (\\%) & Excluded (\\%) & Diff. (pp) & SEM (pp) & $p$ \\\\"
    ]

    return {
        "ch8_mechanism3_bleeding.tex": render_tabular(
            alignment="lrrrrr",
            header_lines=header,
            rows=bleeding_rows,
        ),
        "ch8_mechanism3_ischaemic.tex": render_tabular(
            alignment="lrrrrr",
            header_lines=header,
            rows=ischaemic_rows,
        ),
    }


def build_mechanisms45(audit: list[dict[str, object]]) -> dict[str, str]:
    required = {
        "policy", "regime", "run", "group", "n", "bleed_rate", "isch_weighted_rate"
    }
    data = _read(M45_SOURCE, required)
    _require_run_count(data, ["policy", "regime", "group"], 100, "Mechanisms 4/5")
    if set(data["n"]) != {2789}:
        raise ValueError("Unexpected Mechanisms 4/5 group size")
    order = [
        ("UCB1", "UCB-style", "conflict", "Conflict"),
        ("UCB1", "UCB-style", "net_benefit", "Net-benefit"),
        ("Thompson", "Thompson", "conflict", "Conflict"),
        ("Thompson", "Thompson", "net_benefit", "Net-benefit"),
    ]
    rows = []
    for policy, policy_label, regime, regime_label in order:
        subset = data.loc[(data["policy"] == policy) & (data["regime"] == regime)]
        if subset.empty:
            raise ValueError(f"Missing Mechanisms 4/5 combination: {policy}/{regime}")
        for endpoint, value_col in (
            ("Bleeding", "bleed_rate"),
            ("Ischaemic", "isch_weighted_rate"),
        ):
            result = paired_groups_summary(
                subset,
                run_col="run",
                group_col="group",
                value_col=value_col,
                group_a="selected",
                group_b="random",
            )
            rows.append(
                f"{policy_label} & {regime_label} & {endpoint} & "
                f"{format_rate(result.mean_a_pct)} & {format_rate(result.mean_b_pct)} & "
                f"{format_diff(result.diff_pp)} & {format_sem(result.sem_diff_pp)} & "
                f"{format_p(result.p_value)} \\\\"
            )
            _record(
                audit,
                source=M45_SOURCE,
                table="ch8_mechanisms45",
                policy=f"{policy}/{regime}",
                endpoint=endpoint,
                group_a="selected",
                group_b="matched random",
                summary=result,
            )
    return {
        "ch8_mechanisms45.tex": render_tabular(
            alignment="lllrrrrr",
            header_lines=[
                "Policy & Score & Endpoint & Selected (\\%) & Random (\\%) & Diff. (pp) & SEM (pp) & $p$ \\\\"
            ],
            rows=rows,
        )
    }


def build_outputs() -> tuple[dict[str, str], pd.DataFrame]:
    audit: list[dict[str, object]] = []
    outputs: dict[str, str] = {}
    for builder in (build_mechanism1, build_mechanism2, build_mechanism3, build_mechanisms45):
        overlap = set(outputs) & set(new_outputs := builder(audit))
        if overlap:
            raise ValueError(f"Duplicate output names: {sorted(overlap)}")
        outputs.update(new_outputs)
    audit_frame = pd.DataFrame(audit).sort_values(
        ["table", "policy", "endpoint"], kind="stable"
    )
    return outputs, audit_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="fail if generated files differ from the committed outputs",
    )
    args = parser.parse_args()

    outputs, audit = build_outputs()
    audit_csv = audit.to_csv(index=False, float_format="%.12g")
    expected = {**outputs, AUDIT_NAME: audit_csv}

    if args.check:
        stale = []
        for name, content in expected.items():
            path = args.output_dir / name
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path))
        if stale:
            raise SystemExit("Stale or missing generated files:\n  " + "\n  ".join(stale))
        print(f"OK: {len(outputs)} Chapter 8 tables match their Parquet sources.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in expected.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")
    print(f"Wrote {len(outputs)} LaTeX tables and {AUDIT_NAME} to {args.output_dir}")
    print(f"Audit rows: {len(audit)}; source runs: M1=100, M2=100, M3=100, M4/5=100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
