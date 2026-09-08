import os
import pandas as pd
from scipy.stats import fisher_exact

from online_learning_policies import SCORE_CONVENTION
from run_archiving import start_run_archive

HERE = os.path.dirname(os.path.abspath(__file__))
OL_DIR = os.path.normpath(os.path.join(HERE, ".."))
RUN_DIR = start_run_archive(OL_DIR, "mechanism2_statistics")

RESULTS_FILE = os.path.join(
    OL_DIR, "results",
    "results_mechanism2_sample_select.parquet",
)

OUT_STATS = os.path.join(
    OL_DIR, "results",
    "mechanism2_statistics.parquet",
)

OUT_TEX = os.path.join(
    OL_DIR, "results",
    "mechanism2_statistics.tex",
)

ENDPOINTS = {
    "isch": "isch",
    "bleed": "bleed",
}

results = pd.read_parquet(RESULTS_FILE)


def normalise_saved_policy_ids(df):
    """Return policy identifiers under the current score convention.

    The checked-in Mechanism 2 artifact predates ``SCORE_CONVENTION``: its scalar
    identifiers were reversed (``conflict`` stored the win-win sum and
    ``net_benefit`` stored the conflict difference), and both Angle identifiers
    contain the same +45-degree result. Numeric values are never changed here.

    New, tagged artifacts already contain genuinely distinct +45/-45 Angle runs and
    therefore pass through unchanged. Unknown, missing or mixed tags fail loudly.
    """
    out = df.copy()
    if "score_convention" in out.columns:
        tags = set(out["score_convention"].dropna().astype(str))
        if out["score_convention"].isna().any() or tags != {SCORE_CONVENTION}:
            raise ValueError(f"Unsupported score_convention values: {sorted(tags)}")
        return out, False

    out["variant"] = (
        out["variant"]
        .replace({"conflict": "__legacy_win_win__", "net_benefit": "conflict"})
        .replace({"__legacy_win_win__": "net_benefit"})
    )

    angle_nb = out[out["variant"] == "angle_net_benefit"].drop(columns="variant")
    angle_conflict = out[out["variant"] == "angle_conflict"].drop(columns="variant")
    if not angle_conflict.empty:
        if angle_nb.empty:
            out.loc[
                out["variant"] == "angle_conflict", "variant"
            ] = "angle_net_benefit"
        else:
            order = [c for c in ("run", "group") if c in angle_nb.columns]
            if order:
                angle_nb = angle_nb.sort_values(order).reset_index(drop=True)
                angle_conflict = angle_conflict.sort_values(order).reset_index(drop=True)
            if not angle_nb.equals(angle_conflict):
                raise ValueError(
                    "Legacy Angle identifiers are not duplicates; refusing to infer "
                    "a -45-degree run"
                )
            out = out[out["variant"] != "angle_conflict"].copy()

    out.loc[
        out["variant"] == "angle_net_benefit", "variant"
    ] = "angle_net_benefit_legacy"
    out["score_convention"] = SCORE_CONVENTION
    return out, True


results, normalised_legacy = normalise_saved_policy_ids(results)
if normalised_legacy:
    print(
        "Normalised legacy scalar identifiers; duplicate +45-degree "
        "angle_conflict rows omitted (no -45-degree result inferred)."
    )

required = {
    "variant",
    "run",
    "group",
    "n",
    "isch_n",
    "bleed_n",
}

missing = required - set(results.columns)

if missing:
    raise ValueError(
        f"Missing columns in {RESULTS_FILE}: {sorted(missing)}"
    )

rows = []

for (variant, run), df_run in results.groupby(["variant", "run"]):

    included = df_run[df_run["group"] == "included"]

    discarded = df_run[df_run["group"] == "discarded"]

    if included.empty or discarded.empty:
        continue

    included_n = int(included["n"].iloc[0])
    discarded_n = int(discarded["n"].iloc[0])

    for endpoint in ENDPOINTS:

        event_col = f"{endpoint}_n"

        included_events = int(included[event_col].iloc[0])
        discarded_events = int(discarded[event_col].iloc[0])

        included_nonevents = included_n - included_events
        discarded_nonevents = discarded_n - discarded_events

        table = [
            [included_events, included_nonevents],
            [discarded_events, discarded_nonevents],
        ]

        odds_ratio, p_value = fisher_exact(
            table,
            alternative="two-sided",
        )

        rows.append({
            "variant": variant,
            "score_convention": SCORE_CONVENTION,
            "run": int(run),
            "endpoint": endpoint,
            "included_n": included_n,
            "included_events": included_events,
            "included_rate": included_events / included_n,
            "discarded_n": discarded_n,
            "discarded_events": discarded_events,
            "discarded_rate": discarded_events / discarded_n,
            "odds_ratio": odds_ratio,
            "p_included_vs_discarded": p_value,
        })

stats_df = pd.DataFrame(rows)

stats_df.to_parquet(
    OUT_STATS,
    index=False,
)
stats_df.to_parquet(
    os.path.join(RUN_DIR, os.path.basename(OUT_STATS)),
    index=False,
)

print(f"Saved statistics: {OUT_STATS}")

summary_df = (
    stats_df
    .groupby(["variant", "endpoint"])
    .agg(
        included_rate_mean=("included_rate", "mean"),
        included_rate_std=("included_rate", "std"),
        discarded_rate_mean=("discarded_rate", "mean"),
        discarded_rate_std=("discarded_rate", "std"),
        odds_ratio_mean=("odds_ratio", "mean"),
        p_median=("p_included_vs_discarded", "median"),
        p_min=("p_included_vs_discarded", "min"),
        p_max=("p_included_vs_discarded", "max"),
    )
    .reset_index()
)

print("\n================ SUMMARY ================\n")
print(summary_df.to_string(index=False))


def format_rate(mean, std):
    return (
        f"{100 * mean:.2f} $\\pm$ "
        f"{100 * std:.2f}\\%"
    )


def format_p(p):
    if p < 0.001:
        return r"$<0.001$"
    return f"${p:.3f}$"


latex_rows = []

VARIANT_LABELS = {
    "conflict": r"Conflict ($b-i$)",
    "net_benefit": r"Net-benefit / win-win ($b+i$)",
    "angle_net_benefit": r"Angle net-benefit ($+45^\circ$)",
    "angle_net_benefit_legacy": r"Angle net-benefit (legacy $+45^\circ$)",
    "angle_conflict": r"Angle conflict ($-45^\circ$)",
    "minus_isch": r"$-$isch",
    "isch": r"$+$isch",
    "random": "Random",
}

for _, row in summary_df.iterrows():

    latex_rows.append(
        f"{VARIANT_LABELS.get(row['variant'], row['variant'])} & "
        f"{row['endpoint']} & "
        f"{format_rate(row['included_rate_mean'], row['included_rate_std'])} & "
        f"{format_rate(row['discarded_rate_mean'], row['discarded_rate_std'])} & "
        f"{format_p(row['p_median'])} \\\\"
    )


latex_table = "\n".join([
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{6pt}",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Variant & Endpoint & Included & Discarded & "
    r"$p$-value \\",
    r"\midrule",
    *latex_rows,
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Event rates for included and discarded patients across the repeated runs of Mechanism~2. Conflict denotes $b-i$ and net-benefit/win-win denotes $b+i$. The checked-in Angle estimate is the legacy $+45^\circ$ rule; no $-45^\circ$ Angle-conflict estimate is inferred from it. Values are mean $\pm$ standard deviation across runs. The $p$-value is the median two-sided Fisher exact test comparing included and discarded patients within each run.}",
    r"\label{tab:mechanism2_event_rates}",
    r"\end{table}",
])

with open(
    OUT_TEX,
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_table)
with open(
    os.path.join(RUN_DIR, os.path.basename(OUT_TEX)),
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_table)

print(f"Saved LaTeX table: {OUT_TEX}")
print(f"Archived copy: {RUN_DIR}")