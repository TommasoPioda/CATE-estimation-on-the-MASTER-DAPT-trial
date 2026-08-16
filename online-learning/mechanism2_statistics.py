import os
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS_FILE = os.path.join(
    HERE,
    "results_mechanism2_sample_select.parquet",
)

OUT_STATS = os.path.join(
    HERE,
    "mechanism2_statistics.parquet",
)

OUT_TEX = os.path.join(
    HERE,
    "mechanism2_statistics.tex",
)

ENDPOINTS = {
    "isch": "isch",
    "bleed": "bleed",
}

results = pd.read_parquet(RESULTS_FILE)

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

for _, row in summary_df.iterrows():

    latex_rows.append(
        f"{row['variant']} & "
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
    r"\caption{Event rates for included and discarded patients across the repeated runs of Mechanism~2. Values are mean $\pm$ standard deviation across runs. The $p$-value is the median two-sided Fisher exact test comparing included and discarded patients within each run.}",
    r"\label{tab:mechanism2_event_rates}",
    r"\end{table}",
])

with open(
    OUT_TEX,
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_table)

print(f"Saved LaTeX table: {OUT_TEX}")