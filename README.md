# CATE estimation and use in MASTER DAPT study

Research code, analyses, and thesis sources for the 2026 Bachelor thesis by
**Tommaso Pioda** (Data Science and Artificial Intelligence, SUPSI DTI),
developed in collaboration with Ente ospedaliero cantonale (EOC).

This repository investigates whether the duration of dual antiplatelet therapy
(DAPT) can be personalised for high-bleeding-risk patients in the MASTER DAPT
randomised trial. It estimates patient-level conditional treatment effects and
tests whether they can support guided enrolment in subsequent simulations.

> **Research use only — not clinical software.** This repository is a
> research artefact, not a medical device or clinical decision-support tool.
> Its code, models, and results must not be used to make treatment decisions
> for individual patients.

## Contents

- [Project scope and findings](#project-scope-and-findings)
- [Methods](#methods)
- [Repository map](#repository-map)
- [Data access and confidentiality](#data-access-and-confidentiality)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Building the thesis](#building-the-thesis)
- [Limitations](#limitations)
- [Citation](#citation)
- [Contributing, support, and reuse](#contributing-support-and-reuse)

## Project scope and findings

### Research question

Can heterogeneous treatment effects identify patients for whom abbreviated or
prolonged DAPT is preferable, and can those effects be exploited by a guided
enrolment policy?

### Study at a glance

| Item | Description |
| --- | --- |
| Population | 4,579 high-bleeding-risk patients from MASTER DAPT. |
| Comparison | Abbreviated versus prolonged DAPT. |
| Outcomes | Bleeding, cardiovascular death, myocardial infarction, stroke, revascularisation, and derived composite outcomes. |
| Main analyses | Predictive modelling, CATE estimation and validation, treatment-policy evaluation, and guided-enrolment simulations. |

### Main findings

Across the five CATE estimators, the analyses reproduce the trial's average
bleeding benefit for abbreviated DAPT but find no patient-level heterogeneity
beyond noise. Guided-enrolment policies therefore differ from random selection
without demonstrating a clinical benefit over assigning abbreviated treatment
to every patient.

These findings describe this dataset, endpoint definition, and analysis
pipeline; they are not a recommendation for clinical practice.

## Methods

For a binary endpoint, the project uses the following sign convention:

```text
CATE_i = P(Y=1 | prolonged DAPT, X_i) - P(Y=1 | abbreviated DAPT, X_i)
```

A negative value means that prolonged DAPT has a lower estimated risk for that
endpoint and covariate profile.

The repository contains the following analysis components:

1. Data inspection, cleaning, and exploratory analysis.
2. Baseline outcome-prediction models for the two DAPT regimens.
3. CATE estimation with a T-learner, causal forest, Bayesian causal forest
   (BCF), interaction forest, and CausalPFN.
4. Heterogeneity validation, bleeding/ischaemic trade-off analyses, and policy
   evaluation.
5. Online guided-enrolment simulations, including angular, UCB, and Thompson
   sampling policies.

## Repository map

| Path | Purpose | Suggested starting point |
| --- | --- | --- |
| [`data_cleaning/`](data_cleaning) | Raw-data inspection and cleaning notebooks. | `00_raw_data_info.ipynb`, then `01_data_cleaning.ipynb` |
| [`data_exploration/`](data_exploration) | Outcome, feature, PCA, outlier, and interaction exploration. | `01_target_exploration.ipynb` |
| [`data/`](data) | Feature schema and metadata; local Parquet inputs are intentionally not versioned. | [`dataset_metadata.json`](data/dataset_metadata.json) |
| [`simple_ml_models/`](simple_ml_models) | Baseline predictive models. | `bi-class/01_grouped_ML.ipynb` |
| [`Meta-learning/`](Meta-learning) | CATE estimators, model artefacts, table-generation scripts, and trade-off analyses. | [`CHAPTER6_TABLES.md`](Meta-learning/CHAPTER6_TABLES.md) |
| [`online-learning/`](online-learning) | Guided-enrolment policies, simulations, figures, and tests. | [`CHAPTER8_TABLES.md`](online-learning/CHAPTER8_TABLES.md) |
| [`src/thesis_utils/`](src/thesis_utils) | Shared Python package for pipelines, predictions, and plots. | [`pipeline_utils.py`](src/thesis_utils/pipeline_utils.py) |
| [`markdown_docs/thesis/`](markdown_docs/thesis) | Thesis source, bibliography, figures, and generated table fragments. | [`Thesis.tex`](markdown_docs/thesis/Thesis.tex) |

## Data access and confidentiality

Clinical data and experiment-result Parquet files are deliberately excluded
from version control (`*.parquet` is ignored by Git). They must never be
committed, attached to an issue, or shared through a fork.

To reproduce analyses that use patient data, obtain the required approval from
the data custodians and place the authorised files at:

```text
data/X_features.parquet
data/y_targets.parquet
```

[`data/dataset_metadata.json`](data/dataset_metadata.json) documents the
expected input schema: 4,579 observations, 64 covariates, and 9 endpoints.
Imputation and scaling are fitted within modelling pipelines and validation
folds, rather than globally on the input data.

To regenerate online-learning tables, the per-replication results must also be
available locally in `online-learning/results/`. The expected
`results_*.parquet` files are documented in
[`online-learning/CHAPTER8_TABLES.md`](online-learning/CHAPTER8_TABLES.md).

## Installation

### Requirements

- Python 3.10 or later
- `pip` and a virtual-environment tool
- For thesis compilation only: XeLaTeX and `latexmk`

The repository provides a local `thesis_utils` package but does not yet ship a
lockfile or a fully pinned environment. Use the following as a reproducible
starting point, then record the exact versions used for any new experiment.

```bash
python --version  # must report 3.10 or later
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy pandas scipy scikit-learn matplotlib joblib pyarrow pytest
```

The complete CATE experiments use additional optional packages:

```bash
python -m pip install econml lightgbm optuna tqdm stochtree causalpfn
```

`stochtree` and `causalpfn` may have platform-specific availability or
compatibility constraints. The base environment is enough for the policy test
suite; the controlled clinical data is still required for data-dependent
analyses.

## Reproducibility

Choose the smallest reproduction target that answers your question.

| Target | Required inputs | Command |
| --- | --- | --- |
| Verify online-policy logic | Base Python environment; no clinical data. | `python -m pytest online-learning/tests` |
| Regenerate Chapter 6 tables | Authorised inputs and local model artefacts. | `python Meta-learning/scripts/extract_chapter6_results.py` then `python Meta-learning/scripts/generate_chapter6_tables.py` |
| Regenerate Chapter 8 tables | Saved replication results in `online-learning/results/`. | `python online-learning/scripts/generate_chapter8_tables.py` |

### Verify generated tables without overwriting them

```bash
python Meta-learning/scripts/extract_chapter6_results.py --check
python Meta-learning/scripts/generate_chapter6_tables.py --check
python online-learning/scripts/generate_chapter8_tables.py --check
```

The Chapter 6 extraction can take several minutes because it loads large BCF
artefacts. The table-generation scripts recompute table inputs and audit
trails; they do not retrain models.

For assumptions, source files, replication counts, and statistical formulas,
consult the dedicated documentation:

- [`Meta-learning/CHAPTER6_TABLES.md`](Meta-learning/CHAPTER6_TABLES.md)
- [`online-learning/CHAPTER8_TABLES.md`](online-learning/CHAPTER8_TABLES.md)

### End-to-end workflow

With authorised data, follow the repository in this order:

1. Run the notebooks under `data_cleaning/` and `data_exploration/` to inspect
   the inputs and their schema.
2. Train or inspect the baseline models in `simple_ml_models/`.
3. Run the estimator-specific notebooks and scripts in `Meta-learning/`.
4. Run the online-learning experiments only after the required model and data
   artefacts are available.
5. Regenerate and check the thesis tables before compiling the thesis.

Long-running training and simulation scripts may write model artefacts and
Parquet results. Use a separate, access-controlled local workspace for these
outputs.

## Building the thesis

The complete thesis is defined in
[`markdown_docs/thesis/Thesis.tex`](markdown_docs/thesis/Thesis.tex). With a
TeX distribution that includes XeLaTeX and `latexmk`:

```bash
cd markdown_docs/thesis
latexmk -xelatex Thesis.tex
```

[`build_pdf.sh`](markdown_docs/thesis/build_pdf.sh) serves a different purpose:
it converts the Markdown outlines to PDF using Pandoc and XeLaTeX.

## Limitations

- The repository does not distribute the clinical inputs or all experiment
  outputs, so a public clone cannot reproduce data-dependent results by
  itself.
- There is no locked dependency environment. Exact version capture is required
  for a new computationally reproducible run.
- The models evaluate hypotheses in MASTER DAPT; they have not been validated
  as deployable clinical prediction or treatment-assignment tools.
- The negative heterogeneity result must not be interpreted as evidence that
  no heterogeneity can exist in other populations, endpoint definitions, or
  study designs.

## Citation

If you use this repository or its methodology, please cite the accompanying
thesis:

```bibtex
@thesis{pioda2026cate,
  author = {Pioda, Tommaso},
  title = {CATE estimation and use in MASTER DAPT study},
  type = {Bachelor thesis},
  institution = {SUPSI DTI},
  year = {2026}
}
```

No DOI or `CITATION.cff` file is currently provided. Add an official,
maintainer-approved citation record before creating a release or archive.

## Contributing, support, and reuse

This is a thesis research repository. No formal contribution process is
currently specified. Before proposing changes, contact the author or the
relevant institutional supervisor through the established institutional
channels. Never include clinical data, credentials, or derived patient-level
outputs in a contribution.

No licence file is currently present in this repository. Do not assume that
the code, model artefacts, figures, or thesis material may be reused,
redistributed, or relicensed. Obtain explicit permission from the relevant
rights holders before reuse. The MASTER DAPT clinical data remains subject to
its own access controls and approvals.

**Author:** Tommaso Pioda<br>
**Supervisor:** Gabriele Maroni · **Co-supervisor:** Laura Azzimonti<br>
**Commissioning organisation:** Ente ospedaliero cantonale (EOC)
