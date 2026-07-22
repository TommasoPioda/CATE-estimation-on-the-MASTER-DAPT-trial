"""Is the selection real, or is it noise?

nb 01 fits ONE causal forest, reads each patient's position on the bleeding-vs-ischaemic
plane, and selects on it. This script asks whether that position survives a refit: it
re-runs the whole pipeline `--n-seeds` times on **bootstrap-resampled** cohorts and records
how often each patient is selected.

The knockout bracket is frozen across seeds (see `make_pairs`), so the only thing moving is
the model. Three arms, all scored on that same bracket:

  real     bootstrap-resample the cohort, real treatment -> how reproducible is the selection?
  placebo  same, but the treatment is PERMUTED first, so the true effect is exactly 0 and
           constant -> the reproducibility you get from estimation noise alone.
  random   flip a coin per pair, no model at all -> the absolute chance floor.

Read `decisive`: with a frozen bracket a patient is picked every time (freq 1) or never
(freq 0) if the model really knows something about them, and about half the time (freq 0.5)
if it does not. `random` therefore sits near 0 by construction. If `real` lands on `placebo`
and `random`, a patient's selection is noise: they are picked because this particular
resample nudged them across a boundary, not because the model knows something about them.

  python selection_stability.py --n-seeds 100 --workers 12
  python selection_stability.py --rule topright          # the win-win corner instead

Caveat on `placebo`: permuting T destroys the bleeding ATE as well as the heterogeneity, so
it is a null for "any structure at all", not for heterogeneity alone. It is the floor, not a
like-for-like twin of `real`.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
CF_DIR = os.path.join(ROOT, 'Meta-learning', 'causal_forest')
MODEL_PATH = os.path.join(ROOT, 'Meta-learning', 'models', 'CausalForest',
                          'CausalForest_multioutput_tuned.joblib')
OUT_DIR = os.path.join(ROOT, 'Meta-learning', 'models', 'CausalForest', 'STABILITY')
sys.path.insert(0, CF_DIR)                       # so joblib can unpickle the pipeline

# nb 06's training order / nb 01's endpoints.
CF_TARGETS = ['barc_235', 'death', 'mi', 'stroke', 'bleed']
CF_TARGET_COLS = ['cec_barc235_335d', 'cec_cvdeath_335d', 'cec_mi_335d',
                  'cec_stroke_335d', 'cec_bleed_335d']
ISCH = ['death', 'mi', 'stroke']
FIT_TARGETS = [CF_TARGETS.index(k) for k in ISCH + ['bleed']]   # skip barc_235: nb 01 never reads it


def z(v):
    return (v - v.mean()) / v.std()


def load_cohort():
    """nb 01's cohort, verbatim: the same seed-42 shuffle and the same treatment coding."""
    X = pd.read_parquet(os.path.join(ROOT, 'data', 'X_features.parquet'))
    y = pd.read_parquet(os.path.join(ROOT, 'data', 'y_targets.parquet'))
    perm = X.sample(frac=1, random_state=42).index
    X = X.loc[perm].reset_index(drop=True)
    y = y.loc[perm].reset_index(drop=True)
    Xn = X.select_dtypes(include=[np.number])
    T = X['regimen'].map({'prolonged DAPT': 1, 'abbreviated DAPT': 0}).to_numpy()
    return Xn, y[CF_TARGET_COLS], T


def cate_by_endpoint(pipe, X):
    """CATE per endpoint, skipping the forests `targets=` left unfitted.

    The pipeline's own `predict_cate` walks every model, so it trips over the `None` that
    `targets=FIT_TARGETS` leaves at barc_235. Preprocessing is the pipeline's own, so the
    numbers are identical to what `predict_cate` would return for the fitted columns.
    """
    X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
    X_scaled = pipe.scaler.transform(pipe.imputer_x.transform(X_df))
    return {CF_TARGETS[j]: pipe.models[j].effect(X_scaled) for j in FIT_TARGETS}


def plane(C):
    """nb 01's plane: x = raw bleeding benefit, y = z-scored mean ischaemic benefit.

    The y-axis is z-scored exactly as nb 01 does (the endpoints live on different scales, so
    a raw mean would be whichever endpoint happens to be largest); x stays raw, as in nb 01.
    """
    x = C['bleed']
    y = (z(C['death']) + z(C['mi']) + z(C['stroke'])) / 3
    return x, y


def duel_by_angle(i, j, x, y):
    """nb 01's rule, vectorised over pair arrays `i`/`j`. Same tie-breaks: `i` wins only on
    a strict `>`, so ties go to `j`, exactly like the notebook's `return i if a1 > a2 else j`."""
    a = np.degrees(np.arctan2(y, x))
    both_q1 = (x[i] > 0) & (y[i] > 0) & (x[j] > 0) & (y[j] > 0)
    win_i = np.where(both_q1, x[i] > x[j], a[i] > a[j])
    return np.where(win_i, i, j)


def duel_by_conflict(i, j, x, y):
    c = z(x) + y                       # nb 01's conflict score (y is already the z-scored mean)
    return np.where(c[i] > c[j], i, j)


def make_pairs(n, seed=0):
    """The knockout bracket, drawn ONCE and reused by every seed and every arm.

    nb 01 redraws the pairing each run, which mixes two very different sources of variation:
    who you happen to be matched against, and what the model thinks of you. Only the second
    is the question here, so the bracket is frozen: a patient's selection frequency then reads
    as "how often does a refit still rank me above the SAME opponent". With the bracket free,
    even a perfect model would drop below 1.0 whenever a patient draws a stronger rival, and
    the noise would look like signal.

    The cohort is odd (4579), so the last patient is left unpaired -- the same one nb 01's
    `zip(cand[::2], cand[1::2])` silently drops. Truncating to `n // 2` pairs keeps that
    behaviour instead of letting the two halves differ in length.
    """
    cand = np.random.default_rng(seed).permutation(n)
    m = 2 * (n // 2)
    return cand[:m:2], cand[1:m:2]


def select(rule, x, y, pairs):
    """Indices the model's rule selects: one winner per bracket pair (top half for topright)."""
    if rule == 'topright':
        return np.argsort(-(z(x) + z(y)))[:len(x) // 2]
    i, j = pairs
    duel = duel_by_angle if rule == 'angle' else duel_by_conflict
    return duel(i, j, x, y)


def one_seed(args):
    """One resample -> one refit -> one selection. Returns (selected idx, mean CATE per endpoint)."""
    seed, arm, rule, n_jobs, cf_params, nuisance_params, pairs = args
    from casual_multioutput_pipeline import CausalMultiOutputPipeline

    Xn, y, T = load_cohort()
    rng = np.random.default_rng(seed)
    n = len(T)

    if arm == 'random':                                   # no model at all: the chance floor
        if rule == 'topright':
            return rng.permutation(n)[:n // 2], {}
        i, j = pairs
        return np.where(rng.random(len(i)) < 0.5, i, j), {}

    if arm == 'placebo':
        T = rng.permutation(T)                            # true effect now exactly 0 and constant

    boot = rng.integers(0, n, size=n)                     # bootstrap resample of the cohort
    pipe = CausalMultiOutputPipeline(cf_params=cf_params, nuisance_params=nuisance_params,
                                     n_jobs=n_jobs)
    pipe.fit(Xn.values[boot], y.values[boot], T[boot], targets=FIT_TARGETS)

    C = cate_by_endpoint(pipe, Xn.values)                 # score EVERY patient, as nb 01 does
    x, y_ax = plane(C)
    sel = select(rule, x, y_ax, pairs)
    return sel, {k: float(v.mean()) for k, v in C.items()}


def run_arm(arm, seeds, rule, workers, n_jobs, cf_params, nuisance_params, n, pairs):
    """Run one arm across seeds; return (freq per patient, list of selected sets, mean-CATE frame)."""
    jobs = [(s, arm, rule, n_jobs, cf_params, nuisance_params, pairs) for s in seeds]
    counts = np.zeros(n)
    sets, means = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (sel, mean_cate) in enumerate(ex.map(one_seed, jobs), 1):
            counts[np.unique(sel)] += 1
            sets.append(set(np.unique(sel).tolist()))
            if mean_cate:
                means.append(mean_cate)
            print(f'  [{arm}] seed {k}/{len(seeds)}', flush=True)
    return counts / len(seeds), sets, pd.DataFrame(means)


def jaccard(sets, rng, n_pairs=200):
    """Mean Jaccard overlap between the selected sets of two different seeds."""
    if len(sets) < 2:
        return np.nan
    out = []
    for _ in range(n_pairs):
        a, b = rng.choice(len(sets), size=2, replace=False)
        A, B = sets[a], sets[b]
        out.append(len(A & B) / len(A | B))
    return float(np.mean(out))


def decisive(freq, lo=0.1, hi=0.9):
    """Fraction of patients the seeds agree on: always picked or never picked.

    Real signal makes this large (the frequency histogram is bimodal at 0 and 1). Noise
    makes it small: every patient hovers around the base rate, picked about half the time.
    """
    return float(np.mean((freq <= lo) | (freq >= hi)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-seeds', type=int, default=100)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--rule', choices=['angle', 'conflict', 'topright'], default='angle',
                   help="selection rule from nb 01 (default: angle = duel_by_angle)")
    p.add_argument('--arms', default='real,placebo,random')
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    n_jobs = max(1, (os.cpu_count() or 16) // args.workers)

    # Reuse nb 06's Optuna-tuned config so the refits are the model nb 01 actually selects with.
    tuned = joblib.load(MODEL_PATH)
    cf_params, nuisance_params = dict(tuned.cf_params), dict(tuned.nuisance_params)

    Xn, y, T = load_cohort()
    n = len(T)
    print(f'cohort {n} | rule={args.rule} | seeds={args.n_seeds} | '
          f'workers={args.workers} x n_jobs={n_jobs}')

    # Reference plane: the full-cohort fit, real treatment. The scatter's axes are FIXED to
    # this one model so the selection frequency is read against a stable position -- plotting
    # each seed's own CATE would just be the model re-reading its own scores.
    print('reference fit (full cohort, real T)...', flush=True)
    ref = joblib.load(MODEL_PATH)
    ref.fit(Xn.values, y.values, T, targets=FIT_TARGETS)
    x_ref, y_ref = plane(cate_by_endpoint(ref, Xn.values))

    pairs = make_pairs(n)                 # one frozen bracket for every seed and every arm
    seeds = list(range(args.n_seeds))
    rng = np.random.default_rng(0)
    freqs, report = {}, []
    for arm in args.arms.split(','):
        print(f'arm: {arm}', flush=True)
        freq, sets, means = run_arm(arm, seeds, args.rule, args.workers, n_jobs,
                                    cf_params, nuisance_params, n, pairs)
        freqs[arm] = freq
        report.append({'arm': arm, 'jaccard': jaccard(sets, rng),
                       'decisive': decisive(freq), 'mean_freq': float(freq.mean())})
        if not means.empty:
            print(f'  mean CATE across seeds:\n{means.agg(["mean", "std", "min", "max"]).round(5)}')
            flips = {c: float((np.sign(means[c]) != np.sign(means[c].mean())).mean())
                     for c in means.columns}
            print(f'  fraction of seeds whose MEAN CATE flips sign: {flips}')

    rep = pd.DataFrame(report).set_index('arm')
    print('\n' + '=' * 72)
    print('jaccard  = overlap of the selected sets between two seeds (1.0 = identical)')
    print('decisive = fraction of patients always (>=90%) or never (<=10%) selected')
    print('=' * 72)
    print(rep.round(3))
    if {'real', 'placebo'} <= set(freqs):
        print(f"\nreal vs placebo -- jaccard {rep.loc['real', 'jaccard']:.3f} vs "
              f"{rep.loc['placebo', 'jaccard']:.3f}, decisive "
              f"{rep.loc['real', 'decisive']:.3f} vs {rep.loc['placebo', 'decisive']:.3f}")
        print('If these two lines are close, the selection is noise: a provably constant')
        print('effect reproduces the same selection structure as the real data.')

    pd.DataFrame({'bleed_ref': x_ref, 'isch_ref': y_ref,
                  **{f'freq_{a}': f for a, f in freqs.items()}}).to_csv(
        os.path.join(OUT_DIR, f'selection_frequency_{args.rule}.csv'), index=False)
    rep.to_csv(os.path.join(OUT_DIR, f'stability_report_{args.rule}.csv'))

    # --- the plot: the reference plane, coloured by how often the seeds picked each patient
    fig, ax = plt.subplots(figsize=(7.4, 6))
    sc = ax.scatter(x_ref, y_ref, c=freqs.get('real', list(freqs.values())[0]),
                    cmap='inferno', s=14, vmin=0, vmax=1, linewidths=0)
    ax.axhline(0, color='grey', lw=.8); ax.axvline(0, color='grey', lw=.8)
    ax.set(xlabel='bleeding benefit (shortening)',
           ylabel='mean ischaemic benefit (shortening)',
           title=f'Selection frequency over {args.n_seeds} refits  (brighter = picked more often)')
    fig.colorbar(sc, ax=ax, label='fraction of seeds that selected the patient')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'SELECTION_FREQUENCY_{args.rule}.png'), dpi=150)

    # --- the test: real vs the two floors. Signal is bimodal at 0/1; noise piles up at 0.5.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for arm, f in freqs.items():
        ax.hist(f, bins=40, range=(0, 1), histtype='step', lw=2, label=arm, density=True)
    ax.set(xlabel='fraction of seeds that selected the patient', ylabel='density',
           title='Selection frequency: real vs placebo (effect=0) vs random')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'STABILITY_{args.rule}.png'), dpi=150)
    print(f'\nfigures + csv -> {OUT_DIR}')


if __name__ == '__main__':
    main()
