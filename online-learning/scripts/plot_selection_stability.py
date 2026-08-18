"""Deeper look at selection_stability.py's output: WHERE does the signal live?

selection_stability.py answers "is the selection noise" with two summary numbers
(jaccard, decisive) and two plots. This script reads its CSV output and asks a
sharper, spatial question: does `real` beat the `placebo`/`random` noise floors
uniformly, or only far from the +45 deg decision boundary duel_by_angle selects
on -- and for which patients does `real` disagree with `placebo` specifically?

Needs selection_frequency_{rule}.csv + stability_report_{rule}.csv, already
written by selection_stability.py into Meta-learning/models/CausalForest/STABILITY/.

    python online-learning/scripts/plot_selection_stability.py --rule angle
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
STAB_DIR = os.path.join(ROOT, 'Meta-learning', 'models', 'CausalForest', 'STABILITY')

# Fixed hue per arm, reused across every plot below -- colorblind-safe qualitative triplet.
ARM_COLOR = {'real': '#1b7837', 'placebo': '#b35806', 'random': '#7570b3'}
DECISIVE_LO, DECISIVE_HI = 0.1, 0.9        # same thresholds as selection_stability.decisive()


def load(rule):
    freq = pd.read_csv(os.path.join(STAB_DIR, f'selection_frequency_{rule}.csv'))
    rep = pd.read_csv(os.path.join(STAB_DIR, f'stability_report_{rule}.csv'), index_col='arm')
    arms = [c[len('freq_'):] for c in freq.columns if c.startswith('freq_')]
    return freq, rep, arms


def angle_distance(x, y):
    """Angular distance (deg) to the +45 win-win diagonal duel_by_angle selects on."""
    a = np.degrees(np.arctan2(y, x))
    return np.abs((a - 45 + 180) % 360 - 180)


def plot_ecdf(freq, rep, arms, rule):
    """Cleaner read than a histogram: no bin-width artifacts, and the decisive
    band (shaded) makes the real-vs-floor gap a horizontal distance you can see."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for arm in arms:
        v = np.sort(freq[f'freq_{arm}'])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post',
                color=ARM_COLOR.get(arm, 'grey'), lw=2,
                label=f"{arm}  (decisive={rep.loc[arm, 'decisive']:.2f}, "
                      f"jaccard={rep.loc[arm, 'jaccard']:.2f})")
    ax.axvspan(0, DECISIVE_LO, color='grey', alpha=.08)
    ax.axvspan(DECISIVE_HI, 1, color='grey', alpha=.08)
    ax.set(xlabel='fraction of seeds that selected the patient',
           ylabel='cumulative fraction of patients', xlim=(0, 1), ylim=(0, 1),
           title='Selection frequency, ECDF  (shaded = decisive band)')
    ax.legend(loc='center left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(STAB_DIR, f'ECDF_{rule}.png'), dpi=150)
    plt.close(fig)


def plot_boundary_diagnostic(freq, arms, rule, n_bins=10):
    """Decisiveness vs distance to the decision boundary, per arm.

    If the model reads real structure, patients far from the +45 diagonal should
    be picked (or not) reliably -- decisiveness should RISE with distance for
    `real`. If it's noise, decisiveness stays flat near 0 everywhere, same as
    `placebo`/`random`: a bootstrap refit or a coin flip doesn't care how far a
    patient sits from the boundary.
    """
    d = angle_distance(freq['bleed_ref'].to_numpy(), freq['isch_ref'].to_numpy())
    bins = pd.qcut(d, n_bins, duplicates='drop')
    mid = pd.Series(d).groupby(bins, observed=True).mean()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for arm in arms:
        decisiveness = (freq[f'freq_{arm}'] - 0.5).abs() * 2      # 0 = coin flip, 1 = fully decisive
        m = decisiveness.groupby(bins, observed=True).mean()
        se = decisiveness.groupby(bins, observed=True).sem()
        ax.errorbar(mid, m, yerr=se, color=ARM_COLOR.get(arm, 'grey'), lw=2,
                    marker='o', ms=4, label=arm)
    ax.set(xlabel='angular distance to the +45\N{DEGREE SIGN} boundary (deciles)',
           ylabel='decisiveness  (0 = coin flip, 1 = always/never picked)',
           title='Does decisiveness grow away from the decision boundary?')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(STAB_DIR, f'BOUNDARY_DIAGNOSTIC_{rule}.png'), dpi=150)
    plt.close(fig)


def plot_real_vs_placebo(freq, rule):
    """Per patient, freq_real vs freq_placebo. On the y=x line: that patient's
    real-arm decisiveness is fully explained by resampling noise. Off it: real
    disagrees with the noise floor for that specific patient."""
    if 'freq_placebo' not in freq.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(freq['freq_placebo'], freq['freq_real'], gridsize=25, cmap='inferno', mincnt=1)
    ax.plot([0, 1], [0, 1], color='white', lw=1, ls='--')
    ax.set(xlabel='freq_placebo (effect forced to 0)', ylabel='freq_real',
           xlim=(0, 1), ylim=(0, 1), aspect='equal',
           title='Per-patient: real vs the placebo noise floor')
    fig.colorbar(hb, ax=ax, label='patients')
    fig.tight_layout()
    fig.savefig(os.path.join(STAB_DIR, f'REAL_VS_PLACEBO_{rule}.png'), dpi=150)
    plt.close(fig)


def plot_decisive_map(freq, rule):
    """Bleed/ischaemic plane, categorised by WHERE `real` is decisive that
    `placebo` isn't -- the patients whose selection looks like signal rather
    than the shared noise floor."""
    if 'freq_placebo' not in freq.columns:
        return
    dec_real = (freq['freq_real'] <= DECISIVE_LO) | (freq['freq_real'] >= DECISIVE_HI)
    dec_plac = (freq['freq_placebo'] <= DECISIVE_LO) | (freq['freq_placebo'] >= DECISIVE_HI)
    cat = np.select(
        [dec_real & dec_plac, dec_real & ~dec_plac, ~dec_real & dec_plac, ~dec_real & ~dec_plac],
        ['decisive: both', 'decisive: real only', 'decisive: placebo only', 'decisive: neither'],
    )
    colors = {'decisive: real only': '#1b7837', 'decisive: both': '#7570b3',
             'decisive: placebo only': '#b35806', 'decisive: neither': '#cccccc'}

    fig, ax = plt.subplots(figsize=(7.4, 6))
    for label, color in colors.items():
        m = cat == label
        ax.scatter(freq['bleed_ref'][m], freq['isch_ref'][m], s=12, c=color, lw=0,
                   alpha=.35 if label == 'decisive: neither' else .75,
                   label=f'{label}  (n={int(m.sum())})')
    ax.axhline(0, color='grey', lw=.8); ax.axvline(0, color='grey', lw=.8)
    ax.set(xlabel='bleeding benefit (shortening)', ylabel='mean ischaemic benefit (shortening)',
           title='Where is the selection decisive beyond the placebo noise floor?')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(STAB_DIR, f'DECISIVE_MAP_{rule}.png'), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rule', default='angle', help="matches the --rule selection_stability.py was run with")
    args = p.parse_args()

    freq, rep, arms = load(args.rule)
    print(f'loaded {len(freq)} patients, arms={arms}')
    print(rep.round(3))

    plot_ecdf(freq, rep, arms, args.rule)
    plot_boundary_diagnostic(freq, arms, args.rule)
    plot_real_vs_placebo(freq, args.rule)
    plot_decisive_map(freq, args.rule)
    print(f'\nfigures -> {STAB_DIR}')


if __name__ == '__main__':
    main()
