"""Expanded recovery time courses, distinct from epsilon-versus-endpoint plots.

The input is the merged seed-level recovery summary, not individual directions.
No values or intervals are recomputed here. Failed/undefined entries stay gaps.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

DATASETS = [('mnist', 'MNIST'), ('fashion_mnist', 'Fashion-MNIST'), ('kmnist', 'KMNIST')]
CHECKPOINTS = [2, 10, 20]
EPSILONS = [0.01, 0.05, 0.10, 0.20]
COLORS = {0.01: '#2878B5', 0.05: '#E29424', 0.10: '#49A078', 0.20: '#C74753'}
PROTOCOLS = [('preserve', '-', 'Momentum fortgesetzt'),
             ('reset', (0, (5, 2.5)), 'Momentum zurückgesetzt')]


def _validate_and_select(curves_df):
    required = {'dataset', 'checkpoint', 'layer', 'momentum', 'epsilon', 'update'}
    required |= {f'{metric}_{stat}' for metric in ('ratio', 'distance', 'accuracy', 'loss')
                 for stat in ('mean', 'low', 'high')}
    missing = sorted(required-set(curves_df.columns))
    if missing:
        raise ValueError(f'Missing expanded recovery summary columns: {missing}')
    table = curves_df[(curves_df['layer'] == 'both') & (curves_df['epsilon'] > 0)].copy()
    if 'order' in table:
        table = table[table['order'] == 'same']
    if 'phase' in table:
        table = table[table['phase'] == 'expanded']
    key = ['dataset', 'checkpoint', 'momentum', 'epsilon', 'update']
    if table.duplicated(key).any():
        raise ValueError('Expanded summary contains duplicate dataset/checkpoint/protocol/epsilon/time rows')
    for dataset, _ in DATASETS:
        for checkpoint in CHECKPOINTS:
            cell = table[(table.dataset == dataset) & (table.checkpoint == checkpoint)]
            for protocol, _, _ in PROTOCOLS:
                for epsilon in EPSILONS:
                    line = cell[(cell.momentum == protocol) & np.isclose(cell.epsilon, epsilon)]
                    if line.empty:
                        raise ValueError(f'Incomplete expanded matrix: {dataset}, E{checkpoint}, {protocol}, epsilon={epsilon}')
                    if line['update'].min() != 0:
                        raise ValueError('Recovery time courses must include the immediate post-perturbation state')
    return table.sort_values(key)


def _draw(table, root, metric, stem, title, ylabel, log_y=False):
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'grid.linewidth': .7,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
    }):
        fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
        for row, (dataset, display_name) in enumerate(DATASETS):
            for col, checkpoint in enumerate(CHECKPOINTS):
                ax = axes[row, col]
                cell = table[(table.dataset == dataset) & (table.checkpoint == checkpoint)]
                for epsilon in EPSILONS:
                    for protocol, linestyle, _ in PROTOCOLS:
                        line = cell[(cell.momentum == protocol) & np.isclose(cell.epsilon, epsilon)].sort_values('update')
                        x = line['update'].to_numpy(dtype=float)/235.
                        mean = line[f'{metric}_mean'].to_numpy(dtype=float)
                        low = line[f'{metric}_low'].to_numpy(dtype=float)
                        high = line[f'{metric}_high'].to_numpy(dtype=float)
                        valid = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
                        if log_y:
                            valid &= (mean > 0) & (low > 0) & (high > 0)
                        # Do not interpolate across undefined/failed measurements.
                        shown_mean = np.where(valid, mean, np.nan)
                        ax.plot(x, shown_mean, color=COLORS[epsilon], ls=linestyle,
                                lw=1.8 if protocol == 'preserve' else 1.4,
                                zorder=3 if protocol == 'preserve' else 4)
                        ax.fill_between(x, low, high, where=valid, color=COLORS[epsilon],
                                        alpha=.08 if protocol == 'preserve' else .045,
                                        linewidth=0, interpolate=False, zorder=1)
                if metric == 'ratio':
                    ax.axhline(1., color='#454545', ls=':', lw=.9, zorder=2)
                if log_y:
                    ax.set_yscale('log')
                ax.set_xscale('symlog', linthresh=.2)
                ax.set_xticks([0, .1, 1, 2, 5, 10])
                ax.set_xticklabels(['0', '0,1', '1', '2', '5', '10'])
                ax.set_xlim(0, 10)
                epoch_word='Trainingsepoche' if checkpoint==2 else 'Trainingsepochen'
                ax.set_title(f'{display_name} · E{checkpoint} ({checkpoint/2:g} {epoch_word})', pad=11)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 2:
                    ax.set_xlabel('Weitere Epochen')
        if not log_y:
            axes[0, 0].set_ylim(bottom=0)
        handles = [Line2D([0], [0], color=COLORS[epsilon], lw=2.3, label=f'ε = {epsilon:.2f}')
                   for epsilon in EPSILONS]
        handles += [Line2D([0], [0], color='#333333', lw=1.8, ls=style, label=label)
                    for _, style, label in PROTOCOLS]
        fig.suptitle(title, fontsize=18, y=.99)
        fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False,
                   bbox_to_anchor=(.5, .96), fontsize=10)
        fig.text(.5, .028,
                 'Beide Gewichtsmatrizen gestört · zehn Ausgangsnetze, eine Richtung je Netz und Zelle · '
                 'Bänder: punktweise 95%-Bootstrapintervalle über Seeds',
                 ha='center', fontsize=10, color='#555555')
        fig.text(.5, .008,
                 'Zeitachse im frühen Verlauf gestreckt; ab 0,2 Epochen logarithmisch. '
                 'Alle Panels verwenden dieselbe y-Skala. Intervalle ohne Korrektur für multiple Vergleiche.',
                 ha='center', fontsize=10, color='#555555')
        fig.tight_layout(rect=(.015, .052, .99, .925), h_pad=2.5, w_pad=1.7)
        for extension in ('png', 'pdf'):
            fig.savefig(root/f'{stem}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)


def plot(root, curves_df):
    """Write the complete nine-cell expanded recovery ratio and distance figures.

    All datasets/checkpoints/protocols/epsilon cells are required. The CSV's
    seed-level means and confidence intervals are plotted directly, preserving
    undefined values as missing rather than silently dropping source seeds.
    """
    root = Path(root)
    table = _validate_and_select(curves_df)
    _draw(table, root, 'ratio', 'expanded_recovery_trajectories',
          'Erweiterte Untersuchung · Abstand zur Kontrolltrajektorie während der Recovery',
          r'Relativer Abstand $r(s)=d_Q(s)/d_Q(0^+)$', log_y=True)
    _draw(table, root, 'distance', 'expanded_absolute_trajectories',
          'Erweiterte Untersuchung · Absoluter Gewichtsabstand während der Recovery',
          r'Kanonischer Quotientenabstand $d_Q$ (Frobenius)')
    return {'figures': [f'{stem}.{extension}' for stem in
                        ('expanded_recovery_trajectories', 'expanded_absolute_trajectories')
                        for extension in ('png', 'pdf')],
            'cells': 9, 'lines_per_cell': 8, 'scope': 'both weight matrices; all three datasets and checkpoints'}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    folder = Path(args.root)
    data = pd.read_csv(folder/'expanded_recovery_summary.csv')
    print(json.dumps(plot(folder, data), ensure_ascii=False, indent=2))
