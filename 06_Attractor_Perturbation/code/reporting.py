"""German reports and figures from measured perturbation experiment outputs.

No experiment values are embedded here. The report distinguishes membership,
retention, sampled re-entry, and strict initial-outside/final-inside returns.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

COLORS = {0.01: '#2878B5', 0.05: '#E29424', 0.10: '#49A078', 0.20: '#C74753'}
PROTOCOLS = {'preserve': 'A · Momentum fortgesetzt', 'reset': 'B · Momentum zurückgesetzt'}
BOOL_COLS = ('region_member', 'structural_member', 'functional_member', 'local_member',
             'initial_region_member', 'initially_outside_return', 'retained_region',
             'sampled_return', 'final_contracts', 'q90_member', 'subset_a_member', 'subset_b_member',
             'fixed_region_member', 'fixed_structural_member', 'fixed_functional_member',
             'initial_fixed_region_member', 'initial_fixed_structural_member', 'initial_fixed_functional_member',
             'fixed_return', 'fixed_structural_return', 'initial_baseline_fixed_member',
             'perturbation_induced_return', 'initial_local_member', 'local_return')


def _read_csv(path):
    table = pd.read_csv(path)
    for key in BOOL_COLS:
        if key in table and table[key].dtype == object:
            table[key] = table[key].map({'True': True, 'False': False, 'true': True, 'false': False,
                                          True: True, False: False, '1': True, '0': False})
    return table


def _fmt(value, digits=3, percent=False):
    if value is None or not np.isfinite(float(value)):
        return 'n. a.'
    return f'{100 * float(value):.{digits}f} %' if percent else f'{float(value):.{digits}f}'


def _estimate(row, key, digits=3, percent=False):
    return (f"{_fmt(row.get(key + '_mean'), digits, percent)} "
            f"[{_fmt(row.get(key + '_low'), digits, percent)}; {_fmt(row.get(key + '_high'), digits, percent)}]")


def _primary(table, nonzero=True):
    selected = table.copy()
    for key, value in (('phase', 'pilot'), ('order', 'same'), ('layer', 'both')):
        if key in selected:
            selected = selected[selected[key] == value]
    if nonzero:
        selected = selected[selected.epsilon > 0]
    return selected


def _seed_curve(table, metric, rng):
    """Directions are averaged before source-seed resampling."""
    by_seed = table.groupby(['seed', 'update'], sort=True)[metric].agg(lambda x: x.mean(skipna=False)).unstack('update')
    values = by_seed.to_numpy(dtype=float)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    estimate = np.mean(values, axis=0)
    resamples = np.mean(values[draws], axis=1)
    lower, upper = np.quantile(resamples, [0.025, 0.975], axis=0)
    return by_seed.columns.to_numpy(dtype=float) / 235, estimate, lower, upper


def _style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.19, 'grid.linewidth': 0.7,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
    })


def _save(fig, root, stem):
    fig.savefig(root / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(root / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def _recovery_plot(root, curves, seed):
    _style()
    primary = _primary(curves)
    baselines = _primary(curves, nonzero=False)
    baselines = baselines[baselines.epsilon == 0]
    protocols = [p for p in PROTOCOLS if p in set(primary.momentum)]
    fig, axes = plt.subplots(len(protocols), 4, figsize=(16, 7.6), squeeze=False, sharex=True)
    rng = np.random.default_rng(seed)
    metrics = [('distance', 'Absoluter Quotientenabstand', r'$d_Q$ (Frobenius)'),
               ('ratio', 'Abstand relativ zur Störung', r'$r(s) = d_Q(s) / d_Q(0^+)$'),
               ('accuracy', 'Testgenauigkeit', 'Accuracy (%)'),
               ('loss', 'Testverlust', 'Kreuzentropie')]
    epsilons = sorted(primary.epsilon.unique())
    for row, protocol in enumerate(protocols):
        for col, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for eps in epsilons:
                subset = primary[(primary.momentum == protocol) & np.isclose(primary.epsilon, eps)]
                if subset.empty:
                    continue
                x, mean, low, high = _seed_curve(subset, metric, rng)
                scale = 100 if metric == 'accuracy' else 1
                color = COLORS.get(round(float(eps), 2), '#6C63A8')
                ax.plot(x, mean * scale, lw=1.9, color=color)
                ax.fill_between(x, low * scale, high * scale, color=color, alpha=.16, linewidth=0)
            if metric in ('accuracy', 'loss'):
                subset = baselines[baselines.momentum == protocol]
                if not subset.empty:
                    x, mean, _, _ = _seed_curve(subset, metric, rng)
                    ax.plot(x, mean * (100 if metric == 'accuracy' else 1), color='#333333', lw=1.4, ls='--')
            if metric == 'ratio':
                ax.axhline(1, color='#666666', lw=.9, ls=':')
                maximum = primary.loc[primary.momentum == protocol, 'ratio'].replace([np.inf, -np.inf], np.nan).max()
                if maximum > 8:
                    ax.set_yscale('log')
            ax.set_xscale('symlog', linthresh=.2)
            ax.set_xticks([0, .1, 1, 2, 5, 10])
            ax.set_xticklabels(['0', '0,1', '1', '2', '5', '10'])
            ax.set_xlim(0, 10)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title, pad=12)
            if row == len(protocols)-1:
                ax.set_xlabel('Weitere Epochen (früher Verlauf gestreckt)')
        axes[row, 0].text(-.30, .5, PROTOCOLS[protocol], transform=axes[row, 0].transAxes,
                           rotation=90, va='center', ha='center', fontsize=11, fontweight='bold')
    handles = [Line2D([0], [0], color=COLORS.get(round(float(eps), 2), '#6C63A8'), lw=2,
                      label=f'ε = {eps:.2f}') for eps in epsilons]
    handles.append(Line2D([0], [0], color='#333333', lw=1.4, ls='--', label='Ungestörte Fortsetzung'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, bbox_to_anchor=(.53, .955))
    fig.suptitle('MNIST-Pilot · Störung bei E20, zehn weitere Epochen', fontsize=17, y=1.015)
    fig.text(.5, -.015, 'Mittelwerte über Ausgangsnetze; 95%-Bootstrapintervalle über Seeds. '
             'Fünf Richtungen pro Seed werden zuerst gemittelt.', ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(.03, .03, 1, .91), h_pad=3, w_pad=2)
    _save(fig, root, 'recovery_curves')


def _probability_plot(root, summary):
    _style()
    primary = _primary(summary)
    fixed = 'fixed_return' if 'fixed_return_mean' in primary else 'initially_outside_return'
    fixed_title = 'Eintritt in den festen Zielbereich' if fixed == 'fixed_return' else 'Eintritt in die bewegte Referenzhülle'
    metrics = [('region_member', 'Mitglied der Task-Region am Ende'),
               (fixed, fixed_title),
               ('local_member', 'Nähe zur individuellen Kontrolltrajektorie')]
    if 'perturbation_induced_return_mean' in primary:
        metrics.insert(2, ('perturbation_induced_return', 'Störungsbedingter Austritt + Rückkehr'))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5.4), sharey=True)
    protocol_colors = {'preserve': '#2878B5', 'reset': '#C74753'}
    for ax, (metric, title) in zip(axes, metrics):
        for protocol, label in PROTOCOLS.items():
            rows = primary[primary.momentum == protocol].sort_values('epsilon')
            if rows.empty or metric+'_mean' not in rows:
                continue
            x = rows.epsilon.to_numpy(dtype=float)
            mean = rows[metric+'_mean'].to_numpy(dtype=float)
            low = rows[metric+'_low'].to_numpy(dtype=float)
            high = rows[metric+'_high'].to_numpy(dtype=float)
            offset = -.001 if protocol == 'preserve' else .001
            ax.errorbar(x+offset, mean*100, yerr=np.maximum(0, np.vstack((mean-low, high-mean))*100),
                        color=protocol_colors[protocol], marker='o', ms=5, capsize=4, lw=1.5, label=label)
        if metric == 'perturbation_induced_return' and 'initial_baseline_fixed_member_mean' in primary and primary['initial_baseline_fixed_member_mean'].eq(0).all():
            ax.text(.5, .55, 'Keine Baseline anfangs im Zielbereich\n→ Rückkehr hier nicht prüfbar',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10, color='#555555')
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel('Relative Störstärke ε')
        ax.set_xticks(sorted(primary.epsilon.unique()))
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel('Anteil aller gestörten Forks (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, frameon=False)
    fig.suptitle('Mitgliedschaft, Eintritt in eine Zielregion und Rückkehr nach einer Störung', fontsize=15, y=.99)
    fig.text(.5, .015, '95%-Bootstrapintervalle über Ausgangsnetze; Richtungen innerhalb eines Seeds gemittelt. '
             'Rückkehranteil mit allen Störläufen im Nenner; feste Region: sekundäre Ergänzung vor Ergebnisanalyse.', ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, .16, 1, .91))
    _save(fig, root, 'return_probability')


def _table(summary):
    fixed = 'fixed_return' if 'fixed_return_mean' in summary else 'initially_outside_return'
    lines = ['| Momentum | ε | r am Ende, 95%-KI | Task-Region, 95%-KI | Fester Zielbereich: außen → innen | Störungsbedingte Rückkehr | Individuelle Nähe, 95%-KI |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for _, row in _primary(summary).sort_values(['momentum','epsilon']).iterrows():
        lines.append(f"| {PROTOCOLS.get(row.momentum, row.momentum)} | {row.epsilon:.2f} | "
                     f"{_estimate(row, 'ratio')} | {_estimate(row, 'region_member', 1, True)} | "
                     f"{_estimate(row, fixed, 1, True)} | {_estimate(row, 'perturbation_induced_return', 1, True)} | "
                     f"{_estimate(row, 'local_member', 1, True)} |")
    return '\n'.join(lines)


def _count(table, column):
    return int(table[column].fillna(False).astype(bool).sum()) if column in table else 0


def _conclusion(primary_summary, endpoints):
    """Describe evidence without converting proximity or accuracy into attraction."""
    lower = primary_summary['ratio_low'].to_numpy(dtype=float)
    upper = primary_summary['ratio_high'].to_numpy(dtype=float)
    supported = primary_summary[np.isfinite(upper) & (upper < 1)]
    diverged = primary_summary[np.isfinite(lower) & (lower > 1)]
    n = len(primary_summary)
    if len(supported) == n and n:
        h1 = ('Alle untersuchten Störstärke-/Momentum-Zellen zeigen eine mittlere Annäherung an die '
              'individuelle Kontrolltrajektorie: Die obere Grenze des seedweisen 95%-Intervalls von r liegt unter 1. '
              'Das spricht für lokale Rückstelltendenzen unter dieser Update-Dynamik.')
    elif len(supported):
        cells = ', '.join(f"{r.momentum}, ε={r.epsilon:.2f}" for _,r in supported.iterrows())
        h1 = (f'Eine mittlere Annäherung mit oberer 95%-Intervallgrenze von r unter 1 tritt in {len(supported)}/{n} '
              f'Zellen auf ({cells}). Die Rückstelltendenz ist damit bedingungsabhängig; allgemeine Bahnstabilität '
              'über das getestete Raster ist nicht belegt.')
    elif len(diverged) == n and n:
        h1 = ('In allen untersuchten Störstärke-/Momentum-Zellen liegt selbst die untere 95%-Intervallgrenze '
              'von r über 1. Nach zehn weiteren Epochen sind die Forks von ihrer individuellen Kontrolltrajektorie '
              'weiter entfernt als unmittelbar nach der Störung. H1 wird für diesen Horizont nicht unterstützt.')
    else:
        h1 = ('Keine untersuchte Zelle zeigt eine mittlere Annäherung mit oberer 95%-Intervallgrenze von r unter 1. '
              'Die Daten liefern damit keinen belastbaren Nachweis von H1 für den festgelegten Horizont.')
    returns = _count(endpoints, 'fixed_return')
    initially_outside = len(endpoints)-_count(endpoints, 'initial_fixed_region_member')
    induced = _count(endpoints, 'perturbation_induced_return')
    structural = _count(endpoints, 'fixed_structural_return')
    structural_outside = len(endpoints) - _count(endpoints, 'initial_fixed_structural_member')
    eligible_baselines = _count(endpoints, 'initial_baseline_fixed_member')
    members = _count(endpoints, 'region_member')
    functional = _count(endpoints, 'functional_member')
    h2 = (f'{returns}/{initially_outside} unmittelbar nach der Störung außerhalb des festen Zielbereichs '
          f'liegende Forks erreichen ihn am Ende. Insgesamt gehören {members}/{len(endpoints)} gestörte Endpunkte '
          f'zu dieser Region. In {induced} Fällen lag die ungestörte Ausgangskonfiguration bereits im Zielbereich, '
          'die Störung führte hinaus und die Fortsetzung wieder hinein. Nur diese engere Kategorie weist '
          'einen beobachteten störungsbedingten Austritt mit anschließender Rückkehr aus. ')
    if eligible_baselines == 0:
        h2 += ('Hier lag jedoch keine ungestörte Ausgangskonfiguration bereits im festen kombinierten Zielbereich; '
               'das strengere störungsbedingte Rückkehrkriterium ist damit nicht prüfbar. ')
    if structural_outside == 0:
        h2 += (f'Alle {len(endpoints)} gestörten Ausgangszustände lagen bereits innerhalb der festen strukturellen '
               'Grenze. Die Intervention erzeugte keinen sofortigen strukturellen Austritt; eine Rückkehr nach '
               'einem solchen Austritt wurde in diesem Raster daher nicht getestet. Das ist kein Gegenbeweis '
               'gegen H2. Eintritt in die kombinierte Region betrifft hier Funktionskriterien oder bereits '
               'bestehende strukturelle Nähe und belegt keine strukturelle Attraktion.')
    elif structural:
        h2 += (f'{structural} Forks überschreiten zugleich die feste strukturelle Eintrittsgrenze. '
               'Auch dieser Eintritt kann mit gewöhnlichem Weitertraining verbunden sein; die ungestörten '
               'Fortsetzungen und individuelle Abstände bleiben für die Einordnung entscheidend.')
    else:
        h2 += ('Kein Fork kehrt nach anfänglichem Verlassen der festen strukturellen Grenze zurück. '
               'Ein Eintritt in die kombinierte Task-Region betrifft damit Funktionskriterien oder bereits '
               'bestehende strukturelle Nähe; daraus folgt kein struktureller Attraktionsnachweis.')
    local_return = _count(endpoints, 'local_return')
    h3 = (f'Die vorab festgelegte Regel für individuelle Nähe verwendet 10 % der typischen Distanz '
          f'zeitlich passender unabhängiger Referenznetze plus Funktionskriterien. Diese zeitabhängige Umgebung '
          f'wird in {local_return} Fällen nach anfänglichem Verlassen wieder erreicht. '
          'Bloße Endpunktnähe ist keine lokale Rückkehr; für eine Rückstelltendenz ist zusätzlich Kontraktion '
          'gegenüber der anfänglichen Stördistanz relevant. Eine Basin-Grenze ist mit drei Amplituden und zehn '
          'Ausgangsnetzen nicht lokalisiert.')
    if len(supported):
        classification = 'Bedingungsabhängige lokale Rückstelltendenz; kein allgemeiner Attraktornachweis.'
    elif induced and structural:
        classification = 'Empirische Region-Rückkehr, ohne gesicherten Nachweis individueller Bahnstabilität.'
    elif functional:
        classification = 'Funktionale Eignung am Endpunkt, ohne gesicherten strukturellen Rückkehrnachweis.'
    else:
        classification = 'Keine bestätigte strukturelle Rückkehr unter den festgelegten Kriterien.'
    return classification, h1, h2, h3


def make_report(root):
    root = Path(root)
    curves = _read_csv(root/'recovery_metrics.csv')
    endpoints = _read_csv(root/'endpoint_metrics.csv')
    summary = _read_csv(root/'endpoint_summary.csv')
    config = json.loads((root/'config.json').read_text())
    calibration = json.loads((root/'region_thresholds.json').read_text())
    status_path = root/'analysis_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    primary_endpoints = _primary(endpoints)
    primary_summary = _primary(summary)
    baseline_endpoints = _primary(endpoints, nonzero=False)
    baseline_endpoints = baseline_endpoints[baseline_endpoints.epsilon == 0]
    baseline_final_members = _count(baseline_endpoints, 'region_member')
    if primary_endpoints.empty:
        raise ValueError('No primary pilot endpoints; refusing to generate an empty scientific report')
    _recovery_plot(root, curves, config.get('bootstrap_seed', 620260116))
    _probability_plot(root, summary)
    classification, h1, h2, h3 = _conclusion(primary_summary, primary_endpoints)
    primary_curves = _primary(curves)
    early_minima = primary_curves.groupby('job_id').ratio.min()
    transient_count = int((early_minima < 1).sum())
    if transient_count:
        h1 += (f' Deskriptiv zeigen {transient_count}/{len(early_minima)} Störläufe während der gespeicherten '
               f'Zwischenstände zumindest vorübergehend r<1 (kleinster gemessener Wert {_fmt(early_minima.min(),4)}). '
               'Vorübergehende Kontraktion und fehlende Kontraktion am Zehn-Epochen-Endpunkt sind deshalb getrennte Befunde.')
    final = calibration['thresholds'][-1]
    ref_count = final.get('reference_count', len(calibration.get('reference_ids', [])))
    source_count = primary_endpoints.seed.nunique()
    nruns = len(primary_endpoints)
    allruns = len(endpoints)
    table = _table(summary)
    barriers = primary_endpoints.barrier.dropna() if 'barrier' in primary_endpoints else pd.Series(dtype=float)
    barrier_text = (f'Unter den gültigen Barrierewerten: Median {_fmt(barriers.median(), 5)}, Maximum {_fmt(barriers.max(), 5)} Kreuzentropie-Einheiten '
                    f'({len(barriers)} ausgewertete Störläufe).') if len(barriers) else 'Keine gültigen Barrierewerte verfügbar.'
    robustness_lines = ['| Momentum | ε | q95-Region | q90-Strukturgrenze | Referenzhälfte A | Referenzhälfte B |',
                        '|---|---:|---:|---:|---:|---:|']
    for _, r in primary_summary.sort_values(['momentum','epsilon']).iterrows():
        robustness_lines.append(f"| {r.momentum} | {r.epsilon:.2f} | {_estimate(r,'region_member',1,True)} | "
                                f"{_estimate(r,'q90_member',1,True)} | {_estimate(r,'subset_a_member',1,True)} | "
                                f"{_estimate(r,'subset_b_member',1,True)} |")
    noise = endpoints[endpoints['order'] == 'different'] if 'order' in endpoints else endpoints.iloc[0:0]
    noise_lines = ['| ε | Läufe | Mittlerer Endabstand | Mittlere Accuracy |', '|---:|---:|---:|---:|']
    for eps, group in noise.groupby('epsilon'):
        by_seed = group.groupby('seed')[['distance','accuracy']].agg(lambda x: x.mean(skipna=False))
        noise_lines.append(f"| {eps:.2f} | {len(group)} | {_fmt(by_seed.distance.mean(skipna=False))} | {_fmt(by_seed.accuracy.mean(skipna=False),2,True)} |")
    paired_lines = ['| ε | Differenz r: Fortsetzung − Reset (seedweise) |', '|---:|---:|']
    for eps, group in primary_endpoints.groupby('epsilon'):
        paired = group.groupby(['seed','momentum']).ratio.agg(lambda x: x.mean(skipna=False)).unstack('momentum')
        if {'preserve','reset'} <= set(paired.columns):
            d = (paired['preserve'] - paired['reset']).to_numpy()
            rng = np.random.default_rng(620260117)
            boot = d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
            lo, hi = np.quantile(boot,[.025,.975])
            paired_lines.append(f'| {eps:.2f} | {_fmt(d.mean())} [{_fmt(lo)}; {_fmt(hi)}] |')
    functional_lines = ['| Momentum | ε | Accuracy vor Störung | Direkt danach | Am Ende | Loss direkt danach | Loss am Ende |',
                        '|---|---:|---:|---:|---:|---:|---:|']
    initial_curves = _primary(curves)
    initial_curves = initial_curves[initial_curves['update'] == 0]
    for (protocol, eps), group in initial_curves.groupby(['momentum','epsilon']):
        first = group.groupby('seed')[['baseline_accuracy','accuracy','loss']].agg(lambda x: x.mean(skipna=False)).mean(skipna=False)
        final_group = primary_endpoints[(primary_endpoints.momentum == protocol) & np.isclose(primary_endpoints.epsilon, eps)]
        last = final_group.groupby('seed')[['accuracy','loss']].agg(lambda x: x.mean(skipna=False)).mean(skipna=False)
        functional_lines.append(f"| {protocol} | {eps:.2f} | {_fmt(first.baseline_accuracy,3,True)} | "
                                f"{_fmt(first.accuracy,3,True)} | {_fmt(last.accuracy,3,True)} | "
                                f"{_fmt(first.loss,5)} | {_fmt(last.loss,5)} |")
    biggest = float(primary_endpoints.epsilon.max())
    first_big = initial_curves[np.isclose(initial_curves.epsilon,biggest)]
    last_big = primary_endpoints[np.isclose(primary_endpoints.epsilon,biggest)]
    first_big_mean = first_big.groupby('seed')[['baseline_accuracy','baseline_loss','accuracy','loss']].agg(lambda x:x.mean(skipna=False)).mean(skipna=False)
    last_big_mean = last_big.groupby('seed')[['accuracy','loss','baseline_accuracy','baseline_loss']].agg(lambda x:x.mean(skipna=False)).mean(skipna=False)
    functional_text = (f"Bei der größten getesteten Störung ε={biggest:.2f} verändert sich die mittlere Accuracy "
                       f"unmittelbar von {_fmt(first_big_mean.baseline_accuracy,3,True)} auf {_fmt(first_big_mean.accuracy,3,True)}, "
                       f"der Loss von {_fmt(first_big_mean.baseline_loss,5)} auf {_fmt(first_big_mean.loss,5)}. "
                       f"Nach zehn weiteren Epochen liegen die gestörten Forks bei {_fmt(last_big_mean.accuracy,3,True)} "
                       f"und Loss {_fmt(last_big_mean.loss,5)}; die zugehörigen ungestörten Fortsetzungen bei "
                       f"{_fmt(last_big_mean.baseline_accuracy,3,True)} und Loss {_fmt(last_big_mean.baseline_loss,5)}. "
                       "Dies beschreibt beobachtete Funktionsänderungen und ihre Erholung; kleine Störungen können "
                       "bereits unmittelbar danach kaum messbaren Leistungsverlust zeigen. Die strenge Referenzschwelle "
                       "für Funktionsmitgliedschaft und die direkte Erholung relativ zur eigenen Baseline sind verschiedene Kriterien.")
    initial_members = _count(primary_endpoints, 'initial_region_member')
    moving_returns = _count(primary_endpoints, 'initially_outside_return')
    fixed_initial_members = _count(primary_endpoints, 'initial_fixed_region_member')
    fixed_returns = _count(primary_endpoints, 'fixed_return')
    induced_returns = _count(primary_endpoints, 'perturbation_induced_return')
    baseline_fixed_members = primary_endpoints.groupby('seed').initial_baseline_fixed_member.first().sum() if 'initial_baseline_fixed_member' in primary_endpoints else 0
    retained = int(primary_endpoints.retained_region.fillna(False).astype(bool).sum())
    sampled = int(primary_endpoints.sampled_return.fillna(False).astype(bool).sum())
    extension_status_path = root/'expanded_training_status.json'
    extension_status = json.loads(extension_status_path.read_text()) if extension_status_path.exists() else {}
    if extension_status.get('complete'):
        extra_scope = ('Die vollständig ausgeführte Erweiterung über drei Datensätze, drei Störzeitpunkte, '
                       'drei Schichtvarianten und vier nichttriviale Amplituden wird im '
                       '[erweiterten Ergebnisbericht](expanded_analysis.md) separat ausgewertet. '
                       'Die Zahlen dieses Pilotabschnitts beziehen sich ausschließlich auf die fünf Richtungen des MNIST-Piloten.')
    elif (root/'expanded_preregistration.json').exists():
        extra_scope = ('Die erweiterte Matrix ist in [expanded_preregistration.json](expanded_preregistration.json) '
                       'festgelegt. Die hier berichteten Zahlen betreffen ausschließlich den MNIST-Pilot; '
                       'zusätzliche Ergebnisse werden separat dokumentiert.')
    else:
        extra_scope = ('Dieser Abschnitt wertet ausschließlich den MNIST-Pilot aus. Eine Ausweitung auf weitere '
                       'Datensätze, Störzeitpunkte, einzelne Schichten und die vierte nichttriviale Amplitude '
                       'ist nicht Gegenstand dieser Pilotzahlen.')
    amendment_path = root/'analysis_amendment.json'
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    amendment_note = (f"Die festen Zielbereichsanalysen, der störungsbedingte Austritt mit Rückkehr und die Unterscheidung "
                      f"zwischen anfänglicher und späterer lokaler Nähe wurden als **sekundäre Ergänzung** am "
                      f"{amendment.get('created_at', 'dokumentierten Zeitpunkt')} aufgenommen: während des Pilottrainings, "
                      "vor jeglicher Pilot-Endpunkt- oder Trajektorienauswertung. Sie waren nicht Teil der vor Trainingsstart "
                      "festgeschriebenen primären Analyse. Die primären Schwellen, Läufe und Trainingsprotokolle blieben "
                      "unverändert. [Zeitpunkt und Begründung](analysis_amendment.json).")

    failure_text = json.dumps(status, ensure_ascii=False, indent=2)
    basin = f'''# Basin-Analyse: kontrollierter MNIST-Pilot

{classification}

## Tatsächlich ausgeführter Umfang

{nruns} gestörte Hauptläufe aus {source_count} unabhängigen Ausgangsnetzen bei E20, fünf Richtungen je Störstärke und Momentum-Protokoll. ε = 0,01; 0,05; 0,20 betrifft jeweils beide Gewichtsmatrizen. Hinzu kommen die ungestörten Fortsetzungen und Kontrollen mit abweichender Batchreihenfolge; die Endpunkttabelle enthält insgesamt {allruns} Läufe. Jeder Lauf umfasst zehn weitere Epochen auf dem vollständigen MNIST-Trainingssatz. {ref_count} unabhängige Referenznetze werden bis zu denselben Trainingsständen fortgesetzt.

{extra_scope}

## Vorab festgelegte Region und Rückkehr

Die primären Kriterien der zeitlich passenden Referenzhülle wurden nach Referenztraining und vor den Störungsläufen gespeichert; Zeitstempel und Referenz-Hashes stehen in [region_thresholds.json](region_thresholds.json). Die Referenzmodelle stammen aus unabhängigen Initialisierungen I1–I10 und werden nicht gestört. An jedem Messzeitpunkt wird die Region neu für den entsprechenden Trainingsstand kalibriert. Dieser zeitabhängige Bereich beschreibt einen Korridor geeigneter Lösungen, keinen festen invarianten Attraktor.

{amendment_note}

Der Abstand ist die Frobeniusnorm beider zusammengefasster Gewichtsmatrizen nach kanonischer ReLU-Normierung und Git-Re-Basin-Assignment. Bei einer verborgenen Schicht löst das Hungarian-Assignment die Permutationsoptimierung exakt für diesen kanonischen Abstand. Die Gauge wird nur auf Analysekopien angewendet; das SGD-Training läuft in den unveränderten Koordinaten.

Ein Zustand gehört zur Task-Region, wenn sein medianer Abstand zu den Referenzen höchstens dem 95%-Quantil ihrer Leave-one-out-Mediandistanzen entspricht **und** Accuracy sowie Loss die Referenzschwellen erfüllen. Am Endpunkt gelten:

- Strukturelle Distanzgrenze: {_fmt(final.get('distance_q95'),6)}; alternative q90-Grenze {_fmt(final.get('distance_q90'),6)}.
- Accuracy ≥ {_fmt(final.get('accuracy_q05'),3,True)} und Test-Loss ≤ {_fmt(final.get('loss_q95'),6)}.
- Individueller Näheradius: {_fmt(final.get('local_radius'),6)}; zusätzlich dieselben Funktionskriterien.

Es werden zwei geometrisch verschiedene Definitionen getrennt ausgewertet:

- **Bewegte Referenzhülle:** Referenzen und Schwellen entsprechen jedem jeweiligen Trainingsstand. {initial_members}/{nruns} gestörte Anfangszustände waren bereits Mitglied; {moving_returns}/{nruns} verlaufen von außen nach innen. Weil die Hülle selbst wandert, ist dies keine Rückkehr in eine feste Region. `retained_region` zählt {retained} durchgehend innerhalb liegende Läufe, `sampled_return` {sampled} Eintritte nach einem an Messzeitpunkten beobachteten Außenaufenthalt.
- **Fester Zielbereich:** Die Referenznetze und Schwellen des Endhorizonts werden an allen Zeitpunkten unverändert verwendet. {fixed_initial_members}/{nruns} gestörte Anfangszustände liegen bereits darin; {fixed_returns}/{nruns} wechseln von außen nach innen. Der Eintritt kann auch gewöhnliches Weitertraining abbilden. **Störungsbedingte Rückkehr** verlangt zusätzlich, dass der ungestörte Zustand vor der Intervention innerhalb lag: {induced_returns}/{nruns} Läufe erfüllen dieses strengere Kriterium.

Die ungestörten Ausgangskonfigurationen liegen bei {int(baseline_fixed_members)}/{source_count} Seeds im festen Zielbereich. Am Ende erfüllen {baseline_final_members}/{len(baseline_endpoints)} ungestörte Fortsetzungen die kombinierte Regionbedingung (beide Momentum-Protokolle). Eine Region aus nur zehn Referenzen kann selbst plausible ungestörte Netze ablehnen; ihre Grenzen sind restriktive empirische Kriterien, keine garantierten Populationsgrenzen. Die Return-Rate in den Abbildungen verwendet alle gestörten Läufe als Nenner. Die zeitliche Zuordnung erfolgt nur an gespeicherten Messzeitpunkten; Ereignisse dazwischen können übersehen werden.

## Endpunkte und Seed-Level-Unsicherheit

{table}

Richtungen werden zuerst je Ausgangsnetz gemittelt, dann werden {source_count} Ausgangsnetze 10.000-mal resampelt. Angegeben sind punktweise 95%-Perzentilintervalle, keine simultanen Intervalle und keine unabhängige Stichprobe von {nruns} Netzen. Bei ausschließlich 0- oder 1-Werten kann der empirische Bootstrap ein Intervall ohne Breite ergeben; das beweist keine sichere Populationsrate. Die Kalibrierungsunsicherheit der kleinen Referenzkohorte ist in diesen Intervallen nicht enthalten.

![Recovery-Kurven](recovery_curves.png)

![Region, Rückkehr und individuelle Nähe](return_probability.png)

## Funktionale Änderung und Erholung

{functional_text}

{chr(10).join(functional_lines)}

Die Tabellenwerte mitteln zuerst Richtungen und anschließend die Ausgangsnetze. Sie beschreiben die Funktion separat vom strukturellen Rückkehrkriterium.

## Momentum und SGD-Rauschen

Protokoll A übernimmt den gespeicherten Momentumzustand in beiden Forks. Protokoll B setzt ihn in beiden Forks gleichzeitig auf null. Lernrate, Batchgröße, Aufgabe und zukünftige Batchreihenfolge bleiben innerhalb jedes gepaarten Hauptvergleichs gleich. Der direkte gepaarte Protokollvergleich mittelt Richtungen zunächst innerhalb jedes Seeds:

{chr(10).join(paired_lines)}

Kontrolle C ändert nur die zukünftige Batchreihenfolge des Vergleichsarms. Der ε=0-Lauf quantifiziert die Divergenz, die bereits ohne Gewichtsintervention durch andere SGD-Updates entsteht. r ist bei ε=0 nicht definiert und wird nicht künstlich auf null gesetzt.

{chr(10).join(noise_lines)}

## Alternative Auswertungen und Referenzmengen

Zusätzlich zum Git-Re-Basin-Abstand sind kanonisches PIF/Hungarian-Matching und Alpha-Matching/-Dissimilarität in den Rohmesswerten gespeichert. Alpha ist ein nach Gewichtsbeträgen gewichtetes Vorzeichenmaß und keine austauschbare euklidische Distanz. Die Endpunktmitgliedschaft wird mit der engeren q90-Strukturgrenze sowie zwei getrennten Referenzhälften geprüft:

{chr(10).join(robustness_lines)}

Diese Varianten dienen als Sensitivitätsanalyse. Referenzhälften mit fünf Modellen haben besonders grobe Quantile; unterschiedliche Mitgliedschaft zeigt Abhängigkeit von der operationalen Regionsdefinition.

## Loss-Barrieren

Nach kanonischem und permutativem Matching werden elf gleichmäßig verteilte Interpolationspunkte auf dem vollständigen Testset ausgewertet. Die Barriere ist der größte Loss-Überschuss gegenüber der linearen Verbindung der Endpunkt-Losswerte. {barrier_text}

Dies ist ein diskretes Raster, keine garantierte obere Schranke der kontinuierlichen Barriere. Eine kleine Barriere beschreibt eine günstige Verbindung der Endpunkte, keine Rückkehrdynamik und keinen mathematischen Attraktor.

## Divergenzen, Ausfälle und Integrität

Die maschinenlesbare Auswertungsbilanz lautet:

```json
{failure_text}
```

Fehlgeschlagene Läufe dürfen nicht als erfolgreiche Rückkehr erscheinen. Fehlende kontinuierliche Werte werden in Seed-Mittelwerten und deren Intervallen nicht stillschweigend ausgelassen; das betreffende Gesamtergebnis bleibt undefiniert. Die separat angegebene Barriere-Spannweite beschreibt ausdrücklich nur die gültigen Barrierewerte. Ein gescheiterter oder nicht endlicher Lauf erfüllt die Rückkehrkriterien nicht; kontinuierliche Messwerte sind dort nicht sinnvoll definiert. Prüfungen zu unveränderten Originalarchiven, Symmetrien und deterministischer Fortsetzung liegen im Unterordner `validation/`.

## Interpretation nach Hypothese

**H1 – Trajektorienstabilität.** {h1}

**H2 – Lösungsregion.** {h2}

**H3 – Lokale Lösungsfamilie/Basin.** {h3}

Für die erweiterte faktorielle Untersuchung ist eine Störrichtung pro Seed und Zelle vorgesehen. Ihre vielen punktweisen Intervalle sind explorativ, ohne Korrektur für multiples Vergleichen; sie ersetzen weder die Richtungsrobustheit des Piloten noch eine Kartierung von Basin-Grenzen.

Die Aussage gilt für SGD mit konstanter Lernrate 0,1, Momentum 0,9, dem festgelegten Datenstrom und diesem Zehn-Epochen-Horizont. Gute Testleistung allein ist kein Attraktornachweis. Unabhängige Netzwerke, ReLU-Symmetrien und zufällige SGD-Abweichungen setzen unterschiedliche Vergleichsmaßstäbe.

## Methodische Quellen

- [Ainsworth et al., Git Re-Basin](https://arxiv.org/abs/2209.04836): Permutationssymmetrien und Weight Matching.
- [Garipov et al., Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026): Verbindungen zwischen Lösungen und Loss-Landschaften.
- [Versuchsvorgabe](provenance/experiment_specification.md): Hypothesen, Kontrollen und Ausgaben dieses Experiments.
'''
    (root/'basin_analysis.md').write_text(basin, encoding='utf-8')
    conclusion = f'''# Attraktor-Schlussfolgerung

**{classification}**

{h1}

{h2}

{h3}

{functional_text}

{table}

{amendment_note}

Die Intervalle resampeln Ausgangsnetze nach Mittelung ihrer Störrichtungen. Eine niedrige Loss-Barriere oder gute Accuracy allein gilt nicht als dynamische Rückkehr. Keine dieser Auswertungen beweist einen mathematischen Attraktor oder eine universelle Basin-Struktur.

Untersucht wurden {nruns} gestörte MNIST-Forks aus {source_count} unabhängigen Ausgangsnetzen bei E20 mit zehn zusätzlichen Epochen. {extra_scope}

[Methodik, Kontrollen und vollständige Einordnung](basin_analysis.md) · [Endpunktmesswerte](endpoint_metrics.csv) · [Konfidenzintervalle](endpoint_summary.csv)
'''
    (root/'attractor_conclusion.md').write_text(conclusion, encoding='utf-8')
    readme = f'''# 06 · Attraktorprüfung durch Gewichtsraumstörung

{classification}

Ausführung der Vorgabe [02_Attraktor_Nachweis_durch_Stoerung.md](provenance/experiment_specification.md) in diesem eigenständigen Unterordner. Originalarchive unter `../outputs/` dienen nur als Quellen.

## Ergebnisse

- [Schlussfolgerung](attractor_conclusion.md)
- [Basin-Analyse mit Methodik, Kontrollen und Grenzen](basin_analysis.md)
- [Recovery-Kurven als PNG](recovery_curves.png) · [PDF](recovery_curves.pdf)
- [Regionzugehörigkeit und Rückkehr als PNG](return_probability.png) · [PDF](return_probability.pdf)
- [Messwerte je Zeitpunkt](recovery_metrics.csv), [Endpunkte](endpoint_metrics.csv), [Seed-Level-Intervalle](endpoint_summary.csv)
- [Fork-Manifest](perturbation_manifest.csv), [Konfiguration](config.json), [vorab eingefrorene Referenzschwellen](region_thresholds.json)

## Reproduzieren

Vom Projektverzeichnis aus, mit der vorhandenen Python-Umgebung:

```bash
.venv/bin/python 06_Attractor_Perturbation/code/run.py prepare
.venv/bin/python 06_Attractor_Perturbation/code/run.py references --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/run.py calibrate
.venv/bin/python 06_Attractor_Perturbation/code/run.py pilot --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/run.py analyze --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/run.py verify
```

Vorhandene Läufe werden nur bei passenden Fingerabdrücken wiederverwendet. Konfiguration und Schwellen werden nicht anhand der Störresultate angepasst. `checkpoints/` enthält separate Fork-Gewichte sowie Optimizer-/RNG-Zustände. `code/` enthält die vollständige lokale Implementierung; `provenance/` dokumentiert Vorgabe und Quellen; `validation/` enthält die Prüfungen.

## Umfang

{nruns} gestörte Hauptläufe, {source_count} unabhängige MNIST-Ausgangsnetze, drei Störstärken, fünf Richtungen, zwei Momentum-Protokolle und zehn weitere Epochen; zusätzlich ungestörte und SGD-Rausch-Kontrollen. {ref_count} unabhängige Referenznetze bestimmen die Region. {extra_scope}
'''
    (root/'README.md').write_text(readme, encoding='utf-8')
    return {'classification': classification, 'pilot_perturbations': nruns, 'source_seeds': source_count,
            'figures': ['recovery_curves.png','recovery_curves.pdf','return_probability.png','return_probability.pdf'],
            'reports': ['basin_analysis.md','attractor_conclusion.md','README.md']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    print(json.dumps(make_report(args.root), ensure_ascii=False, indent=2))
