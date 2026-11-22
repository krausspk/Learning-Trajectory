"""German report and scientific figures from measured history-decoding results.

All reported estimates are read from the saved result tables. No paper numbers
or synthetic display values are used. Run only after the probe analysis ends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[1]
NAMES = {
    'W0': 'Eingangsgewichte W0',
    'W1': 'Ausgangsgewichte W1',
    'both': 'Beide Gewichtsmatrizen',
    'diagnostics': 'Accuracy, Loss und Normen',
    'functional_logits': 'Logits auf festen Inputs',
    'covariate_adjusted_both': 'Gewichte nach Kovariatenabzug',
}
COLORS = {
    'W0': '#6F7A89', 'W1': '#B37435', 'both': '#156B73',
    'diagnostics': '#AB6F79', 'functional_logits': '#6855A1',
    'covariate_adjusted_both': '#478D5C',
}


def bools(series):
    return series.astype(str).str.lower().isin(['true', '1'])


def pformat(value):
    if not np.isfinite(float(value)):
        return '–'
    return f'{float(value):.4f}'


def pct(value, digits=1):
    return f'{100 * float(value):.{digits}f} %'


def select(frame, **criteria):
    result = frame
    for name, value in criteria.items():
        result = result[result[name] == value]
    return result


def markdown_table(frame, fields, labels=None, percent_fields=()):
    labels = labels or fields
    lines = ['| ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for row in frame.to_dict('records'):
        values = []
        for field in fields:
            value = row[field]
            if field in percent_fields:
                value = pct(value, 2)
            elif isinstance(value, (float, np.floating)):
                value = f'{value:.4f}' if np.isfinite(value) else '–'
            else:
                value = str(value)
            values.append(value.replace('|', '/'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def save_figure(fig, folder, name):
    for extension in ('png', 'pdf'):
        fig.savefig(folder / f'{name}.{extension}', dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def figures(out, results):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.labelcolor': '#273341',
        'text.color': '#273341', 'xtick.color': '#4E5966',
        'ytick.color': '#4E5966', 'pdf.fonttype': 42,
    })
    ordinary = results[~results.negative_control]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    settings = [
        ('balanced_accuracy', 'ba_ci_low', 'ba_ci_high', 'Balancierte Accuracy'),
        ('auc', 'auc_ci_low', 'auc_ci_high', 'ROC-AUC'),
    ]
    for ax, (estimate, lower, upper, label) in zip(axes, settings):
        markers = {'W0': ('^', 11, 'none', '--'), 'W1': ('s', 8, 'none', ':'), 'both': ('o', 4, COLORS['both'], '-')}
        plotted = []
        for feature in ('W0', 'W1', 'both'):
            part = select(ordinary, condition='reset', features=feature).sort_values('c_epochs')
            if part.empty:
                continue
            x, y = part.c_epochs.to_numpy(), part[estimate].to_numpy()
            error = np.maximum(0, np.vstack([y - part[lower].to_numpy(), part[upper].to_numpy() - y]))
            marker, marker_size, face, line_style = markers[feature]
            ax.errorbar(x, y, yerr=error, color=COLORS[feature], lw=1.8, marker=marker, capsize=4,
                        label=NAMES[feature], markersize=marker_size, markerfacecolor=face,
                        markeredgewidth=1.4, linestyle=line_style)
            plotted.append(np.column_stack((x, y, part[lower].to_numpy(), part[upper].to_numpy())))
        if len(plotted) == 3 and all(np.array_equal(plotted[0], values) for values in plotted[1:]):
            ax.text(.035, .14, 'W0, W1 und beide: identische Werte\nund Intervalle; Kurven überlagern sich.',
                    transform=ax.transAxes, fontsize=9, color='#455563', va='bottom')
        continued = select(ordinary, condition='continued', features='both').sort_values('c_epochs')
        if not continued.empty:
            x, y = continued.c_epochs.to_numpy(), continued[estimate].to_numpy()
            error = np.maximum(0, np.vstack([y - continued[lower].to_numpy(), continued[upper].to_numpy() - y]))
            ax.errorbar(x, y, yerr=error, color='#2877B7', ls='--', lw=1.6, marker='s', capsize=3,
                        label='Beide, Momentum fortgeführt', markersize=4)
        ax.axhline(.5, color='#9A4343', lw=1.1, ls=':')
        ax.set(xticks=[1, 5, 10], xlabel='Gemeinsame C-Epochen', ylabel=label, ylim=(0.25, 1.035))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis='y', alpha=.18)
    axes[0].set_title('Dekodierbarkeit im Gewichtsraum', loc='left', pad=12)
    axes[1].set_title('Trennung der Historien', loc='left', pad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
    fig.suptitle('Welche Lernreihenfolge bleibt nach Aufgabe C erkennbar?', x=.055, ha='left', fontsize=15, fontweight='bold')
    fig.text(.055, .025, 'Fehlerbalken: 95-%-Seed-Block-Bootstrap, bedingt auf die trainierte Probe. Gepunktet: 50-%-Zufallsbasis.', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.32, wspace=.25)
    save_figure(fig, out, 'history_decoding_by_C_duration')

    final = select(ordinary, condition='reset', c_epochs=10)
    rows, labels, colors = [], [], []
    for feature in ('both', 'W0', 'W1', 'covariate_adjusted_both', 'functional_logits', 'diagnostics'):
        part = select(final, features=feature)
        if len(part):
            rows.append(part.iloc[0])
            labels.append(NAMES[feature])
            colors.append(COLORS[feature])
    negative = results[results.negative_control & (results.condition == 'reset') & (results.c_epochs == 10)]
    if len(negative):
        rows.append(negative.iloc[0])
        labels.append('Zufällige Historienlabels')
        colors.append('#A8ADB4')
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    positions = np.arange(len(rows))
    values = np.array([r.balanced_accuracy for r in rows])
    lo = np.array([r.ba_ci_low for r in rows])
    hi = np.array([r.ba_ci_high for r in rows])
    ax.barh(positions, values, height=.58, color=colors, alpha=.9)
    ax.errorbar(values, positions, xerr=np.maximum(0, np.vstack([values-lo, hi-values])), fmt='none', ecolor='#233442', capsize=4, lw=1.3)
    for y, row in enumerate(rows):
        ax.text(1.02, y, f'{pct(row.balanced_accuracy, 2)}   p={pformat(row.permutation_p)}', va='center', fontsize=9.5)
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, 1.30), xlabel='Balancierte Accuracy auf neuen Seed-Blöcken')
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.invert_yaxis()
    ax.axvline(.5, color='#9A4343', ls=':', lw=1.2)
    ax.grid(axis='x', alpha=.13)
    ax.set_title('Nach 10 C-Epochen: Gewichte, Outputs und Kontrollen', loc='left', fontsize=15, pad=22)
    fig.text(.03, .045, 'Schichtproben nach gemeinsamer Gauge und Ausrichtung beider Matrizen. Momentum-Reset an Aufgabenwechseln.\nFehlerbalken: 95-%-Block-Bootstrap. Rechts: unadjustierte p-Werte; sekundäre Holm-Korrektur im Bericht.', fontsize=9)
    fig.subplots_adjust(left=.30, right=.97, top=.86, bottom=.18)
    save_figure(fig, out, 'history_probe_ablation')


def make_report(out=DEFAULT_OUT):
    out = Path(out)
    results = pd.read_csv(out / 'history_probe_results.csv')
    if len(results) != 20 or set(results.condition) != {'reset', 'continued'}:
        raise ValueError('Final report requires all 20 planned endpoints from both momentum conditions')
    results['negative_control'] = bools(results['negative_control'])
    results['primary'] = bools(results['primary'])
    primary_rows = results[results.primary]
    if len(primary_rows) != 1:
        raise ValueError('Report requires exactly one declared primary endpoint')
    primary = primary_rows.iloc[0]
    ordinary = results[~results.negative_control]
    performance = pd.read_csv(out / 'performance_summary.csv')
    paired = pd.read_csv(out / 'performance_paired.csv')
    diagnostics = json.loads((out / 'probe_diagnostics.json').read_text())
    plan = json.loads((out / 'plan.json').read_text()) if (out / 'plan.json').exists() else {}
    final = select(ordinary, condition='reset', c_epochs=10).copy()
    final['feature_name'] = final.features.map(NAMES).fillna(final.features)
    testperf = select(performance, condition='reset', c_epochs=10, split='test')
    testpaired = select(paired, condition='reset', c_epochs=10, split='test')
    paired_display = testpaired.copy()
    accuracy_rows = paired_display.metric == 'accuracy'
    for field in ('diff_H1_minus_H0', 'ci_low', 'ci_high'):
        paired_display.loc[accuracy_rows, field] *= 100
    paired_display['metric'] = paired_display.metric.map({'accuracy': 'Accuracy (Prozentpunkte)', 'loss': 'Cross-Entropy-Loss'})
    perf0 = testperf[testperf.history == 0].iloc[0]
    perf1 = testperf[testperf.history == 1].iloc[0]
    gap = testpaired[testpaired.metric == 'accuracy'].iloc[0]
    unequal = gap.ci_high < 0 or gap.ci_low > 0
    performance_qualification = (f'Die finale C-Accuracy beträgt für A→B→C {pct(perf0.accuracy_mean, 2)} und für B→A→C '
                                 f'{pct(perf1.accuracy_mean, 2)}. Die gepaarte Differenz H1−H0 ist '
                                 f'{100*gap.diff_H1_minus_H0:.2f} Prozentpunkte '
                                 f'(95-%-Intervall [{100*gap.ci_low:.2f}; {100*gap.ci_high:.2f}]). '
                                 + ('Die Endleistung unterscheidet sich damit messbar. Eine historische Spur unter gleicher finaler Leistung ist durch diesen Vergleich nicht nachgewiesen.'
                                    if unequal else 'Dieser Vergleich allein belegt keine Äquivalenz der finalen Leistung.'))
    primary_sig = primary.balanced_accuracy > .5 and primary.permutation_p <= .05
    if primary_sig:
        headline = ('Die Reihenfolge A→B→C beziehungsweise B→A→C ist in diesem Pilotversuch '
                    'aus den finalen Gewichten nach zehn gemeinsamen C-Epochen über Zufallsniveau dekodierbar.')
    else:
        headline = ('Der vorab festgelegte primäre Test liefert in diesem Pilotversuch keinen statistisch '
                    'abgesicherten Nachweis der Dekodierbarkeit nach zehn gemeinsamen C-Epochen.')
    diagnostic = select(final, features='diagnostics')
    functional = select(final, features='functional_logits')
    adjusted = select(final, features='covariate_adjusted_both')
    control_lines = []
    if len(diagnostic):
        r = diagnostic.iloc[0]
        if r.balanced_accuracy > .5 and r.secondary_holm_p <= .05:
            interpretation = 'Bereits diese groben Eigenschaften unterscheiden die Historien nach Holm-Korrektur; die Gewichtsdekodierung ist daher zusammen mit diesen Unterschieden zu interpretieren.'
        else:
            interpretation = 'Diese groben Eigenschaften ergeben nach Holm-Korrektur keinen abgesicherten Dekodierungserfolg; dies belegt keine vollständige Gleichheit der Gruppen.'
        control_lines.append(f'Die reine Diagnoseprobe aus Accuracy, Loss und Normen erreicht {pct(r.balanced_accuracy, 2)} (p={pformat(r.permutation_p)}, '
                             f'Holm-p={pformat(r.secondary_holm_p)}). {interpretation}')
    if len(adjusted):
        r = adjusted.iloc[0]
        control_lines.append(f'Nach ausschließlich im Trainingssplit geschätztem linearem Kovariatenabzug verbleiben {pct(r.balanced_accuracy, 2)} '
                             f'(p={pformat(r.permutation_p)}). Dieser Abzug kontrolliert lineare Zusammenhänge mit den erfassten Diagnosen; '
                             'er ist kein Beweis vollständiger funktionaler Gleichheit.')
    if len(functional):
        r = functional.iloc[0]
        if r.balanced_accuracy > .5 and r.secondary_holm_p <= .05:
            interpretation = 'Auch die unnormalisierten Netzwerkoutputs auf dem festen Inputsatz enthalten damit nach Holm-Korrektur dekodierbare Historieninformation.'
        else:
            interpretation = 'Dieser sekundäre Test zeigt nach Holm-Korrektur keine abgesicherte funktionale Dekodierbarkeit auf dem festen Inputsatz; funktionale Gleichheit folgt daraus nicht.'
        control_lines.append(f'Die Probe auf finalen Logits erreicht {pct(r.balanced_accuracy, 2)} (p={pformat(r.permutation_p)}, '
                             f'Holm-p={pformat(r.secondary_holm_p)}). {interpretation} '
                             'Die Probe betrachtet Werte vor der Softmax-Normierung: Gemeinsame Logit-Verschiebungen können dekodierbar sein, obwohl Softmax-Wahrscheinlichkeiten und vorhergesagte Klassen gleich bleiben. Unterschiede dieser Größen wurden mit dieser Probe nicht unmittelbar geprüft.')
    functional_sig = len(functional) and functional.iloc[0].balanced_accuracy > .5 and functional.iloc[0].secondary_holm_p <= .05
    if functional_sig:
        summary_qualification = (f'Die finale Logit-Probe ist ebenfalls erfolgreich ({pct(functional.iloc[0].balanced_accuracy, 2)}; '
                                 f'Holm-p={pformat(functional.iloc[0].secondary_holm_p)}). Der Befund zeigt somit eine historische Spur '
                                 'in Gewichten und unnormalisierten Outputs; eine ausschließlich im Gewichtsraum verbleibende Spur bei identischen Netzwerkoutputs ist hier nicht nachgewiesen.')
    else:
        summary_qualification = ('Die funktionale Kontrolle wird unten separat berichtet. Ein negatives Resultat dieser Probe allein kann keine vollständige Gleichheit des Netzwerkverhaltens begründen.')
    negative = results[results.negative_control]
    negative_text = '\n\n'.join(
        f'Zufällige, innerhalb der Seed-Paare getauschte Historienlabels: {pct(r.balanced_accuracy, 2)}, '
        f'95-%-Intervall [{pct(r.ba_ci_low, 2)}; {pct(r.ba_ci_high, 2)}], p={pformat(r.permutation_p)}. '
        'Eine einzelne endliche Stichprobe muss nicht exakt 50 % ergeben.' for r in negative.itertuples())
    temporal = select(ordinary, condition='reset', features='both').sort_values('c_epochs')
    temporal_text = '; '.join(f'C{int(r.c_epochs)}: {pct(r.balanced_accuracy, 2)} (p={pformat(r.permutation_p)})' for r in temporal.itertuples())
    continued = select(ordinary, condition='continued', features='both').sort_values('c_epochs')
    continued_text = ('; '.join(f'C{int(r.c_epochs)}: {pct(r.balanced_accuracy, 2)} (p={pformat(r.permutation_p)})' for r in continued.itertuples())
                      if len(continued) else 'Keine separat ausgewertete Kohorte mit fortgeführtem Momentum liegt vor.')
    degenerate = ordinary[np.isclose(ordinary.ba_ci_low, ordinary.ba_ci_high)]
    degenerate_text = ('Bei mindestens einem Endpunkt kollabiert das Bootstrap-Intervall auf einen Punkt. Insbesondere bei fehlerfreier '
                       'Klassifikation lässt der empirische Bootstrap keine neuen Fehlertypen entstehen; daraus folgt keine garantierte '
                       'perfekte Generalisierung auf die Population.' if len(degenerate) else '')
    symmetry_rows = []
    for condition, detail in diagnostics.get('conditions', {}).items():
        check = detail.get('symmetry_checks', {})
        if check:
            symmetry_rows.append({'condition': condition, 'rows': check['aligned_rows_checked'],
                                  'permutation_error': f"{check['permutation_max_weight_error']:.3g}",
                                  'gauge_error': f"{check['gauge_max_weight_error']:.3g}"})
    symmetry_table = (markdown_table(pd.DataFrame(symmetry_rows),
                                    ['condition', 'rows', 'permutation_error', 'gauge_error'],
                                    ['Momentum', 'Gewichtskonfigurationen', 'Max. Umordnungsfehler', 'Max. Gauge-Fehler'])
                      if symmetry_rows else 'Keine Symmetrieprüfung im Ergebnisarchiv gefunden.')
    all_results = results.copy()
    all_results['label'] = all_results.features.map(NAMES).fillna(all_results.features)
    all_results.loc[all_results.negative_control, 'label'] = 'Zufällige Historienlabels'
    manifest = pd.read_csv(out / 'history_run_manifest.csv') if (out / 'history_run_manifest.csv').exists() else None
    n_train, n_val, n_test = (int(primary[k]) for k in ('n_train_blocks', 'n_val_blocks', 'n_test_blocks'))
    n_blocks = n_train + n_val + n_test
    amendment_path = out / 'provenance' / 'pre_analysis_amendment.json'
    amendment_text = ('Vor dem Anpassen irgendeiner Probe und vor Einsicht in die zurückgehaltenen Resultate wurde der Plan dokumentiert geändert: '
                      '999 statt 199 Permutationen auch für sekundäre Endpunkte; die Negativkontrolle gehört ebenfalls zur Holm-Familie mit 19 Endpunkten. '
                      'Mit 199 Permutationen wäre der kleinste erreichbare p-Wert zu groß für diese Korrektur gewesen. '
                      'Trainingsparameter und primärer Endpunkt blieben gleich; Einzelheiten in `provenance/pre_analysis_amendment.json`. '
                      'Dies ist ein lokal eingefrorener Analyseplan, keine öffentliche Präregistrierung.'
                      if amendment_path.exists() else '')
    repeat_path = out / 'provenance' / 'analysis_repeat_check.json'
    execution_note = ''
    if repeat_path.exists():
        repeat = json.loads(repeat_path.read_text())
        if repeat.get('passed') and not repeat.get('protocol_changed'):
            execution_note = (f'Die langlaufenden Prozesse wurden nach rund 20 Minuten mit einem Terminationssignal (Exit-Code 143) beendet und mit unverändertem Protokoll neu gestartet. '
                              f'Vollständige Trainingsarchive wurden vor Wiederverwendung validiert. Die {repeat["completed_endpoints_repeated"]} '
                              'bereits fertigen Analyseendpunkte wurden erneut berechnet; gespeicherte Scores, Permutationsverteilungen, '
                              'Koeffizienten und Indizes stimmen bitweise überein (`provenance/analysis_repeat_check.json`). '
                              'Der Neustart ist keine zusätzliche experimentelle Replikation.')
    sections = [
        '# Experiment 3: Aufgabenhistorie aus finalen Gewichten dekodieren',
        headline,
        (f'Der **primäre Endpunkt** (beide kanonisch ausgerichteten Gewichtsmatrizen, Momentum-Reset, C10) erreicht '
         f'**{pct(primary.balanced_accuracy)} balancierte Accuracy** auf {n_test} vollständig zurückgehaltenen Seed-Paaren '
         f'({2*n_test} Netzen), mit 95-%-Seed-Block-Intervall [{pct(primary.ba_ci_low)}; {pct(primary.ba_ci_high)}] '
         f'und blockweisem Permutations-p={pformat(primary.permutation_p)}. ROC-AUC: {primary.auc:.4f} '
         f'[{primary.auc_ci_low:.4f}; {primary.auc_ci_high:.4f}]. Der Effekt gegenüber 50 % beträgt '
         f'{100*primary.effect_above_chance:.1f} Prozentpunkte.'),
        performance_qualification,
        summary_qualification,
        '## Design und vorab festgelegter Endpunkt',
        (f'{n_blocks} unabhängige Initialisierungsseeds mit je zwei Historien: H=0 ist A→B→C, H=1 ist B→A→C. '
         'A verwendet Ziffernidentität, B `(y+1) mod 10`, C `(y+3) mod 10`; alle Aufgaben nutzen dieselben MNIST-Bilder und zehn Ausgänge. '
         'Architektur: 784–64–10, ReLU, feste Null-Biases. SGD mit Lernrate 0,1, Momentum 0,9 und Batchgröße 256; '
         'je eine vollständige A- und B-Epoche, anschließend zehn vollständige C-Epochen. Jede Epoche umfasst 235 Updates; '
         'gespeicherte C-Horizonte sind 1, 5 und 10. Beide Historien eines Seeds beginnen mit identischen W0/W1 und erhalten '
         'dieselbe Bildreihenfolge für denselben zeitlichen Epochenslot. In der Hauptkohorte wird Momentum an beiden Phasenübergängen zurückgesetzt.'),
        (f'Die Seed-Blöcke wurden in {n_train} Trainings-, {n_val} Validierungs- und {n_test} Testblöcke geteilt. '
         'Die L2-logistische Probe nutzt ausschließlich finale Gewichte am jeweiligen C-Horizont. '
         'Die kanonische ReLU-Gauge gleicht eingehende und ausgehende Neuronennormen an; Git Re-Basin richtet jede finale '
         'Gewichtskonfiguration auf eine eingefrorene, ausschließlich aus Trainingsgewichten bestimmte Referenz aus. '
         'Frühere Checkpoints, Momentum, Seed-Werte, Dateinamen und Historienmetadaten sind keine Gewichtsfeatures. '
         'Skalierung und Kovariatenabzug werden ausschließlich im Trainingssplit geschätzt; die Regularisierung wird auf '
         'dem blockweise getrennten Validierungssplit gewählt. Der Testsplit wird nicht für Modellwahl oder Referenzwahl verwendet.'),
        amendment_text,
        '## Primärer Test: Confusion Matrix',
        '| Wahre Historie | Vorhersage A→B→C | Vorhersage B→A→C |\n| --- | ---: | ---: |\n'
        f'| A→B→C | {int(primary.tn)} | {int(primary.fp)} |\n| B→A→C | {int(primary.fn)} | {int(primary.tp)} |',
        '## Persistenz nach gemeinsamem C-Training',
        temporal_text + '.',
        '![Dekodierbarkeit über die C-Trainingsdauer](history_decoding_by_C_duration.png)',
        ('C10 ist der einzige primäre Horizont. C1 und C5 sowie Schicht- und Baselinevergleiche sind sekundär. '
         'Die drei Zeitpunkte stammen aus denselben Trajektorien und sind keine unabhängigen Wiederholungen. '
         'Unterschiede der Punktschätzer allein belegen keinen signifikanten zeitlichen Anstieg oder Abfall.'),
        '## Schichten, Kovariaten und funktionales Verhalten',
        ('W0 und W1 sind hier Schicht-Feature-Ablationen **nach gemeinsamer Normalisierung und Ausrichtung beider Matrizen**. '
         'Sowohl die kanonischen Skalierungsfaktoren als auch die Re-Basin-Zuordnung hängen von W0 und W1 ab. '
         'Erfolg einer solchen Schichtprobe belegt deshalb keine unabhängige Dekodierbarkeit aus der jeweiligen rohen Matrix allein; '
         'der primäre Endpunkt verwendet ausdrücklich beide Matrizen.'),
        markdown_table(final, ['feature_name','balanced_accuracy','ba_ci_low','ba_ci_high','auc','permutation_p','secondary_holm_p'],
                       ['Probe','Bal. Accuracy','KI unten','KI oben','AUC','p','Holm-p sekundär'],
                       ['balanced_accuracy','ba_ci_low','ba_ci_high']),
        '\n\n'.join(control_lines),
        '![Schichtproben und Kontrollen nach zehn C-Epochen](history_probe_ablation.png)',
        '## Finale Aufgabe und Leistungsniveau',
        'Leistung auf dem unveränderten offiziellen MNIST-Testsplit, für die Historien der zurückgehaltenen Seed-Blöcke:',
        markdown_table(testperf, ['history','n','accuracy_mean','accuracy_sd','loss_mean','loss_sd'],
                       ['Historie','Netze','Accuracy Mittel','Accuracy SD','Loss Mittel','Loss SD'], ['accuracy_mean','accuracy_sd']),
        'Gepaarte Differenz H1 minus H0, mit Seed-Block-Bootstrap-Intervall:',
        markdown_table(paired_display, ['metric','diff_H1_minus_H0','ci_low','ci_high'],
                       ['Messgröße','Differenz H1−H0','KI unten','KI oben']),
        (performance_qualification + ' Ähnliche Mittelwerte oder ein Intervall, das null enthält, begründen keine formale Äquivalenz. '
         'Es wurde keine nachträgliche Auswahl von Netzen nach ihrer Testleistung vorgenommen. '
         'Die Diagnoseprobe und der Kovariatenabzug ergänzen diese Leistungsbeschreibung.'),
        '## Momentum- und Negativkontrolle',
        'Fortgeführtes Momentum, beide Gewichtsmatrizen: ' + continued_text + '.',
        (f'Zusätzlich zu den {2*n_blocks} primären Trainingsläufen wurden {2*n_blocks} Läufe mit fortgeführtem Momentum berechnet '
         f'({4*n_blocks} neue Läufe insgesamt). Die Momentumkontrolle verwendet dieselben Seed-Blöcke und ist eine gepaarte Robustheitsanalyse; '
         'sie zählt nicht als unabhängige Replikation.' if len(continued) else ''),
        negative_text,
        '## Symmetrie-Kontrolle',
        symmetry_table,
        ('Jede finale Gewichtskonfiguration wurde erneut nach zufälliger Hidden-Unit-Umordnung und nach positiver '
         'ReLU-Reskalierung ausgerichtet. Berichtet ist die maximale absolute Änderung der ausgerichteten Gewichtsfeatures. '
         'Identische Features liefern für dieselbe eingefrorene Probe identische Vorhersagen; Gauge-Abweichungen im Bereich '
         'der numerischen Rundung werden separat ausgewiesen.'),
        '## Inferenz und Reichweite',
        (f'Der primäre p-Wert basiert auf {int(primary.n_permutations)} Monte-Carlo-Permutationen. '
         'In jeder Permutation werden die Historienlabels unabhängig innerhalb vollständiger Seed-Paare getauscht, '
         'die logistische Probe neu angepasst und die Regularisierung erneut auf dem Validierungssplit gewählt. '
         'Labelunabhängige, bereits ausschließlich aus Trainingsdaten berechnete Transformationen können dabei identisch bleiben. '
         'Der einseitige p-Wert verwendet die Plus-eins-Korrektur; er misst den Anteil mindestens so hoher Test-Accuracy unter der Nullverteilung. '
         'Sekundäre Einzel-p-Werte werden zusätzlich innerhalb der deklarierten Sekundärfamilie nach Holm korrigiert; '
         'der primäre Test wird separat berichtet.'),
        ('95-%-Intervalle stammen aus dem Resampling vollständiger Test-Seed-Blöcke mit beiden Historien. '
         'Sie sind bedingt auf den fest trainierten und ausgewählten Klassifikator und erfassen nicht die zusätzliche '
         'Variation durch eine neue Trainingskohorte, einen neuen Split oder eine andere Referenz. ' + degenerate_text),
        ('Ein positiver Befund zeigt rekonstruierbare Information über genau diese beiden Aufgabenreihenfolgen in den finalen Gewichten. '
         'Er beweist weder die Rekonstruktion jedes SGD-Schrittes noch einen Attraktor. Eine erfolgreiche funktionale Probe '
         'zeigt darüber hinaus verbleibende Information in unnormalisierten Outputs auf den festgelegten Probe-Inputs. Ein negatives lineares '
         'Resultat schließt nichtlineare Dekodierbarkeit oder andere historische Labels nicht aus. '
         'Der Pilot ist keine unabhängige Bestätigungskohorte; bei kleinen oder instabilen Effekten ist eine weitere Kohorte erforderlich.'),
        '## Vollständige Endpunkte',
        markdown_table(all_results, ['condition','c_epochs','label','balanced_accuracy','auc','permutation_p','secondary_holm_p','n_permutations'],
                       ['Momentum','C-Epochen','Probe','Bal. Accuracy','AUC','p','Holm-p','Permutationen'], ['balanced_accuracy']),
        '## Reproduzierbare Dateien',
        execution_note,
        ('- `plan.json` und `probe_splits.json`: vorab eingefrorene Aufgaben, Seeds, Split und Analyseparameter.\n'
         '- `history_run_manifest.csv`: Trainingsläufe, Paarung, Initialisierungs- und Checkpointhashes.\n'
         '- `history_probe_results.csv`: sämtliche Endpunkte einschließlich Kontrollen und Intervallen.\n'
         '- `performance_summary.csv` und `performance_paired.csv`: finale Aufgabenleistung und gepaarte Differenzen.\n'
         '- `probe_diagnostics.json`: Referenz, Transformationen, Optimierung und Symmetrieprüfungen.\n'
         '- `checkpoints/`: neue Trainingsarchive; die ursprünglichen Reproduktionsarchive werden nur gelesen.\n'
         '- `code/training.py`, `code/probes.py`, `code/reporting.py`: ausführbarer Trainings-, Analyse- und Berichtscode.\n'
         '- `provenance/experiment_specification.md`: unveränderte Kopie der ausgeführten Experimentbeschreibung.'),
    ]
    report = '\n\n'.join(section for section in sections if section) + '\n'
    (out / 'history_decoding_report.md').write_text(report, encoding='utf-8')
    figures(out, results)
    (out / 'provenance' / 'report_receipt.json').write_text(json.dumps({
        'source': 'history_probe_results.csv', 'primary_endpoint_count': len(primary_rows),
        'endpoint_count': len(results), 'report_language': 'de',
        'figures': ['history_decoding_by_C_duration.png', 'history_decoding_by_C_duration.pdf',
                    'history_probe_ablation.png', 'history_probe_ablation.pdf'],
        'training_manifest_rows': None if manifest is None else len(manifest),
    }, indent=2) + '\n')
    return out / 'history_decoding_report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(make_report(args.output))
