"""German report of the fully crossed initialization/shuffle experiment."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd

WKEY='normalized_frobenius|total|optimal|canonical';CKEY='one_minus_cka|hidden|none|raw'

def f(x):
    if isinstance(x,(float,np.floating)):
        return '-' if not np.isfinite(x) else (f'{x:.3g}' if 0<abs(x)<1e-4 else f'{x:.4f}')
    return str(x).replace('|',' / ')
def table(frame,cols=None):
    if cols:frame=frame[cols]
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join('---' for _ in frame.columns)+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def make_report(out:Path):
    out=Path(out);init=pd.read_csv(out/'initialization_retrieval.csv');shuffle=pd.read_csv(out/'shuffle_retrieval.csv');traj=pd.read_csv(out/'initial_trajectory_retrieval.csv')
    primary=init[init.metric_key.eq(WKEY)];cka=init[init.metric_key.eq(CKEY)];contrasts=pd.read_csv(out/'distance_contrasts.csv');groups=pd.read_csv(out/'distance_groups.csv');perf=pd.read_csv(out/'performance_decomposition.csv');perf=perf[perf.checkpoint.eq(20)]
    manifest=json.loads((out/'data/manifest.json').read_text());assert manifest['complete'] and len(manifest['runs'])==150
    success=bool((primary.p_holm_less<.05).all() and (primary.mean_rank<5.5).all())
    conclusion=('Die Initialisierung ist aus den Endgewichten auch über unabhängig variierte Batch-Reihenfolgen hinweg in allen drei Datensätzen überzufällig identifizierbar.' if success else 'Die Identifikation der Initialisierung über verschiedene Batch-Reihenfolgen ist nicht in allen drei Datensätzen abgesichert.')
    cols=['dataset','mean_rank','mean_rank_ci_low','mean_rank_ci_high','top1','p_perm_less','p_holm_less']
    text=f'''# 05: Initialisierung und Batch-Reihenfolge getrennt variieren

{conclusion}

## Neues gekreuztes Training

150 vollständig neu trainierte Netze: drei Datensätze × zehn Anfangskonfigurationen × fünf Batch-Reihenfolgen. Jede der zehn Initialisierungen eines Datensatzes wurde mit jeder der fünf Reihenfolgen kombiniert. Die Initialgewichte sind bitgleich zu den ursprünglichen E0-Checkpoints. Identische Shuffle-Faktoren erzeugen identische Epochenreihenfolgen über alle Initialisierungen; die fünf Seed-Bereiche überlappen nicht.

Die fünf Shuffle-Baseseeds sind 11001, 12001, 13001, 14001 und 15001; pro Epoche wird die nullbasierte Epochenzahl addiert. Architektur 784-64-10, ReLU, feste Null-Biases, SGD mit Lernrate 0,1, Momentum 0,9, Batchgröße 256, zehn Epochen. Alle 3.150 Checkpoints inklusive Momentum, Modell-/Optimierer-/RNG-Zuständen und konkreter Datenreihenfolge sind gespeichert. Es gibt eine Beobachtung pro 10×5-Zelle, keine zusätzlichen Wiederholungen innerhalb einer Zelle.

## Primär: gleiche Initialisierung über verschiedene Reihenfolgen erkennen

Für jedes gerichtete Paar verschiedener Shuffle-Level wird jedes Endnetz mit den zehn Endnetzen des anderen Levels verglichen. Der richtige Kandidat teilt dessen Initialisierung. Alle 20 gerichteten Shuffle-Paarungen werden ausgewertet; zunächst wird je Initialisierung über diese Vergleiche gemittelt, anschließend über die zehn Initialisierungen. Primär ist die kanonische, gemeinsam auf Eins normierte W0/W1-Gewichtsdistanz nach optimalem gemeinsamem Matching. Zufall: mittlerer Rang 5,5, Top-1=10%.

{table(primary,cols)}
Die drei primären Datensatztests werden mit Holm korrigiert. Ein Treffer ist hier **Endnetz gegen Endnetz über verschiedene Trainingsreihenfolgen**, nicht der Anfangs-Ende-Test aus Experiment 02. Damit wird der Initialisierungsfaktor bei variiertem Shuffle untersucht.

## Sekundär: Endrepräsentationen

Die gleiche Faktoridentifikation wurde vor der Trainingsauswertung sekundär für Hidden-CKA festgelegt. Alle 50 Endnetze eines Datensatzes werden auf demselben gehashten 3.000-Bilder-Probe ausgewertet. Kosten sind 1 minus zentrierte lineare CKA; es gibt keine Neuronenzuordnung.

{table(cka,['dataset','mean_rank','mean_rank_ci_low','mean_rank_ci_high','top1','p_perm_less'])}
Alle drei Datensätze zeigen hier überzufällige Erkennung (97 %, 45,5 % und 87 % Top-1). Die gemeinsame Initialisierung prägt damit die finale Hidden-Repräsentationsstruktur trotz unterschiedlicher Batch-Reihenfolgen. Ein entsprechender Fingerabdruck in finalen Logits oder Vorhersagen wurde hier nicht getestet.

Dies ist eine andere Frage als Experiment 04: Hier werden **Endrepräsentationen unter unterschiedlichen Reihenfolgen** verglichen. Eine fehlende direkte Wiedererkennung der Anfangsrepräsentation schließt einen Initialisierungseinfluss auf die später gelernten Repräsentationen nicht aus. Die CKA-Ergebnisse bleiben sekundär/explorativ.

## Sekundär: gleiche Reihenfolge über verschiedene Initialisierungen erkennen

Umgekehrt vergleichen wir alle 90 gerichteten Paare verschiedener Initialisierungen. Jeder Query hat fünf Endnetze der anderen Initialisierung als Kandidaten; der richtige Kandidat teilt dessen Shuffle-Level. Zufall: mittlerer Rang 3, Top-1=20%. Gewichte und CKA werden separat ausgewertet.

{table(shuffle,['dataset','metric','mean_rank','mean_rank_ci_low','mean_rank_ci_high','top1','p_perm_less'])}
Diese Auswertung fragt nach einem über verschiedene Initialisierungen wiederkehrenden Fingerabdruck der Reihenfolge. Ein schwacher Identifikationsbefund bedeutet nicht, dass die Reihenfolge keine Auswirkungen auf ein einzelnes Training hat.

## Statistische Abhängigkeiten

Pro Datensatz werden 10.000 konsistente blockierte Labelpermutationen gerechnet. Für Initialisierungs-Retrieval wird innerhalb jedes Shuffle-Levels die Zuordnung von Initialisierungslabels gemeinsam für alle zugehörigen Vergleiche permutiert. Der erste Shuffle-Block bleibt fix, weil eine gemeinsame globale Umbenennung aller Initialisierungen die Statistik nicht verändern würde. Alle anderen Blockzuordnungen sind unabhängig. Bei Shuffle-Retrieval werden analog Shuffle-Labels innerhalb der Initialisierungsblöcke permutiert; der erste Init-Block bleibt fix.

Die primäre Unsicherheit löscht eine komplette Initialisierungsgruppe mit allen fünf Trainingsläufen und rangiert mit neun Initialisierungskandidaten neu. Sie ist damit konditional auf die fünf gewählten Shuffle-Level. Die Shuffle-Auswertung löscht jeweils eine komplette Shuffle-Gruppe und ist konditional auf die zehn Initialisierungen. Die 50 Netze sind keine 50 unabhängigen Initialisierungen, die vielen Paarvergleiche keine unabhängigen Replikate. Normierte Rangscores korrigieren die veränderte Kandidatenzahl beim Jackknife. Leave-one-Level-Ergebnisse für den jeweils anderen Faktor stehen separat als Robustheitskontrolle, ohne zusätzlichen Hypothesentest.

Retrieval-p-Werte testen einseitig auf kleinere korrekte Ränge. Alle p-Werte verwenden die Plus-eins-Monte-Carlo-Regel; die Untergrenze ist 1/10001 (gerundet 0,00010). Nur die drei gewichtsbasierten Initialisierungs-Endpunkte sind primär; Shuffle-, CKA-, Trajektorien- und Distanzkontrasttests sind sekundär/explorativ. Normale 95%-Jackknife-Intervalle sind bei zehn bzw. fünf Faktorstufen begrenzt präzise. Die ungekürzten Tabellenintervalle können theoretische Ranggrenzen überschreiten; in Abbildungen werden sie auf den möglichen Wertebereich begrenzt. Bei perfekter empirischer Erkennung können sie zu einem Punkt kollabieren, ohne perfekte Erkennung neuer Netze zu garantieren.

## Geometrische Kontraste

Pro Datensatz gibt es 100 ungeordnete Paare mit gleicher Initialisierung und unterschiedlichem Shuffle, 225 mit unterschiedlicher Initialisierung und gleichem Shuffle sowie 900 mit beiden Faktoren unterschiedlich. Die Kontraste vergleichen jeweils die Distanz bei beiden unterschiedlichen Faktoren mit jener bei gleichem Zielfaktor. Positive Werte sprechen für geringere Distanzen bei gemeinsamem Faktor. Die Nullverteilungen bewahren dieselbe Blockstruktur wie die Retrievaltests.

{table(contrasts,['dataset','metric','factor','same_factor_mean','different_factor_mean','contrast','ci_low','ci_high','p_perm_greater'])}

## Testleistung und deskriptive Zerlegung

Die vollständigen Accuracy-/Loss-Tabellen enthalten jede 10×5-Zelle an allen 21 Checkpoints. Die folgende Tabelle zeigt E20-Mittelwerte über alle 50 Läufe pro Datensatz und zerlegt die beobachtete Quadratsumme in additive Initialisierungs-, additive Shuffle- und Interaktions-/Restanteile:

{table(perf,['dataset','metric','grand_mean','initialization_fraction','shuffle_fraction','interaction_residual_fraction'])}
Die Freiheitsgrade betragen 9 für Initialisierung, 4 für Shuffle und 36 für Interaktion/Rest. Größere Quadratsummenanteile allein sind kein unverzerrter Vergleich zufälliger Varianzkomponenten und kein Signifikanznachweis. Mit nur einer Beobachtung pro Zelle können Interaktion und ein zusätzlicher Restfehler nicht separat geschätzt werden. Deshalb werden keine ANOVA-F-Tests oder zugehörigen p-Werte behauptet. Identifizierbare Gewichts- oder Repräsentationsunterschiede sind nicht automatisch große Unterschiede in der Testgenauigkeit.

## Wiedererkennung der originalen Initialgewichte über das Training

Zusätzlich wird jedes der 50 aktuellen Netze eines Datensatzes zu jedem Checkpoint mit den zehn Originalinitialisierungen verglichen. Wiederholungen derselben Initialisierung über die fünf Shuffles werden gemeinsam behandelt. Endpunkt:

{table(traj[traj.checkpoint.eq(20)],['dataset','mean_rank','mean_rank_ci_low','mean_rank_ci_high','top1','p_perm_less'])}

## Grenzen und Dateien

Das vollständige gekreuzte Design trennt Initialisierung und Reihenfolge innerhalb dieses festen Trainingsprotokolls. Die Aussagen bleiben auf die gewählten Datensätze, Architektur, zehn Anfangszustände und fünf Reihenfolgen beschränkt. Es wird keine vollständige Rekonstruktion einer Trainingsgeschichte oder ein dynamischer Attraktor nachgewiesen.

- [Alle Ergebnisse als PDF](results.pdf), [Figurengalerie](index.html).
- `data/manifest.json`: 150 Läufe und Quell-/E0-/Shuffle-Hashes. `data/checkpoints/`: NPZ und PT mit allen 3.150 Checkpoints.
- `initialization_retrieval.csv`, `shuffle_retrieval.csv`, `aggregate.csv`: Faktoridentifikation und Unsicherheit.
- `initial_trajectory_retrieval.csv`, `distance_groups.csv`, `distance_contrasts.csv`, `performance_decomposition.csv`: ergänzende Auswertungen.
- `data/*_distances.npz`: alle Endzustandsdistanzen, CKA, Anfangs-Kandidatenkosten und Matchings. `data/*_inference.npz`: komplette blockierte Nullzuordnungen, Nullstatistiken und Jackknife-Replikate.
- `analysis_plan.json` und `analysis_design.json`: vorab dokumentierte Trainings- und Auswertungsentscheidungen; `provenance/`: Tests, unabhängige Reviews, Validierungen und Prüfsummen.

Reproduktion: `.venv/bin/python 05_Initialization_Shuffle/code/run.py`. Vorhandene vollständige Trainingsarchive werden nur nach strenger Validierung wiederverwendet. Frühere Experimente bleiben erhalten.
'''
    (out/'summary.md').write_text(text,encoding='utf-8')
    figures=[('factorial_training','Alle Trainingszellen'),('endpoint_retrieval_matrices','Endzustandsgeometrie der 50 Netze'),('factor_retrieval_results','Identifikation beider Faktoren'),('initial_trajectory_retrieval','Initialisierungs-Retrieval über das Training'),('factor_geometry_and_performance','Geometrie und Testleistung')]
    cards=''.join(f'<section><h2>{html.escape(label)}</h2><a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(label)}"></a><p><a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">PDF</a></p></section>' for stem,label in figures)
    (out/'index.html').write_text(f'''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>05: Initialisierung und Shuffle</title><style>body{{font:17px/1.55 system-ui;max-width:1300px;margin:35px auto;padding:0 22px;background:#f5f7fa;color:#243448}}header,section{{background:white;padding:25px;border:1px solid #dce2e8;border-radius:12px;margin:20px 0}}img{{width:100%;height:auto}}a{{color:#215d96}}</style><header><h1>Initialisierung und Batch-Reihenfolge getrennt variieren</h1><p>{html.escape(conclusion)}</p><p>150 neue Trainingsläufe · 3 Datensätze · 10 Initialisierungen × 5 Reihenfolgen · 3.150 gespeicherte Checkpoints</p><p><a href="results.pdf">Ergebnisbericht als PDF</a> · <a href="summary.md">Vollständiger Bericht</a> · <a href="analysis_design.json">Analyseplan</a> · <a href="initialization_retrieval.csv">Primäre Ergebnistabelle</a></p></header>{cards}</html>''',encoding='utf-8')
    (out/'README.md').write_text('# 05 Initialization x Shuffle\n\n[Ergebnisbericht als PDF](results.pdf) · [Vollständiger Bericht](summary.md) · [Figuren](index.html)\n\n150 neue Trainingsläufe mit vollständigen Checkpoints. Primär: Initialisierungsidentifikation zwischen verschiedenen Shuffle-Leveln; ergänzend Shuffle-Identifikation, CKA und deskriptive Leistungszerlegung.\n',encoding='utf-8')
