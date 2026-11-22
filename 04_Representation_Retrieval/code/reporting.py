"""German summary and local figure gallery from computed representation results."""
from pathlib import Path
from html import escape
import pandas as pd

DATASETS=('mnist','fashion_mnist','kmnist')
NAMES={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST'}
KEYS=('one_minus_cka|hidden|none|raw','one_minus_cka|hidden|none|standardized','one_minus_cka|logits|none|raw')
LABELS=('Rohe Hidden-CKA (primär)','Standardisierte Hidden-CKA','Logit-CKA')
FIGURES=(('representation_endpoint_matrices','Kandidatenränge am Trainingsende'),('representation_over_training','Verlauf über alle 21 Checkpoints'),('representation_endpoint_comparison','Vergleich der drei Repräsentationen'))


def make_reports(out:Path):
    out=Path(out).resolve();frame=pd.read_csv(out/'retrieval_by_dataset.csv');aggregate=pd.read_csv(out/'aggregate.csv')
    endpoint=frame[frame.checkpoint.eq(20)];endagg=aggregate[aggregate.checkpoint.eq(20)]
    primary=endpoint[endpoint.metric_key.eq(KEYS[0])].set_index('dataset')
    agg=endagg[endagg.metric_key.eq(KEYS[0])].iloc[0]
    rows=[]
    for dataset in DATASETS:
        r=primary.loc[dataset]
        rows.append(f'| {NAMES[dataset]} | {r.mean_rank:.1f} | [{r.mean_rank_ci_low:.3f}, {r.mean_rank_ci_high:.3f}] | {r.mean_rank_p_perm_less:.6f} | {r.p_holm_less:.6f} | {r.top1:.0%} |')
    alternatives=[]
    for key,label in zip(KEYS,LABELS):
        r=endagg[endagg.metric_key.eq(key)].iloc[0]
        alternatives.append(f'| {label} | {r.mean_rank:.3f} | [{r.mean_rank_ci_low:.3f}, {r.mean_rank_ci_high:.3f}] | {r.mean_rank_p_perm_less:.6f} | {r.top1:.2%} | {r.mrr:.4f} |')
    text=f'''# Punkt 3: Initialisierungs-Retrieval aus Repräsentationen

Die vorab festgelegte rohe Hidden-CKA erkennt am Ende des Trainings die zugehörige Initialisierung **nicht konsistent besser als die Zufallszuordnung**. Der gleichgewichtete mittlere Rang über die drei Datensätze beträgt {agg.mean_rank:.3f} (95%-Run-Jackknife-Intervall [{agg.mean_rank_ci_low:.3f}, {agg.mean_rank_ci_high:.3f}], einseitiges Permutations-p={agg.mean_rank_p_perm_less:.6f}); Top-1 beträgt {agg.top1:.2%}. Das ist ein ausbleibender Nachweis in diesem Test, kein Äquivalenztest und kein Beweis, dass die Repräsentationen keinerlei historische Information enthalten.

## Festgelegtes Experiment

Es wurden ausschließlich die 30 bestehenden Netze der Hauptkohorte verwendet: zehn je MNIST, Fashion-MNIST und KMNIST, jeweils alle 21 gespeicherten Halbepochen-Checkpoints E=0…20. Für jedes Dataset und jeden Checkpoint wurden sämtliche 100 Vergleiche zwischen aktueller Query i und Initialisierungskandidat j bei E=0 berechnet. Kein neues Training, keine Gewichtsraumdistanz und keine Neuronenzuordnung gingen in diese Auswertung ein.

Die gemeinsame gehashte Probe enthält 3.000 Beispiele, 1.000 pro Dataset, in unveränderter Reihenfolge. Primär wird 1−CKA der rohen Hidden-Aktivierungen verglichen. Die biased centered linear CKA ist

`CKA(X,Y) = ||XcᵀYc||²_F / (||XcᵀXc||_F · ||YcᵀYc||_F)`.

Die Zentrierung erfolgt über die Probezeilen jeder Feature-Spalte. Zwei vorab festgelegte Robustheiten verwenden (a) zusätzlich jede nichtkonstante Hidden-Spalte mit Populations-SD=1 und (b) rohe Logits `H @ W1`, deren feste Null-Biases geprüft wurden. Konstante Spalten bleiben null; eine vollständig varianzlose Repräsentation führt zu einem Fehler. Rohe Hidden-CKA ist nicht invariant gegen beliebige positive Skalierung einzelner Hidden-Neuronen; die Standardisierungsvariante kontrolliert genau diese Empfindlichkeit. Logits sind unveränderte Ausgabewerte, keine Softmax-Wahrscheinlichkeiten.

## Primäre Endpunkte bei E=20

Kleinere Ränge sind besser. Bei zehn Kandidaten beträgt der Zufallsmittelrang 5,5 und die Top-1-Chance 10%.

| Dataset | Mittlerer Rang | 95%-Run-JK-Intervall | p, Rang kleiner | Holm-p | Top-1 |
|---|---:|---|---:|---:|---:|
{chr(10).join(rows)}

Keine der drei primären Dataset-Hypothesen besteht die Holm-Korrektur. E=0 liefert Selbst-Retrieval mit Rang 1 und ist lediglich ein Sanity-Check ohne p-Werte. Die Verlaufsgrafik zeigt die Abnahme der Identifizierbarkeit nach Beginn des Trainings; andere Checkpoints wurden nicht zu neuen primären Endpunkten erklärt.

## Aggregierte Robustheiten

Die Zusammenfassung mittelt die drei Dataset-Statistiken gleichgewichtet. Die Zusatzvarianten und Top-1/MRR sind explorativ; die unadjustierten p-Werte dieser Tabelle bilden keine zusätzliche bestätigende Testfamilie.

| Repräsentation | Mittlerer Rang | 95%-Run-JK-Intervall | p, Rang kleiner | Top-1 | MRR |
|---|---:|---|---:|---:|---:|
{chr(10).join(alternatives)}

Weder die standardisierte Hidden-CKA noch die Logit-CKA liefern hier einen konsistenten Vorteil über die Datensätze. Die detaillierten Einzelwerte und Verläufe bleiben in den CSV-Dateien erhalten; es wurde keine Variante anhand eines positiven Ergebnisses ausgewählt.

## Statistische Einheit und Grenzen

Die Netzwerke sind die Replikationseinheiten, nicht die 100 Matrixzellen. Je Dataset wurden 10.000 vollständige bijektive Zuordnungen der Kandidatenidentitäten getestet, mit den festen Seeds 2026112001–2026112003. Die Nullzuordnung verwendet jeweils dieselbe globale Kandidatenpermutation für alle Query-Zeilen. Der gerichtete primäre Test fragt nach kleinerem mittleren Rang. Holm korrigiert die drei rohen Hidden-Endpunkte bei E=20. Unsicherheit stammt aus dem gepaarten Delete-one-Run-Jackknife: Query-Zeile und zugehörige Kandidatenspalte werden gemeinsam entfernt, die Kandidatenränge neu berechnet und für die neun verbleibenden Kandidaten kalibriert. Die Aggregation verwendet unabhängige Permutationen und getrennte Jackknife-Zentrierung je Dataset.

CKA misst die Ähnlichkeit der Repräsentationen auf dieser festen Probe. Eine erfolgreiche Erkennung würde weder beweisen, dass historische Information für die Klassifikation genutzt wird, noch eine kausale Rolle der Initialisierung identifizieren. Das hier ausbleibende Retrieval schließt Information in anderen Merkmalen, Proben oder Verfahren nicht aus. Der Kandidatenraum umfasst nur zehn bekannte Initialisierungen pro Dataset; dies ist keine Rekonstruktion einer unbekannten Initialisierung. Die vorhandene Hauptkohorte koppelt Initialisierungs- und Trainings-/Shuffle-Zufall über die ursprünglichen Run-Seeds und isoliert diese Ursachen nicht experimentell. Die spätere faktorielle Untersuchung ist davon getrennt.

## Nachvollziehbare Artefakte

- [Vorab festgelegte Definitionen](analysis_plan.json), [Distanzdefinitionen](data/distance_definitions.json), [Quell- und Rechenvalidierung](data/validation.json).
- [Dataset-Ergebnisse](retrieval_by_dataset.csv), [Aggregation](aggregate.csv), [einzelne Query-Ränge](per_query_ranks.csv), [Inferenzdiagnostik](inference_diagnostics.json).
- [Berechnung](code/representations.py), [Tests](tests/test_representations.py), [unverändert wiederverwendete Inferenzquelle](provenance/inference_source.json).
- [Grafikübersicht](index.html). PNG und PDF zeigen dieselben berechneten Werte; Intervalle werden nur zur Darstellung auf zulässige Wertebereiche begrenzt, die unveränderten Grenzen stehen in den CSVs.
'''
    (out/'summary.md').write_text(text)
    panels=''.join(f'<section><h2>{escape(title)}</h2><a href="{stem}.png"><img src="{stem}.png" alt="{escape(title)}"></a><p><a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">PDF</a></p></section>' for stem,title in FIGURES)
    primary_table=''.join(f'<tr><td>{NAMES[d]}</td><td>{primary.loc[d].mean_rank:.1f}</td><td>{primary.loc[d].p_holm_less:.4f}</td><td>{primary.loc[d].top1:.0%}</td></tr>' for d in DATASETS)
    html=f'''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Punkt 3 · Repräsentations-Retrieval</title><style>body{{font:16px/1.5 system-ui,sans-serif;background:#f4f6f9;color:#182430;margin:0}}main{{max-width:1400px;margin:auto;padding:30px}}section,header{{background:white;border:1px solid #dce2e8;border-radius:10px;padding:24px;margin-bottom:22px}}h1{{margin:0 0 14px}}h2{{font-size:21px}}img{{width:100%;height:auto}}a{{color:#185a91}}table{{border-collapse:collapse}}th,td{{padding:8px 22px;border-bottom:1px solid #dce2e8;text-align:left}}.muted{{color:#526070}}nav a{{display:inline-block;margin-right:18px}}</style><main><header><h1>Initialisierungs-Retrieval aus Repräsentationen</h1><p>Die vorab festgelegte rohe Hidden-CKA liefert am Trainingsende keinen konsistenten Erkennungsnachweis. Aggregiert: Rang {agg.mean_rank:.3f}, p={agg.mean_rank_p_perm_less:.4f}, Top-1 {agg.top1:.2%}. Zufallsreferenz: Rang 5,5 und Top-1 10%.</p><table><thead><tr><th>Dataset</th><th>Rang</th><th>Primäres Holm-p</th><th>Top-1</th></tr></thead><tbody>{primary_table}</tbody></table><p class="muted">30 bestehende Netze, 21 Checkpoints, eine feste Probe mit 3.000 Beispielen. Keine Neuronenzuordnung oder Gewichtsraumdistanz. Ein fehlender Nachweis beweist keine Abwesenheit historischer Information.</p><nav><a href="summary.md">Deutscher Bericht</a><a href="analysis_plan.json">Analyseplan</a><a href="data/distance_definitions.json">Definitionen</a><a href="data/validation.json">Validierung und Quellenhashes</a><a href="retrieval_by_dataset.csv">Dataset-CSV</a><a href="aggregate.csv">Aggregierte CSV</a><a href="inference_diagnostics.json">Statistik</a></nav></header>{panels}</main></html>'''
    (out/'index.html').write_text(html)
    return out/'summary.md',out/'index.html'


if __name__=='__main__':
    for path in make_reports(Path(__file__).resolve().parents[1]):print(path)
