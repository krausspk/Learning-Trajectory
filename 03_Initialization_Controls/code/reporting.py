"""Report coordinate, layer, norm, residual and query-activity controls."""
from pathlib import Path
import html,json
import numpy as np
import pandas as pd

DATASETS=('mnist','fashion_mnist','kmnist')
NAMES={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST'}
PRIMARY=dict(metric='normalized_frobenius',layer='total',alignment='optimal',gauge='canonical')


def select(frame,**criteria):
    for field,value in criteria.items():frame=frame[frame[field].eq(value)]
    return frame


def fmt(x):
    if isinstance(x,(float,np.floating)):
        if not np.isfinite(x):return '-'
        return f'{x:.3g}' if 0<abs(x)<1e-4 else f'{x:.4f}'
    return str(x).replace('|',' / ')


def table(frame,columns=None):
    frame=frame if columns is None else frame[columns]
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join('---' for _ in frame.columns)+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def make_report(out:Path):
    out=Path(out)
    main=pd.read_csv(out/'retrieval_by_dataset.csv')
    ranks=pd.read_csv(out/'norm_rank_controls/retrieval_by_dataset.csv')
    end=main[main.checkpoint.eq(20)].copy()
    rankend=ranks[ranks.checkpoint.eq(20)].copy()
    all_end=pd.concat([end.assign(analysis_family='planned_controls'),rankend.assign(analysis_family='posthoc_norm_rank')],ignore_index=True)
    all_end.to_csv(out/'all_control_endpoints.csv',index=False)
    mask=pd.read_csv(out/'data/query_mask_counts.csv')
    maskend=mask[mask.checkpoint.eq(20)].groupby(['dataset','subset']).selected_neurons.agg(['min','mean','max']).reset_index()
    changes=pd.read_csv(out/'data/historical_weight_changes_descriptive.csv')
    changeend=changes[changes.checkpoint.eq(20)].groupby(['dataset','layer']).agg(n_runs=('run_id','size'),median_abs_change_mean=('abs_change_p50','mean'),p75_abs_change_mean=('abs_change_p75','mean'),relative_frobenius_change_mean=('relative_frobenius_change','mean'),fraction_exactly_unchanged_mean=('fraction_exactly_unchanged','mean'),fraction_change_le_1e_minus6_mean=('fraction_abs_change_le_1e_minus6','mean')).reset_index()
    norms=select(end,metric='norm_only')
    basis=end[end.metric.isin(['frobenius','normalized_frobenius']) & end.layer.eq('total')]
    layers=end[end.metric.eq('normalized_frobenius') & end.gauge.eq('canonical')]
    residual=end[end.metric.str.contains('residual')]
    activity=end[end.metric.str.startswith('activity_')]
    reference=select(end,**PRIMARY)
    inactive=select(mask,checkpoint=20,subset='inactive').selected_neurons
    diagnostic=json.loads((out/'inference_diagnostics.json').read_text())
    val=json.loads((out/'data/validation.json').read_text())
    min_top1=reference.top1.min()
    title='Die vollständige Gewichtskonfiguration bleibt über Normierung und Neuronenumordnung hinweg wiedererkennbar.' if min_top1==1 else 'Die Kontrollen zeigen, unter welchen Gewichtsdarstellungen die Herkunft erkennbar bleibt.'
    norm_rank_total=select(rankend,layer='total')
    rank_aggregate=select(pd.read_csv(out/'norm_rank_controls/aggregate.csv'),checkpoint=20,layer='total')
    rank_summary='; '.join(f"{NAMES[r.dataset]} ({r.gauge}): Rang {r.mean_rank:.2f}, p={r.p_perm_less:.4f}" for r in norm_rank_total.itertuples())
    summary=f'''# 03: Kontrollen zur Wiedererkennung der Initialisierung

{title}

## Zentrale Beobachtungen

Das Referenzprotokoll aus Punkt1 wird exakt reproduziert: alle 30 Endnetze erkennen ihre Initialisierung als nächsten Kandidaten. Auch rohe Gesamtgewichte und die Eingangsschicht W0 erreichen in den geprüften Basisprotokollen perfekte Wiedererkennung. Die Schicht W1 verhält sich dagegen stark abhängig vom Matching: In den gespeicherten Neuronenkoordinaten bleibt sie identifizierbar; nach gemeinsamer kanonischer W0/W1-Zuordnung ist ihre eigene Initialisierung häufig weit hinten. **Daraus folgt keine generelle Löschung der Initialisierungsinformation in W1.**

{table(select(layers,alignment='optimal'),['dataset','layer','mean_rank','top1','p_perm_less'])}
Die Kontrolle mit häufig aktiven Neuronen liefert weiterhin starke Wiedererkennung der Gesamtgewichte. Ebenso bleibt sie bei den großen kandidatbezogenen Residuen erhalten. Kleine Residuen oder geringe Aktivität allein erklären das Muster deshalb unter diesen konkreten Kostenprotokollen nicht. Die Kontrollen sind keine kausale Zerlegung der gespeicherten Information.

## Umfang und Statistik

42 vor der Kontrollrechnung festgelegte Varianten × 21 Checkpoints × drei Datensätze; jeweils zehn Query-Netze und zehn Initialisierungen. Das sind 264.600 Distanzwerte und 2.646 Datensatz-Schätzer. Zusätzlich sechs nachträglich ergänzte Normrang-Kontrollen (378 Schätzer), unten begründet. Hauptkohorte und Checkpoints bleiben identisch zu Punkt1. Es wurde nicht neu trainiert und kein funktionaler Aktivierungs-Fingerprint zwischen Anfang und Ende berechnet.

Alle Kontrolltests sind **explorativ**. 10.000 globale bijektive Herkunftslabel-Permutationen je Datensatz, Seeds2026111801/02/03 für die 42 Kontrollen, Seeds2026111901/02/03 für die Zusatzkontrolle. Die Mittelwerte der Ränge werden einseitig gegen kleinere Werte getestet. Es gibt hier keine neue konfirmatorische Holm-Familie; Einzel-p-Werte über viele Varianten sind keine unabhängigen Bestätigungen. E0 bleibt eine Sanity-Prüfung ohne p-Werte.

Die Unsicherheit stammt aus dem gepaarten Netzwerk-Jackknife: Query-Zeile und Initialisierungsspalte desselben Runs entfernen, verbleibende Kandidaten neu rangieren, gegen die bei neun Kandidaten geänderte Zufallserwartung normieren und auf die Zehn-Kandidaten-Skala zurücktransformieren. Die Null-Mappings, Einzelquery-Ergebnisse, Nullarrays und Jackknife-Replikate sind gespeichert. **Bei perfekter Wiedererkennung können die Jackknife-Intervalle zu [1,1] kollabieren; das garantiert keine perfekte Wiedererkennung neuer Netze.**

## Koordinaten, Gauge, Normierung

Vier Protokolle: rohe bzw. kanonische Gewichte, jeweils gespeicherte Koordinaten (`identity`) oder gemeinsames optimales W0/W1-Matching (`optimal`). Zwei Kosten: Frobenius sowie normalisierte Frobenius-Distanz. Bei letzterer normiert ein **gemeinsamer Gesamtfaktor beide Schichten**; layerweise Werte sind Beiträge zu diesem Gesamtvektor, keine separat auf Eins normierten Layer. Die Neuronenzuordnung optimiert beide Schichten gemeinsam und bleibt bei ihrer separaten Auswertung dieselbe. `identity` ist ein Koordinatengedächtnis-Diagnostikum und nicht permutationsinvariant.

{table(basis,['dataset','metric','alignment','gauge','mean_rank','top1','p_perm_less'])}
Damit bleibt die Wiedererkennung der Gesamtgewichte auch nach Ausschalten der globalen Norm und Neuronenumordnung erhalten. Der Befund bezieht sich auf die zehn Kandidaten dieses Datensatzes, nicht auf die vollständige Rekonstruktion unbekannter Initialgewichte.

## Schichtkontrollen

{table(layers,['dataset','layer','alignment','mean_rank','top1','p_perm_less'])}
Die kanonische gemeinsame Zuordnung kann für die Gesamtgeometrie sinnvoll sein und zugleich die in W1 allein erkennbare Koordinatenkorrespondenz verändern. Das W1-Ergebnis ist deshalb **bedingt auf dieses Matching**. Hier wurde kein nachträgliches W1-Spezialmatching optimiert, um die Erfolgsquote zu erhöhen.

## Normen: ursprüngliche und ergänzende Kontrolle

Die geplante Normkontrolle verwendet |Norm(Query E20) − Norm(Kandidat E0)|, getrennt nach W0/W1/Gesamt und roh/kanonisch.

{table(norms,['dataset','layer','gauge','mean_rank','top1','p_perm_less'])}
**Grenze dieser Regel:** Wenn alle Endnormen über allen Anfangsnormen liegen, erhält jeder Query dieselbe Kandidatenrangfolge. Bei einer bijektiven Herkunftszuordnung ist der mittlere Rang dann konstruktiv 5,5. Dieses Ergebnis allein kann keine allgemeine Abwesenheit von Norminformation belegen.

Nach Erkennen dieser Degeneration wurde deshalb eine zusätzliche, explizit **nachträgliche explorative Normrang-Kontrolle** gerechnet: Innerhalb jedes Checkpoints und Datensatzes werden die zehn Normen rangiert; der Kostenwert ist |Normrang(Query Et) − Normrang(Kandidat E0)|/9. Das entfernt gemeinsame Änderungen des Normniveaus und prüft die Erhaltung der Normrangfolge. Die Kalibrierung verwendet keine Herkunftslabels, ist aber transduktiv über das gesamte Ensemble. Beim Jackknife werden die Normränge unter den verbleibenden neun Netzen **neu kalibriert**; es werden nicht einfach Zeilen/Spalten aus den Zehn-Netz-Kosten entfernt.

{table(rankend,['dataset','layer','gauge','mean_rank','top1','p_perm_less'])}
Gesamtnorm-Rangkontrollen: {rank_summary}. Auch diese Tests sind explorativ und keine Auswahl eines neuen primären Endpunkts. Quellen, Motivation und eigene Seeds stehen in `norm_rank_controls/analysis_note.json`.

Gleichgewichtete Zusammenfassung der Normrang-Kontrollen:

{table(rank_aggregate,['gauge','mean_rank','top1','p_perm_less'])}
Die Normrang-Kontrolle reproduziert die nahezu perfekte Wiedererkennung der vollständigen Gewichte nicht. Einzelne unadjustierte Layer-Befunde sind bei dieser Anzahl explorativer Tests vorsichtig zu lesen; aus ihnen folgt kein konsistenter Normeffekt über alle drei Datensätze. Die kanonischen W0-, W1- und Gesamtnormen sind durch die Normangleichung mathematisch proportional und keine drei unabhängigen Kontrollen.

## Kleine und große Residuen

Nach kanonischer Gauge, gemeinsamer Gesamtnormierung und derselben optimalen Paarzuordnung werden **für jedes Kandidatenpaar separat** die absoluten Gewichtsresiduen berechnet. Pro Schicht wird das untere bzw. obere Viertel ausgewählt, einschließlich aller Grenzwertbindungen. Kosten sind RMS auf den ausgewählten Einträgen; Gesamtwerte poolen beide Layer. Schwellen und Mengen sind für alle Kandidatenpaare gespeichert.

{table(residual,['dataset','metric','layer','mean_rank','top1','p_perm_less'])}
Das ist eine Sensitivität gegenüber verschiedenen Teilen der Residuenverteilung. Diese Residuen sind **nicht die tatsächlichen historischen Trainingsänderungen**: Das Matching und die Normierung verändern den Vergleich, und bei fremden Kandidaten gibt es keine entsprechende gemeinsame Trainingsgeschichte. Die Auswertung ist auch keine Intervention, die schwach veränderte Parameter kausal aus dem Netz entfernt.

Die tatsächlichen Änderungen der gespeicherten Rohgewichte vom eigenen E0 zu E20 wurden separat deskriptiv ausgewertet; folgende Werte sind Mittel über zehn Runs je Datensatz:

{table(changeend)}
Kein gespeicherter Gewichtseintrag blieb am Endpunkt exakt unverändert. Die mittlere Frobenius-Bewegung je Datensatz und Schicht beträgt das {changeend.relative_frobenius_change_mean.min():.2f}- bis {changeend.relative_frobenius_change_mean.max():.2f}-Fache der jeweiligen Anfangsnorm. Die Wiedererkennung beruht damit nicht auf einem Bestand exakt eingefrorener Parameter. Das allein quantifiziert noch nicht, wie viel Information kleine gegenüber großen Änderungen tragen.

Diese historische Tabelle wird weder für Kandidatenkosten noch für Masken oder inferenzielle p-Werte verwendet. Damit erfährt der Wiedererkennungstest aus ihr nicht, welcher Kandidat der richtige ist.

## Query-Aktivität und inaktive Neuronen

Für jedes Query-Netz und jeden Checkpoint wird auf dem bereits vorhandenen festen Probe mit 3.000 Bildern pro Hidden-Neuron der Anteil positiver ReLU-Aktivierungen bestimmt. Das untere bzw. obere Aktivitätsquartil liefert die Maske; alle Grenzwertbindungen werden aufgenommen. Ausschließlich Query-Daten bestimmen die Maske, nicht Kandidatenaktivierungen oder tatsächliche Herkunft. Ein neues maskenspezifisches gemeinsames Hungarian-Matching minimiert die vollständige quadratische Distanz einschließlich Kandidatennormtermen für alle Kandidaten gleich.

{table(activity,['dataset','metric','layer','mean_rank','top1','p_perm_less'])}
Anzahl ausgewählter Hidden-Neuronen am Endpunkt:

{table(maskend)}
Am Endpunkt sind insgesamt {int(inactive.sum())} Neuronen über die 30 Netze auf diesem Probe vollständig inaktiv. »Inaktiv« bezeichnet ausschließlich Nullaktivität auf diesem konkreten Probe, keine globale Aussage für alle möglichen Eingaben. Ein Test mit einer leeren Inaktivitätsmaske wäre undefiniert; stattdessen werden die nichtleeren Aktivitätsquartile und die tatsächlichen Inaktivitätszahlen berichtet. Die Ergebnisse häufiger aktiver Neuronen sind in dieser Analyse kein Vergleich funktionaler Anfangs-/Endrepräsentationen und kein Beweis, dass der erkennbare Fingerabdruck für die Klassifikation benötigt wird.

## Fazit und Grenzen

Punkt1 zeigt die Wiedererkennbarkeit der eigenen Initialisierung; Punkt2 bestätigt ihre Robustheit für mehrere Gesamtgewichts-Darstellungen und lokalisiert deutliche Unterschiede zwischen Schichten und Zuordnungen. Die schwache Ensemble-RSA aus01 bleibt davon unberührt. Beide Ergebnisse beantworten unterschiedliche Fragen.

Initialisierung und Batch-Reihenfolge sind weiterhin durch denselben Run-Seed gekoppelt. Diese Auswertung trennt ihre kausalen Beiträge nicht. Für eine solche Trennung wäre das vorgeschlagene gekreuzte Training erforderlich; es wurde nicht beauftragt und nicht gestartet. Keine Aussage über eine vollständige Trajektorienrekonstruktion oder einen dynamischen Attraktor folgt aus diesen Ergebnissen.

## Reproduktion und Dateien

- [Figurenübersicht](index.html), [alle Endpunktkontrollen inklusive Normrang-Zusatz](all_control_endpoints.csv).
- `retrieval_by_dataset.csv` enthält die 42 geplanten Varianten; `norm_rank_controls/retrieval_by_dataset.csv` die sechs nachträglichen Normrang-Varianten. Beide enthalten sämtliche Zeitpunkte und ungeglättete Ergebnisse.
- `data/distance_definitions.json`: vollständige Kosten-, Matching-, Normierungs- und Maskendefinitionen.
- `data/*_retrieval.npz`: alle Kosten, neuronalen Zuordnungen, Normen, Residualschwellen und -mengen; `data/*_query_masks.npz`: jede Query-Maske und Aktivitätsrate.
- `data/*_inference.npz`: sämtliche Nullstatistiken, bijektive Herkunftsmappings und Jackknife-Replikate. Entsprechende Dateien für Normränge unter `norm_rank_controls/data/`.
- Prüfsummenbelege unter `provenance/`; das Punkt1-Referenzprotokoll stimmt in allen Matrizen mit maximalem Fehler {max(r['max_primary_difference_from_point1'] for r in val['outputs']):.1g} überein.

Wiederholung: `.venv/bin/python 03_Initialization_Controls/code/run.py`. Alle Ergebnisse liegen ausschließlich hier. Original-Reproduktion, Experiment01 und der vor Beginn dieser Kontrolle eingefrorene Punkt1 bleiben unverändert.
'''
    (out/'summary.md').write_text(summary,encoding='utf-8')
    figures=[('control_endpoint_summary','Alle Kontrollprotokolle am Endpunkt'),('control_trajectories','Ausgewählte Kontrollen im Trainingsverlauf'),('layer_activity_controls','Schichten und Aktivitätskontrollen'),('residual_and_change_controls','Residuen und tatsächliche Gewichtsänderungen')]
    cards=[]
    for stem,label in figures:
        if (out/f'{stem}.png').exists():cards.append(f'<section><h2>{html.escape(label)}</h2><a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(label)}"></a><p><a href="{stem}.png">PNG</a> · <a href="{stem}.pdf">Vektor-PDF</a></p></section>')
    page=f'''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>03: Kontrollen</title><style>body{{max-width:1250px;margin:40px auto;padding:0 22px;font:17px/1.55 system-ui;color:#243448;background:#f6f8fa}}header,section{{background:white;border:1px solid #dce2e8;border-radius:12px;padding:25px;margin:22px 0}}img{{width:100%;height:auto}}a{{color:#215d96}}</style><header><h1>03: Kontrollen zur Wiedererkennung</h1><p>{html.escape(title)}</p><p>42 geplante Kontrollvarianten plus sechs nachträgliche Normrang-Kontrollen. Alle Kontrolltests sind explorativ.</p><p><a href="summary.md">Ergebnisbericht</a> · <a href="all_control_endpoints.csv">Alle Endpunkte (CSV)</a> · <a href="retrieval_config.json">Konfiguration</a> · <a href="../02_Initialization_Retrieval/index.html">Punkt 1</a></p></header>{''.join(cards)}</html>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'README.md').write_text('# 03 Initialization Controls\n\n[Ergebnisbericht](summary.md) · [Figurengalerie](index.html) · [Alle Endpunktkontrollen](all_control_endpoints.csv)\n\n42 geplante Varianten und sechs ausdrücklich nachträgliche Normrang-Kontrollen; keine neuen Trainingsläufe. Punkt1 und alle bisherigen Ergebnisse bleiben unverändert.\n',encoding='utf-8')
