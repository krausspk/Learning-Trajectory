"""German point-1 result report, built only from computed retrieval results."""
from pathlib import Path
import html,json
import numpy as np
import pandas as pd
from plots import DATASETS,NAMES,PRIMARY,primary


def fmt(x):
    if isinstance(x,(float,np.floating)):
        if not np.isfinite(x):return '-'
        return f'{x:.3g}' if 0<abs(x)<1e-4 else f'{x:.4f}'
    return str(x).replace('|',' / ')


def table(frame,columns=None):
    frame=frame if columns is None else frame[columns]
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join('---' for _ in frame.columns)+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def gallery(out,title,intro,figures):
    cards=[]
    for name,label in figures:
        if (out/f'{name}.png').exists():
            cards.append(f'<section><h2>{html.escape(label)}</h2><a href="{name}.png"><img src="{name}.png" alt="{html.escape(label)}"></a><p><a href="{name}.png">PNG</a> · <a href="{name}.pdf">Vektor-PDF</a></p></section>')
    content=f'''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>body{{max-width:1250px;margin:40px auto;padding:0 22px;font:17px/1.55 system-ui;background:#f6f8fa;color:#243448}}section,header{{background:white;padding:25px;margin:22px 0;border:1px solid #dce2e8;border-radius:12px}}img{{width:100%;height:auto}}a{{color:#215d96}}h1,h2{{line-height:1.25}}</style><header><h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p><p><a href="summary.md">Ergebnisbericht</a> · <a href="retrieval_by_dataset.csv">Alle Ergebnisse (CSV)</a> · <a href="retrieval_config.json">Konfiguration</a></p></header>{''.join(cards)}</html>'''
    (out/'index.html').write_text(content,encoding='utf-8')


def make_report(out:Path):
    out=Path(out)
    frame=primary(pd.read_csv(out/'retrieval_by_dataset.csv'))
    endpoint=frame[frame.checkpoint.eq(20)].copy()
    endpoint['dataset']=pd.Categorical(endpoint.dataset,DATASETS,ordered=True)
    endpoint=endpoint.sort_values('dataset')
    aggregate=primary(pd.read_csv(out/'aggregate.csv'))
    last=aggregate[aggregate.checkpoint.eq(20)].iloc[0]
    positive=bool((endpoint.mean_rank<5.5).all() and (endpoint.p_holm_less<.05).all())
    conclusion=('Die Endgewichte erkennen ihre zugehörige Initialisierung in allen drei Datensätzen überzufällig wieder.' if positive else 'Der Wiedererkennungstest liefert keine über alle drei Datensätze abgesicherte Identifikation der Initialisierung.')
    counts=', '.join(f'{NAMES[str(r.dataset)]}: {r.top1*10:g}/10' for r in endpoint.itertuples())
    text=f'''# 02: Wiedererkennung der Initialisierung

{conclusion}

## Beobachtung

Es wurden die gespeicherten Hauptkohorten-Netze analysiert: zehn je MNIST, Fashion-MNIST und KMNIST; 21 Checkpoints E0 bis E20. Jedes aktuelle Netz i wird mit jeder Initialisierung j seines Datensatzes verglichen. Die 10×10-Matrix enthält sämtliche 100 Kombinationen, einschließlich der echten Herkunftspaare, ohne deren Sonderbehandlung. Insgesamt 6.300 Matchings; kein neues Training.

{table(endpoint,['dataset','mean_rank','mean_rank_ci_low','mean_rank_ci_high','top1','mrr','p_perm_less','p_holm_less'])}
Top-1-Erkennung am Endpunkt: **{counts}**. Zufallserwartung: 1/10 pro Datensatz bzw. 10%. Der mittlere richtige Rang beträgt unter zufälliger Zuordnung 5,5; kleiner ist besser. Rang 1 bedeutet, dass die eigene Initialisierung von zehn Kandidaten am nächsten liegt. MRR ist der mittlere Kehrwert des richtigen Ranges.

Gleichgewichtete Zusammenfassung über die drei Datensätze: mittlerer Rang **{last.mean_rank:.3f}**, Top-1 **{last.top1:.1%}**, einseitiges Permutations-p **{last.p_perm_less:.4g}**. Die Einzeldatensätze bleiben die primären Endpunkte.

## Festgelegtes Distanzprotokoll

Positive kanonische ReLU-Gauge für jedes Netz: eingehende und ausgehende Norm jedes Hidden-Neurons angleichen. Anschließend beide Gewichtsmatrizen gemeinsam auf Gesamt-Frobenius-Norm Eins skalieren. Diese Gesamtnormierung dient dem Analysevergleich; anders als die kompensierende ReLU-Gauge ist sie keine funktionsbewahrende Transformation.

Für jedes Query/Kandidaten-Paar wird unabhängig die gemeinsame optimale W0/W1-Hungarian-Zuordnung bestimmt. Die Distanz ist die Frobenius-Distanz der beiden so normalisierten Gesamtgewichtsvektoren, also sqrt(2 × Cosine-Dissimilarität) nach Matching. Alle 100 Paare folgen demselben Verfahren. Die Initialisierung wird allein anhand dieser gespeicherten Gewichte als Kandidat bewertet; Run-ID und tatsächliche Herkunft werden erst bei der Auswertung der Ränge verwendet.

## Statistik

Primärer Endpunkt: mittlerer Rang der korrekten Initialisierung bei E20, einseitig gegen kleinere Ränge. Je Datensatz 10.000 globale bijektive Herkunftslabel-Permutationen mit Seeds MNIST2026111701, Fashion-MNIST2026111702 und KMNIST2026111703. Die komplette Zuordnung zwischen zehn Query-Netzen und zehn Initialisierungen wird vertauscht; einzelne Distanzen werden nicht unabhängig permutiert. p=(1+Anzahl mindestens so guter Nullwerte)/10.001. Holm korrigiert die drei primären Datensatztests. Andere Checkpoints und Zusatzstatistiken sind explorativ. E0 ist ausschließlich eine Identitäts-Sanity-Prüfung ohne inferenzielle p-Werte.

Rangbindungen erhalten Durchschnittsränge. Bei mehreren nächsten Kandidaten wird Top-1 anteilig gewertet (1/Anzahl der gebundenen Minima). MRR-Nullerwartungen werden bei Bindungen aus allen Rangzellen exakt bestimmt. Alle Nullarrays, Labelpermutationen und Einzelquery-Ränge sind gespeichert.

Unsicherheit: Jeweils einen vollständigen Trainingslauf als Query-Zeile und Initialisierungs-Spalte entfernen, sodass neun Kandidaten verbleiben; Ränge neu berechnen. Jackknife auf dem normierten Rangscore (n−mittlerer Rang)/(n−1), Zufall=.5; für Top-1 und MRR entsprechend gegen ihre jeweilige Nullerwartung normieren. Anschließend auf die Skala mit zehn Kandidaten zurücktransformieren. Die unbeschnittenen normalapproximierten 95%-Intervalle stehen in den Tabellen. **Bei perfekter Wiedererkennung kann der Jackknife ein Intervall ohne Breite ergeben. Das beschreibt die vorliegende Stichprobe und garantiert keine perfekte Erkennung neuer Netze.**

## Bedeutung und Grenzen

Dieser Test untersucht die Identität des Ausgangszustands, während 01_RSA die Erhaltung der relativen Geometrie zwischen verschiedenen Netzen prüfte. Erfolgreiche Identifikation ist daher mit einer schwachen Anfangs-Ende-RSA vereinbar.

Ein Treffer zeigt Wiedererkennbarkeit unter den zehn verfügbaren Kandidaten, keine vollständige Rekonstruktion der Initialgewichte aus einem einzelnen Endnetz. Die Übertragbarkeit auf unbekannte Initialisierungen oder andere Trainingsbedingungen wurde hier nicht getestet. Derselbe Run-Seed bestimmt Initialisierung und Batch-Reihenfolge; der Befund betrifft daher die Herkunft des Trainingslaufs, ohne beide Ursachen kausal zu trennen. Die getrennt beauftragten Kontrollen in Experiment03 untersuchen Normen, Koordinaten, Schichten und die Rolle kleiner Residuen bzw. geringer Neuronenaktivität.

## Dateien und Wiederholung

- [Figurengalerie](index.html): Endpunkt-Matrizen, Verlauf und alle individuellen Endpunktränge, jeweils PNG/PDF.
- `retrieval_by_dataset.csv`, `aggregate.csv`, `per_query_ranks.csv`, `retrieval_jackknife.csv` und NPZ-Dateien unter `data/`.
- `data/*_retrieval.npz`: Distanzen, Reihenfolge beider Netzwerkachsen, Seeds, sämtliche Neuronenpermutationen und Normen.
- `data/*_inference.npz`: vollständige bijektive Labelpermutationen, Nullstatistiken und Jackknife-Replikate.
- `analysis_plan.json`: vor der Ergebnisauswertung festgelegtes Protokoll; `retrieval_config.json`: Software-/Codeprovenienz; `artifact_manifest.json`: Dateihashes.

Wiederholung vom Projektordner: `.venv/bin/python 02_Initialization_Retrieval/code/run.py`. Alle Originaldateien werden ausschließlich gelesen; Prüfsummenvergleich in `provenance/preservation_check.json`.
'''
    (out/'summary.md').write_text(text,encoding='utf-8')
    gallery(out,'02: Wiedererkennung der Initialisierung',conclusion,[('retrieval_endpoint_matrices','Endpunkt: Kandidatenrang für jedes Netz'),('retrieval_over_training','Wiedererkennung über das Training'),('retrieval_endpoint_ranks','Alle individuellen Endpunktränge')])
    (out/'README.md').write_text('# 02 Initialization Retrieval\n\n[Ergebnisbericht](summary.md) · [Abbildungen](index.html) · [Konfiguration](retrieval_config.json)\n\nAlle Ausgaben sind aus den bestehenden Trainings-Checkpoints gerechnet. Originaldaten und Experiment01 bleiben unverändert.\n',encoding='utf-8')
