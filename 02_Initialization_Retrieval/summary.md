# 02: Wiedererkennung der Initialisierung

Die Endgewichte erkennen ihre zugehörige Initialisierung in allen drei Datensätzen überzufällig wieder.

## Beobachtung

Es wurden die gespeicherten Hauptkohorten-Netze analysiert: zehn je MNIST, Fashion-MNIST und KMNIST; 21 Checkpoints E0 bis E20. Jedes aktuelle Netz i wird mit jeder Initialisierung j seines Datensatzes verglichen. Die 10×10-Matrix enthält sämtliche 100 Kombinationen, einschließlich der echten Herkunftspaare, ohne deren Sonderbehandlung. Insgesamt 6.300 Matchings; kein neues Training.

| dataset | mean_rank | mean_rank_ci_low | mean_rank_ci_high | top1 | mrr | p_perm_less | p_holm_less |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | 1 | 1 | 1 | 1 | 1 | 0.0001 | 0.0003 |
| fashion_mnist | 1 | 1 | 1 | 1 | 1 | 0.0001 | 0.0003 |
| kmnist | 1 | 1 | 1 | 1 | 1 | 0.0001 | 0.0003 |

Top-1-Erkennung am Endpunkt: **MNIST: 10/10, Fashion-MNIST: 10/10, KMNIST: 10/10**. Zufallserwartung: 1/10 pro Datensatz bzw. 10%. Der mittlere richtige Rang beträgt unter zufälliger Zuordnung 5,5; kleiner ist besser. Rang 1 bedeutet, dass die eigene Initialisierung von zehn Kandidaten am nächsten liegt. MRR ist der mittlere Kehrwert des richtigen Ranges.

Gleichgewichtete Zusammenfassung über die drei Datensätze: mittlerer Rang **1.000**, Top-1 **100.0%**, einseitiges Permutations-p **9.999e-05**. Die Einzeldatensätze bleiben die primären Endpunkte.

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
