# 05: Initialisierung und Batch-Reihenfolge getrennt variieren

Die Initialisierung ist aus den Endgewichten auch über unabhängig variierte Batch-Reihenfolgen hinweg in allen drei Datensätzen überzufällig identifizierbar.

## Neues gekreuztes Training

150 vollständig neu trainierte Netze: drei Datensätze × zehn Anfangskonfigurationen × fünf Batch-Reihenfolgen. Jede der zehn Initialisierungen eines Datensatzes wurde mit jeder der fünf Reihenfolgen kombiniert. Die Initialgewichte sind bitgleich zu den ursprünglichen E0-Checkpoints. Identische Shuffle-Faktoren erzeugen identische Epochenreihenfolgen über alle Initialisierungen; die fünf Seed-Bereiche überlappen nicht.

Die fünf Shuffle-Baseseeds sind 11001, 12001, 13001, 14001 und 15001; pro Epoche wird die nullbasierte Epochenzahl addiert. Architektur 784-64-10, ReLU, feste Null-Biases, SGD mit Lernrate 0,1, Momentum 0,9, Batchgröße 256, zehn Epochen. Alle 3.150 Checkpoints inklusive Momentum, Modell-/Optimierer-/RNG-Zuständen und konkreter Datenreihenfolge sind gespeichert. Es gibt eine Beobachtung pro 10×5-Zelle, keine zusätzlichen Wiederholungen innerhalb einer Zelle.

## Primär: gleiche Initialisierung über verschiedene Reihenfolgen erkennen

Für jedes gerichtete Paar verschiedener Shuffle-Level wird jedes Endnetz mit den zehn Endnetzen des anderen Levels verglichen. Der richtige Kandidat teilt dessen Initialisierung. Alle 20 gerichteten Shuffle-Paarungen werden ausgewertet; zunächst wird je Initialisierung über diese Vergleiche gemittelt, anschließend über die zehn Initialisierungen. Primär ist die kanonische, gemeinsam auf Eins normierte W0/W1-Gewichtsdistanz nach optimalem gemeinsamem Matching. Zufall: mittlerer Rang 5,5, Top-1=10%.

| dataset | mean_rank | mean_rank_ci_low | mean_rank_ci_high | top1 | p_perm_less | p_holm_less |
| --- | --- | --- | --- | --- | --- | --- |
| mnist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0001 | 0.0003 |
| fashion_mnist | 1.2200 | 1.0149 | 1.4251 | 0.9000 | 0.0001 | 0.0003 |
| kmnist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0001 | 0.0003 |

Die drei primären Datensatztests werden mit Holm korrigiert. Ein Treffer ist hier **Endnetz gegen Endnetz über verschiedene Trainingsreihenfolgen**, nicht der Anfangs-Ende-Test aus Experiment 02. Damit wird der Initialisierungsfaktor bei variiertem Shuffle untersucht.

## Sekundär: Endrepräsentationen

Die gleiche Faktoridentifikation wurde vor der Trainingsauswertung sekundär für Hidden-CKA festgelegt. Alle 50 Endnetze eines Datensatzes werden auf demselben gehashten 3.000-Bilder-Probe ausgewertet. Kosten sind 1 minus zentrierte lineare CKA; es gibt keine Neuronenzuordnung.

| dataset | mean_rank | mean_rank_ci_low | mean_rank_ci_high | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | 1.0600 | 0.9743 | 1.1457 | 0.9700 | 0.0001 |
| fashion_mnist | 2.8000 | 2.0884 | 3.5116 | 0.4550 | 0.0001 |
| kmnist | 1.2750 | 1.0329 | 1.5171 | 0.8700 | 0.0001 |

Alle drei Datensätze zeigen hier überzufällige Erkennung (97 %, 45,5 % und 87 % Top-1). Die gemeinsame Initialisierung prägt damit die finale Hidden-Repräsentationsstruktur trotz unterschiedlicher Batch-Reihenfolgen. Ein entsprechender Fingerabdruck in finalen Logits oder Vorhersagen wurde hier nicht getestet.

Dies ist eine andere Frage als Experiment 04: Hier werden **Endrepräsentationen unter unterschiedlichen Reihenfolgen** verglichen. Eine fehlende direkte Wiedererkennung der Anfangsrepräsentation schließt einen Initialisierungseinfluss auf die später gelernten Repräsentationen nicht aus. Die CKA-Ergebnisse bleiben sekundär/explorativ.

## Sekundär: gleiche Reihenfolge über verschiedene Initialisierungen erkennen

Umgekehrt vergleichen wir alle 90 gerichteten Paare verschiedener Initialisierungen. Jeder Query hat fünf Endnetze der anderen Initialisierung als Kandidaten; der richtige Kandidat teilt dessen Shuffle-Level. Zufall: mittlerer Rang 3, Top-1=20%. Gewichte und CKA werden separat ausgewertet.

| dataset | metric | mean_rank | mean_rank_ci_low | mean_rank_ci_high | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- | --- |
| mnist | normalized_frobenius | 2.8356 | 2.6622 | 3.0089 | 0.2444 | 0.0142 |
| mnist | one_minus_cka | 2.7111 | 2.5583 | 2.8639 | 0.2800 | 0.0001 |
| fashion_mnist | normalized_frobenius | 2.5178 | 2.3225 | 2.7131 | 0.3311 | 0.0001 |
| fashion_mnist | one_minus_cka | 2.4578 | 2.2388 | 2.6767 | 0.3467 | 0.0001 |
| kmnist | normalized_frobenius | 2.6222 | 2.3555 | 2.8890 | 0.3044 | 0.0001 |
| kmnist | one_minus_cka | 2.6422 | 2.5651 | 2.7194 | 0.2800 | 0.0001 |

Diese Auswertung fragt nach einem über verschiedene Initialisierungen wiederkehrenden Fingerabdruck der Reihenfolge. Ein schwacher Identifikationsbefund bedeutet nicht, dass die Reihenfolge keine Auswirkungen auf ein einzelnes Training hat.

## Statistische Abhängigkeiten

Pro Datensatz werden 10.000 konsistente blockierte Labelpermutationen gerechnet. Für Initialisierungs-Retrieval wird innerhalb jedes Shuffle-Levels die Zuordnung von Initialisierungslabels gemeinsam für alle zugehörigen Vergleiche permutiert. Der erste Shuffle-Block bleibt fix, weil eine gemeinsame globale Umbenennung aller Initialisierungen die Statistik nicht verändern würde. Alle anderen Blockzuordnungen sind unabhängig. Bei Shuffle-Retrieval werden analog Shuffle-Labels innerhalb der Initialisierungsblöcke permutiert; der erste Init-Block bleibt fix.

Die primäre Unsicherheit löscht eine komplette Initialisierungsgruppe mit allen fünf Trainingsläufen und rangiert mit neun Initialisierungskandidaten neu. Sie ist damit konditional auf die fünf gewählten Shuffle-Level. Die Shuffle-Auswertung löscht jeweils eine komplette Shuffle-Gruppe und ist konditional auf die zehn Initialisierungen. Die 50 Netze sind keine 50 unabhängigen Initialisierungen, die vielen Paarvergleiche keine unabhängigen Replikate. Normierte Rangscores korrigieren die veränderte Kandidatenzahl beim Jackknife. Leave-one-Level-Ergebnisse für den jeweils anderen Faktor stehen separat als Robustheitskontrolle, ohne zusätzlichen Hypothesentest.

Retrieval-p-Werte testen einseitig auf kleinere korrekte Ränge. Alle p-Werte verwenden die Plus-eins-Monte-Carlo-Regel; die Untergrenze ist 1/10001 (gerundet 0,00010). Nur die drei gewichtsbasierten Initialisierungs-Endpunkte sind primär; Shuffle-, CKA-, Trajektorien- und Distanzkontrasttests sind sekundär/explorativ. Normale 95%-Jackknife-Intervalle sind bei zehn bzw. fünf Faktorstufen begrenzt präzise. Die ungekürzten Tabellenintervalle können theoretische Ranggrenzen überschreiten; in Abbildungen werden sie auf den möglichen Wertebereich begrenzt. Bei perfekter empirischer Erkennung können sie zu einem Punkt kollabieren, ohne perfekte Erkennung neuer Netze zu garantieren.

## Geometrische Kontraste

Pro Datensatz gibt es 100 ungeordnete Paare mit gleicher Initialisierung und unterschiedlichem Shuffle, 225 mit unterschiedlicher Initialisierung und gleichem Shuffle sowie 900 mit beiden Faktoren unterschiedlich. Die Kontraste vergleichen jeweils die Distanz bei beiden unterschiedlichen Faktoren mit jener bei gleichem Zielfaktor. Positive Werte sprechen für geringere Distanzen bei gemeinsamem Faktor. Die Nullverteilungen bewahren dieselbe Blockstruktur wie die Retrievaltests.

| dataset | metric | factor | same_factor_mean | different_factor_mean | contrast | ci_low | ci_high | p_perm_greater |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | normalized_frobenius | initialization | 0.7475 | 0.9144 | 0.1669 | 0.1560 | 0.1777 | 0.0001 |
| mnist | normalized_frobenius | shuffle | 0.9125 | 0.9144 | 0.0019 | 0.0008 | 0.0029 | 0.0031 |
| mnist | one_minus_cka | initialization | 0.1613 | 0.2058 | 0.0446 | 0.0393 | 0.0498 | 0.0001 |
| mnist | one_minus_cka | shuffle | 0.2031 | 0.2058 | 0.0027 | 0.0013 | 0.0042 | 0.0001 |
| fashion_mnist | normalized_frobenius | initialization | 0.8103 | 0.8791 | 0.0688 | 0.0584 | 0.0793 | 0.0001 |
| fashion_mnist | normalized_frobenius | shuffle | 0.8718 | 0.8791 | 0.0074 | 0.0045 | 0.0102 | 0.0001 |
| fashion_mnist | one_minus_cka | initialization | 0.2390 | 0.2599 | 0.0209 | 0.0165 | 0.0253 | 0.0001 |
| fashion_mnist | one_minus_cka | shuffle | 0.2521 | 0.2599 | 0.0078 | 0.0054 | 0.0101 | 0.0001 |
| kmnist | normalized_frobenius | initialization | 0.8072 | 0.9503 | 0.1431 | 0.1318 | 0.1544 | 0.0001 |
| kmnist | normalized_frobenius | shuffle | 0.9470 | 0.9503 | 0.0033 | 0.0016 | 0.0050 | 0.0001 |
| kmnist | one_minus_cka | initialization | 0.2660 | 0.3156 | 0.0496 | 0.0448 | 0.0544 | 0.0001 |
| kmnist | one_minus_cka | shuffle | 0.3116 | 0.3156 | 0.0041 | 0.0023 | 0.0059 | 0.0001 |


## Testleistung und deskriptive Zerlegung

Die vollständigen Accuracy-/Loss-Tabellen enthalten jede 10×5-Zelle an allen 21 Checkpoints. Die folgende Tabelle zeigt E20-Mittelwerte über alle 50 Läufe pro Datensatz und zerlegt die beobachtete Quadratsumme in additive Initialisierungs-, additive Shuffle- und Interaktions-/Restanteile:

| dataset | metric | grand_mean | initialization_fraction | shuffle_fraction | interaction_residual_fraction |
| --- | --- | --- | --- | --- | --- |
| mnist | accuracy | 0.9727 | 0.1807 | 0.1657 | 0.6536 |
| mnist | loss | 0.1150 | 0.1946 | 0.1344 | 0.6710 |
| fashion_mnist | accuracy | 0.8638 | 0.1111 | 0.4378 | 0.4510 |
| fashion_mnist | loss | 0.4041 | 0.0835 | 0.3023 | 0.6141 |
| kmnist | accuracy | 0.8605 | 0.1989 | 0.2664 | 0.5347 |
| kmnist | loss | 0.7921 | 0.2486 | 0.2499 | 0.5015 |

Die Freiheitsgrade betragen 9 für Initialisierung, 4 für Shuffle und 36 für Interaktion/Rest. Größere Quadratsummenanteile allein sind kein unverzerrter Vergleich zufälliger Varianzkomponenten und kein Signifikanznachweis. Mit nur einer Beobachtung pro Zelle können Interaktion und ein zusätzlicher Restfehler nicht separat geschätzt werden. Deshalb werden keine ANOVA-F-Tests oder zugehörigen p-Werte behauptet. Identifizierbare Gewichts- oder Repräsentationsunterschiede sind nicht automatisch große Unterschiede in der Testgenauigkeit.

## Wiedererkennung der originalen Initialgewichte über das Training

Zusätzlich wird jedes der 50 aktuellen Netze eines Datensatzes zu jedem Checkpoint mit den zehn Originalinitialisierungen verglichen. Wiederholungen derselben Initialisierung über die fünf Shuffles werden gemeinsam behandelt. Endpunkt:

| dataset | mean_rank | mean_rank_ci_low | mean_rank_ci_high | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | 1 | 1 | 1 | 1 | 0.0001 |
| fashion_mnist | 1 | 1 | 1 | 1 | 0.0001 |
| kmnist | 1 | 1 | 1 | 1 | 0.0001 |


## Grenzen und Dateien

Das vollständige gekreuzte Design trennt Initialisierung und Reihenfolge innerhalb dieses festen Trainingsprotokolls. Die Aussagen bleiben auf die gewählten Datensätze, Architektur, zehn Anfangszustände und fünf Reihenfolgen beschränkt. Es wird keine vollständige Rekonstruktion einer Trainingsgeschichte oder ein dynamischer Attraktor nachgewiesen.

- [Alle Ergebnisse als PDF](results.pdf), [Figurengalerie](index.html).
- `data/manifest.json`: 150 Läufe und Quell-/E0-/Shuffle-Hashes. `data/checkpoints/`: NPZ und PT mit allen 3.150 Checkpoints.
- `initialization_retrieval.csv`, `shuffle_retrieval.csv`, `aggregate.csv`: Faktoridentifikation und Unsicherheit.
- `initial_trajectory_retrieval.csv`, `distance_groups.csv`, `distance_contrasts.csv`, `performance_decomposition.csv`: ergänzende Auswertungen.
- `data/*_distances.npz`: alle Endzustandsdistanzen, CKA, Anfangs-Kandidatenkosten und Matchings. `data/*_inference.npz`: komplette blockierte Nullzuordnungen, Nullstatistiken und Jackknife-Replikate.
- `analysis_plan.json` und `analysis_design.json`: vorab dokumentierte Trainings- und Auswertungsentscheidungen; `provenance/`: Tests, unabhängige Reviews, Validierungen und Prüfsummen.

Reproduktion: `.venv/bin/python 05_Initialization_Shuffle/code/run.py`. Vorhandene vollständige Trainingsarchive werden nur nach strenger Validierung wiederverwendet. Frühere Experimente bleiben erhalten.
