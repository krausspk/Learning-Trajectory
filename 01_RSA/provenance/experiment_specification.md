# Experiment 1 — RSA zwischen Initialisierung und trainiertem Gewichtsraum

## Zweck und Arbeitsauftrag an Codex

Diese Spezifikation beschreibt **eine zusätzliche, zunächst rein analytische Auswertung** auf Basis der vollständig reproduzierten Daten aus Krauss et al. (2026), *Task-dependent clustering in permutation-aligned neural network weights*, *Frontiers in Cognition* 5:1960468, [DOI 10.3389/fcogn.2026.1960468](https://doi.org/10.3389/fcogn.2026.1960468).

**Erst starten, wenn die laufende Reproduktion abgeschlossen und gegen das Paper geprüft ist.** Die reproduzierte Trainings-, Metrik- und Matching-Pipeline nicht überschreiben. Nur lesen oder separat versionierte Auswertungsdateien anlegen. Keine stillschweigenden Änderungen an Seeds, Initialisierungen, Checkpoints oder Originalergebnissen. Falls benötigte Gewichte fehlen, die konkreten fehlenden Eingaben melden; nicht ungefragt ein anderes Experiment als Ersatz rechnen.

## Wissenschaftliche Hypothese

Das Paper zeigt, dass unabhängig initialisierte Netze während des Lernens aufgabenspezifische Gewichtsraumcluster bilden, ohne auf einen einzelnen Parameterpunkt zu konvergieren. **Neue Hypothese (Initial-Condition Geometry Persistence):** Die relativen Abstandsverhältnisse der verschiedenen Zufallsinitialisierungen bleiben innerhalb eines aufgabenspezifischen Clusters teilweise erhalten. Ein anfänglich relativ ähnliches Netzwerkpaar sollte daher auch am Ende des Trainings relativ ähnlicher sein.

Dies ist stärker als der bestehende Befund `within-task distance < between-task distance`: Es testet, ob die **individuelle Geometrie der Initialisierungsensemble** die Geometrie des trainierten Ensembles vorhersagt.

## Vorhandene Daten und Validierung

- Hauptkohorte: 30 unabhängige 784–64–10-ReLU-MLPs, zehn je MNIST, Fashion-MNIST, KMNIST; Gewichte `W0` (784 × 64) und `W1` (64 × 10), feste Null-Biases.
- Originaltraining: SGD, Lernrate 0,1, Momentum 0,9, Mini-Batch 256, zehn Epochen, 21 Checkpoints `E=0..20` in Halbepochen-Schritten.
- Pro Datensatz `10 choose 2 = 45` unabhängige **Netzwerkidentitäten**, aber 45 voneinander statistisch abhängige Paarwerte. Keine 435 Paarwerte als 435 unabhängige Beobachtungen behandeln.
- Checkpoint-Dateien, Run-IDs, Datensatz- und Seed-Zuordnung ausschließlich anhand des reproduzierten Manifests auflösen. Shapes, Finite-Werte, Checkpoint-Reihenfolge, Bias-Zustände und Zuordnung zu Originalfiguren prüfen.
- Die dicht gesampelte Frühtrainingskohorte ist optional eine separate Replikation; nicht ohne Kennzeichnung mit der Hauptkohorte vermischen.

## Primäre Berechnung

Für jedes Dataset separat und jedes Netzwerkpaar `(i,j)` eine symmetrieadjustierte Gewichtsraumdistanz am Anfang und zu allen späteren Checkpoints bestimmen:

\[
D^{(t)}_{ij}=d_Q(W_i^{(t)},W_j^{(t)}).
\]

Primäres Distanzprotokoll:

1. Jede gespeicherte ReLU-Gewichtskonfiguration **nur für die Analyse** in dieselbe kanonische positive ReLU-Gauge überführen (Norm der eingehenden und ausgehenden Gewichte pro Hidden Unit angleichen; Nullnorm-Sonderfälle deterministisch behandeln).
2. Für das **einzelne Hidden Layer** die Permutation mit dem Git-Re-Basin-Gewichtsabgleich (globale Hungarian-Zuordnung für das gemeinsame `W0/W1`-Skalarprodukt) bestimmen.
3. Als primäre Distanz die euklidische Gesamtgewichts-Distanz nach kanonischer Gauge und optimaler Permutation auswerten:

\[
d_Q^2(A,B)=\min_P \bigl(\|\widetilde W_{0,A}-\widetilde W_{0,B}P\|_F^2
+\|\widetilde W_{1,A}-P^\top\widetilde W_{1,B}\|_F^2\bigr).
\]

4. Für `E=0` und `E=20` aus den oberen Dreiecken ohne Diagonale die RSA berechnen:

\[
\rho_d(E)=\operatorname{Spearman}\bigl(
\operatorname{upper}(D_d^{(0)}),
\operatorname{upper}(D_d^{(E)})\bigr).
\]

Die Distanzdefinition, das Neuronenzuordnungsverfahren und die Gauge-Normierung dürfen über die Checkpoints nicht wechseln. Die optimale Paarpermutation selbst darf für jeden Checkpoint neu bestimmt werden, denn die zu vergleichende Größe ist die Distanz im permutationsbereinigten Raum. Zusätzlich ein Sensitivitätsexperiment mit am Anfang bzw. am Ende **festgehaltener** Paarpermutation durchführen, um Alignment-Wechsel als Einflussquelle zu quantifizieren.

## Analysereihenfolge

1. **Sanity Check:** `rho_d(0)=1` (außer bei degenerierten identischen Distanzen); Symmetrie, Nulldiagonale, exakt 45 Werte je Dataset.
2. **Primärer Endpunkt:** `rho_d(20)` für die drei Datensätze separat, einschließlich Streudiagramm `D(0)` gegen `D(20)` und Rangkorrelation.
3. **Verlaufsanalyse:** `rho_d(E)` über alle 21 Checkpoints (die Abnahme von 1 und ein etwaiges positives Plateau sind zentral).
4. **Schichtanalysen:** dieselbe RSA für `W0` und `W1` getrennt.
5. **Metrik-Robustheit:** Gesamt- und Layer-balancierte Frobenius-Distanz, normierte Distanz bzw. Cosine-Dissimilarität; außerdem Alpha-Dissimilarität (`alpha=2`) und SND entsprechend der reproduzierten Pipeline. Nicht suggerieren, dass die verschiedenen Maße dieselbe Geometrie definieren.
6. **Optional:** Die initialen nächsten Nachbarn eines Netzwerks ermitteln und prüfen, wie oft sie am Ende weiterhin zu seinen nächsten Nachbarn zählen (Top-k-Neighbor-Retention).

## Statistik, Kontrollen und Fallstricke

- Die **Netzwerke** sind die Replikationseinheiten, nicht die 45 Paarabstände. Nullverteilung durch Permutation der Identitätszuordnung der finalen RDM gegenüber der initialen RDM **innerhalb desselben Datensatzes**, mit simultaner Zeilen-/Spaltenpermutation. Mindestens 10.000 Permutationen für den primären Endpunkt, fixer Zufallsseed.
- Ein zusammenfassender Effekt über die drei Datensätze kann als gleichgewichtiges Mittel der Fisher-z-transformierten Datensatzkorrelationen ausgewiesen werden; Nullverteilung durch unabhängige Identitätspermutation innerhalb aller drei Strata. Die drei Einzelergebnisse stets zusätzlich zeigen.
- Unsicherheitsangaben vorzugsweise mit Run-Level-Jackknife wie im Paper, nicht mit naivem Paar-Bootstrap. Bei nur zehn Runs pro Stratum Grenzen der Präzision offen benennen.
- **Norm-Artefakt:** Raw-Frobenius-Distanzen können Anfangsnormen widerspiegeln. Normierte Metriken und explizite Analyse initialer Gewichts-/Layernormen als Kontrolle.
- **Hochdimensionale Konzentration:** Streuung der Initialisierungsdistanzen und Rangstabilität berichten. Keine Interpretation aus statistisch nahezu gleichen Zufallsdistanzen erzwingen.
- **Task-Konfundierung:** Zunächst ausschließlich innerhalb von MNIST, Fashion-MNIST bzw. KMNIST korrelieren. Ein gemeinsames Pooling aller 435 Paare würde die Datensatzcluster mit dem Initialisierungseffekt vermischen.
- **Alignment-Artefakt:** Paarweise Zuordnungen sind nicht automatisch zykluskonsistent über alle Netze. Die gewählte Quotient-Distanz und die Sensitivität gegenüber festgehaltenen Alignments dokumentieren.
- Die Identitätspermutation des finalen Ensembles muss die Korrelation gegen null bringen; andernfalls Pipeline und Indexzuordnung untersuchen.

## Erwartete Ausgaben

In einem neuen, eindeutigen Ergebnisverzeichnis für dieses Experiment:

- `rsa_config.json` mit Quellenmanifest, Versionsstand, Distanzdefinitionen, Seeds und Analyseoptionen.
- `rsa_by_dataset.csv` mit `dataset`, `checkpoint`, `metric`, `layer`, `rho`, Permutations-p und Unsicherheitsmaßen.
- `rsa_initial_vs_final.png/pdf`: drei Scatterplots, ein Panel pro Datensatz.
- `rsa_over_training.png/pdf`: RSA-Zeitverlauf pro Datensatz; `E=0` sichtbar.
- `distance_distributions.png/pdf`: Verteilungen und Streubreiten der Anfangs- und Endabstände.
- `rsa_robustness.md`: Schicht-/Metrik-/Alignment-/Norm-Kontrollen sowie exakte Stichprobenzahlen.
- `rsa_summary.md`: Trennung von Beobachtung, statistischer Evidenz und Interpretation.

## Wissenschaftliches Entscheidungskriterium

Ein **robuster, über Datensätze reproduzierbarer positiver Zusammenhang** der Anfangs- und End-RDMs wäre Evidenz für persistierende Information über die relativen Ausgangsbedingungen. Ein ausbleibender Zusammenhang würde lediglich bedeuten, dass diese *paarweise Geometrie* nicht erhalten ist; historische Information könnte weiterhin in anderen Merkmalen der Endgewichte liegen. Auch ein positiver Zusammenhang beweist **keine** Rekonstruktion einer vollständigen Trajektorie und **keinen** dynamischen Attraktor.

## Referenzen

- Krauss et al. (2026): https://doi.org/10.3389/fcogn.2026.1960468
- Ainsworth, Hayase & Srinivasa (2023), *Git Re-Basin*: https://arxiv.org/abs/2209.04836
