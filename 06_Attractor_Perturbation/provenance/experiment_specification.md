# Experiment 2 — Dynamische Rückkehr nach gezielten Gewichtsraumstörungen

## Zweck und Anspruch

Das Ausgangspaper von Krauss et al. (2026), [DOI 10.3389/fcogn.2026.1960468](https://doi.org/10.3389/fcogn.2026.1960468), weist aufgabenabhängige Cluster trainierter Netze nach, aber **keine dynamischen Attraktoren**. Dieses Experiment untersucht, ob Netzwerke nach kontrollierten Störungen ihrer Gewichtskonfigurationen unter **derselben Lernregel und Aufgabe** zu einer vorher definierten Lösungsklasse beziehungsweise Region zurückkehren.

**Auftrag an Codex:** Zuerst die laufende Reproduktion des Papers unverändert abschließen und dokumentieren. Für dieses Experiment eigene Fork-Runs und eigene Ergebnispfade verwenden. Originalcheckpoints nicht überschreiben. Trainings- und Optimizer-Zustände sorgfältig verwalten, experimentelle Random-Seeds festhalten, Pilot vor Vollauswertung.

## Getrennte Hypothesen

- **H1 – Trajektorienstabilität:** Zwei zur gleichen Zeit identische Trainingsläufe, von denen einer in den Gewichten gestört wird, nähern sich unter identischen zukünftigen Updates im permutations- und skalierungsbereinigten Gewichtsraum wieder an.
- **H2 – Attraktion zu einer Lösungsregion:** Gestörte Runs kehren in eine vorher definierte, aufgabenspezifische Menge funktional und strukturell geeigneter Zustände zurück, statt nur irgendeine Lösung mit guter Testgenauigkeit zu erreichen.
- **H3 – Endpunkt-/Basinstabilität:** Bei hinreichend kleinen Störungen kehren Netze zur gleichen lokalen Lösungsfamilie zurück; stärkere Störungen können zu anderen Lösungsfamilien führen. Der Anteil von Rückkehren als Funktion der Störstärke charakterisiert eine *empirische* Basin-Grenze.

Eine bloße Erholung der Klassifikationsaccuracy ist **kein Attraktornachweis**. Ebenso wenig beweist geringe Distanz nach Rückkehr einen strengen mathematischen Attraktor, solange die kontinuierliche Dynamik und das Basin nicht charakterisiert sind.

## Ausgangssystem

- Originalarchitektur und Trainingsbedingungen zunächst beibehalten: 784–64–10 ReLU, Null-Biases, SGD mit Lernrate 0,1 und Momentum 0,9, Batchgröße 256.
- Start mit MNIST; nach erfolgreichem Pilot auf Fashion-MNIST und KMNIST erweitern.
- Störzeitpunkte aus reproduzierten Checkpoints: etwa `E=2` (nach einer Epoche), `E=10` (nach fünf Epochen) und `E=20` (nach zehn Epochen). Auch ein dichter früher Checkpoint ist optional.
- Referenz ist pro Seed die **unveränderte Weitertrainings-Trajektorie** ab dem jeweiligen Checkpoint; zusätzlicher Vergleich zu unabhängigen Netzwerken derselben Aufgabe.

## Kontrolliertes Fork-Design

Für jedes ausgewählte Netzwerk und jeden Störzeitpunkt:

1. Ausgangsgewichte und nach Möglichkeit den **vollständigen SGD-Optimizer-Zustand inklusive Momentum** sowie die genaue RNG-/Mini-Batch-Reihenfolge rekonstruieren. Falls nur Gewichte gespeichert wurden, Originaltraining bis zum Störzeitpunkt deterministisch replayen; falls das nicht möglich ist, in **beiden** Fork-Armen den Optimizer identisch neu initialisieren und diese Variante gesondert kennzeichnen.
2. Aus demselben Ausgangszustand einen `baseline`-Fork (`epsilon=0`) und mehrere `perturbed`-Forks erzeugen. Beide bekommen **dieselben zukünftigen Batches in derselben Reihenfolge** und dieselben Lernraten.
3. Einmalig direkt nach dem Checkpoint additive Gauß-Störung in die Gewichte des Perturbations-Forks injizieren:

\[
W_\ell^+=W_\ell+\Delta W_\ell,\qquad
\Delta W_\ell=\epsilon\,\|W_\ell\|_F\,
\frac{Z_\ell}{\|Z_\ell\|_F},\quad Z_\ell\sim\mathcal N(0,I).
\]

4. Schichtweise Varianten ausführen: nur `W0`, nur `W1`, beide Matrizen. Initial `epsilon ∈ {0, 0.01, 0.05, 0.10, 0.20}`; bei vollständigem Leistungseinbruch kleineren Bereich ergänzen. Mindestens fünf unabhängig gezogene Störrichtungen pro Seed/Störstärke im Pilot.
5. Nach der Störung ausreichend lange unter derselben Aufgabe weitertrainieren, zunächst weitere zehn Epochen oder bis zu einer vorab festgelegten Plateau-/Stopregel. Nicht selektiv nur erfolgreich rückkehrende Runs auswerten.

**Wichtig:** Gewichte nicht zur Störung in eine andere ReLU-Gauge überführen. Die kanonische Gauge ist **nur eine Analysetransformation**: Eine Reskalierung vor dem nächsten SGD-Update kann selbst bei identischer Funktion den weiteren Optimierungsverlauf verändern.

## Momentum und andere wichtige Kontrollen

Momentum kann eine vor der Störung vorhandene Bewegungsrichtung fortsetzen und dadurch scheinbare Rückkehr erzeugen. Deshalb getrennt auswerten:

- **Kontrolle A (volle Fortsetzung):** In beiden Forks identischer gespeicherter Momentumzustand.
- **Kontrolle B (Momentum-Reset):** In beiden Forks Momentum gleichzeitig auf null setzen, sonst identisch weitertrainieren.
- **Kontrolle C:** Gleiche Störbedingungen, aber unterschiedliche zukünftige Mini-Batch-Reihenfolgen; isoliert die Robustheit gegenüber SGD-Rauschen.
- **Kontrolle D:** Unabhängige Initialisierungen bzw. andere aufgabengleiche Netzwerke als Referenzen. Wiedererreichen eines allgemeinen Task-Clusters ist nicht automatisch Rückkehr zur ursprünglichen individuellen Lösung.
- **Symmetrie-Sanity:** Funktionsäquivalente Neuronenpermutation bzw. positive ReLU-Skalierung ohne Lernschritt darf die gemessene Quotient-Distanz nicht erhöhen; dies dient der Validierung der Messung, nicht als Attraktorbefund.

Bei Unterbrechungen keine stillschweigende Verwendung des Originaloptimizers für nur einen Fork. Exakt dokumentieren, ob Momentum, Weight Decay und RNG-Zustände vergleichbar sind.

## Primäre Messgrößen

Die Gewichte für die **Auswertung** kanonisch normieren und wie im Paper permutationsadjustieren (Git Re-Basin als Primärmethode, PIF/Hungarian und Alpha als Robustheit). Mindestens folgende Größen über die Recovery-Updates messen:

1. **Abstand zur individuellen Kontrolltrajektorie:**

\[
r(s)=\frac{d_Q(W_{\mathrm{pert}}(t_p+s),\,W_{\mathrm{base}}(t_p+s))}
{d_Q(W_{\mathrm{pert}}(t_p^+),\,W_{\mathrm{base}}(t_p))}.
\]

   `r(s)<1` bedeutet Annäherung relativ zur Stördistanz; es ist alleine noch kein Nachweis eines Punktattraktors. Den **absoluten** Abstand immer mitplotten.
2. **Distanz zu einer zuvor definierten Task-Lösungsmenge:** Eine unabhängige, zurückgehaltene Referenzkohorte derselben Aufgabe bei gleichem Trainingshorizont beziehungsweise vergleichbarem Trainingsstand auf Leistung und Gewichtsdistanzen charakterisieren. Größe und Zahl der Referenzmodelle zwischen Tasks gleich halten. Die Rückkehrschwelle vor Kenntnis der Störresultate aus der Verteilung unabhängiger aufgabengleicher Netze ableiten.
3. **Rückkehrwahrscheinlichkeit pro epsilon:** Anteil der gestörten Forks, die die vorab definierte Region wieder erreichen; Unsicherheit auf Ebene der **Ausgangsnetzwerke** bestimmen.
4. **Funktionskontrolle:** Testverlust, Accuracy, optional CKA verborgener Aktivierungen auf festem Testset. Funktionale und strukturelle Erholung getrennt berichten.
5. **Loss-Barrieren (sekundär):** Nach symmetrieadjustiertem Matching lineare Interpolation zwischen Kontroll- und Recovery-Endpunkt auswerten. Eine niedrige Barriere ist ein Hinweis auf eine verbundene Lösungsregion, aber alleine kein Beweis dynamischer Attraktion.

## Entscheidung zwischen möglichen Ergebnissen

- **Nur funktionale Erholung:** Accuracy/Loss erholen sich, Gewichtsabstand zur Ausgangsregion nicht → robuste Klassifikation, aber kein Nachweis derselben Gewichtsraum-Lösungsmenge.
- **Task-Cluster-Rückkehr:** Gewichtsabstand zur Task-Region wird wieder klein, Abstand zur individuellen Kontrolltrajektorie bleibt groß → Evidenz für Attraktion auf eine *Menge* von Lösungen, nicht auf denselben Punkt.
- **Lokale Rückkehr:** Paired Perturbations-Forks nähern sich reproduzierbar der ungestörten Trajektorie bzw. ihrer identifizierten lokalen Lösungsfamilie; Rückkehrwahrscheinlichkeit sinkt mit Störamplitude → Evidenz für lokale Rückstell-/Basin-Eigenschaften.
- **Keine strukturelle Rückkehr:** Netze lernen die Aufgabe erfolgreich erneut, landen aber in anderen Regionen → Clusterbildung ohne nachgewiesene dynamische Attraktion.

Bei konstantem SGD-Learning-Rate und Rauschen kann ein stabiler Bereich statt eines Fixpunkts vorliegen. Ergebnisse als **empirische Stabilität/Attraktion unter der spezifizierten Update-Dynamik** formulieren, nicht als universellen mathematischen Satz.

## Statistik und Ablauf

Pilot: MNIST, zehn Seeds, ein später Störzeitpunkt (`E=20`), drei nichttriviale Störstärken, fünf Richtungen und beide Momentum-Protokolle. Danach Pre-Registration der erweiterten Matrix (Datasets × Checkpoints × Layers × epsilons × Richtungen). Für Konfidenzintervalle zunächst über Seeds aggregieren oder hierarchisch über Seeds und Störrichtungen resampeln; Richtungen aus demselben Startmodell sind keine unabhängigen Netzwerkläufe. Paired-Analysen bevorzugen. Kontrolliere die Häufigkeit von Divergenzen und numerischen NaNs ausdrücklich.

## Erwartete Ausgaben

- Separater `perturbation_manifest.csv` mit Ursprung, Seed, Dataset, Störzeit, epsilon, betroffener Schicht, Richtung, Momentum-Protokoll und RNG.
- Fork-Checkpoints getrennt von Originaldaten, reproduzierbare `config.json` und Checksummen.
- `recovery_curves.png/pdf`: absolute/relative Quotient-Distanz und Accuracy/Loss als Funktion der Fortsetzungsschritte.
- `return_probability.png/pdf`: Return-Rate gegen epsilon nach Seed-Level-Unsicherheit.
- `basin_analysis.md`: Return-Kriterien, alternative Referenzmengen, Momentum-Kontrollen und Loss-Barrier-Auswertungen.
- `attractor_conclusion.md`: explizite Entscheidung, ob lediglich Leistungsrestauration, Task-Region-Rückkehr, lokale Bahnstabilität oder keine strukturelle Attraktion beobachtet wurde.

## Referenzen

- Krauss et al. (2026): https://doi.org/10.3389/fcogn.2026.1960468
- Ainsworth et al. (2023), *Git Re-Basin*: https://arxiv.org/abs/2209.04836
- Garipov et al. (2018), *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*: https://arxiv.org/abs/1802.10026
