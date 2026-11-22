# Experiment 3 — Dekodierung der Aufgabenhistorie A→B→C versus B→A→C

## Ziel

Ein direktes Experiment zur **Historical Trace Hypothesis**: Kann man anhand der **finalen Gewichtsmatrizen allein** erkennen, in welcher Reihenfolge ein Netzwerk zuvor bestimmte Aufgaben gelernt hat, obwohl alle Netze am Ende dieselbe Aufgabe bearbeiten und ähnlich gut lösen?

Dies geht über die RSA des Ausgangspapiers hinaus. Eine positive RSA zwischen Initialisierung und Endzustand weist allenfalls persistierende Initialisierungsgeometrie nach; dieses Experiment zielt ausdrücklich auf **zeitlich geordnete Lernereignisse**.

**Codex-Arbeitsgrenze:** Die Reproduktion von Krauss et al. (2026), [DOI 10.3389/fcogn.2026.1960468](https://doi.org/10.3389/fcogn.2026.1960468), zuerst beenden. Dieses Experiment benötigt **neue**, kontrollierte Trainingsläufe. Originaltraining, bestehende Checkpoints und publizierte Ergebnisse unangetastet lassen.

## Vorab definierte Aufgaben

Als primärer Versuch dasselbe MNIST-Bildmaterial und dieselbe Architektur 784–64–10 verwenden. Alle drei Aufgaben müssen **identische Input- und Output-Dimensionalität** besitzen; die frühere MNIST-Ziffern-versus-Parität-Kontrolle eignet sich wegen 10 versus 2 Outputklassen nicht ohne Architekturänderung für einen direkten Vergleich aller Gewichte.

Eine einfache kontrollierte Variante mit fester Bildverteilung:

- **A:** zehnklassige Ziffernidentität (`y_A = y`).
- **B:** feste, vorab dokumentierte Permutation der zehn Klassenlabels (z. B. `y_B = (y+1) mod 10`).
- **C:** andere feste Permutation (`y_C = (y+3) mod 10`).

Gleiche MNIST-Bilder und Sample-Order in den entsprechenden Phasen verwenden. A/B/C unterscheiden sich zunächst nur durch die gelernte Entscheidungsregel, nicht durch ihre Inputstatistik. Ein späteres Experiment mit unterschiedlichen Datensätzen kann separat folgen.

## Hauptdesign: gepaarte Historien

Für jeden unabhängigen Seed eine **identische** `W0/W1`-Initialisierung erzeugen und davon zwei Trainingsläufe abzweigen:

- Historie `H=0`: A → B → C.
- Historie `H=1`: B → A → C.

Die A- und B-Phasen erhalten gleich viele Updates, identische Datensätze und möglichst identische Batch-Indizes; nur die Zuordnung der Aufgaben zu den ersten zwei Phasen wird getauscht. In beiden Armen folgt dieselbe ausreichend lange C-Phase mit identischer Lernrate und Batch-Reihenfolge. Primär SGD-Momentum **an jedem Phasenübergang in beiden Armen identisch zurücksetzen** (insbesondere vor C), damit spätere Unterschiede nicht einfach direkt im mitgeschleppten Momentumzustand liegen. Eine zusätzliche Analyse mit fortgesetztem Momentum ist als Robustheitskontrolle zulässig.

Vorgeschlagener Pilot: 200 unabhängige Seed-Blöcke mit je zwei Historien; A und B zunächst je eine Epoche, C mit vorab festgelegten Beobachtungshorizonten etwa 1, 5 und 10 Epochen. Bei positivem, aber instabilem oder kleinem Effekt unabhängige Bestätigung mit weiterer Seed-Kohorte (z. B. 500–1000 Blöcke). Exakte Batchzahlen und Horizonte vor Testauswertung einfrieren.

**Jede Probe bekommt nur `W_T` der finalen C-Phase als Eingabe.** Label ist `H ∈ {0,1}`. Keine früheren Gewichte, keine Optimizer-Zustände, keine Historien-Metadaten als Probe-Features zulassen.

## Primäre Probe

1. Finalgewichte in die im Ausgangspaper validierte kanonische ReLU-Gauge überführen und die Hidden Units mit Git Re-Basin gegen **eine nur aus Trainingsdaten ausgewählte, eingefrorene Referenz** ausrichten. Im Testsplit keinerlei Matching anhand früherer Checkpoints oder anhand des Geschichtslabels.
2. `W0` und `W1` vektorisieren; deren Skalen aus dem Trainingssplit bestimmen. Primär ein **linearer L2-regularisierter logistischer Klassifikator** (`linear probe`) für `H`, gegebenenfalls mit train-only-PCA zur Dimensionsreduktion. Hyperparameter ausschließlich innerhalb der Trainingsfolds bestimmen.
3. Zusätzlich Probes nur auf `W0`, nur auf `W1` und auf beide Matrizen gemeinsam trainieren. Bei guter Generalisierung ein kleines MLP als sekundären Test nichtlinearer Dekodierbarkeit zulassen.
4. Datensätze **blockweise nach Initialisierungsseed** splitten: Beide historischen Varianten desselben Seeds müssen im selben Train-, Validierungs- oder Testsplit liegen. Sonst memorisiert die Probe möglicherweise Seed-Eigenschaften.
5. Präregistrierter primärer Endpunkt: balancierte Accuracy bzw. ROC-AUC auf vollständig neuen, zurückgehaltenen Seed-Blöcken, nach einer festgelegten C-Trainingsdauer. Die Probe muss deutlich über der binären Zufallsbasis von 50 % liegen.

## Unverzichtbare Kontrollen

- **Gleiche finale Aufgabe und Leistungsniveau:** C-Accuracy und C-Loss beider Gruppen berichten. Nicht stillschweigend ungleich trainierte Gruppen vergleichen. Zusätzlich eine Probe auf ausschließlich Accuracy/Loss und Gewichts-/Aktivierungsnormen trainieren und eine Analyse mit diesen Variablen als Kovariaten durchführen.
- **Seed-Kontrolle:** Historienlabels innerhalb gepaarter Seed-Blöcke permutieren bzw. tauschen und die gesamte Trainings-/Validierungspipeline erneut auswerten. Signifikanz auf Blockebene, nicht auf individueller Gewichtsmatrixebene.
- **Symmetrie-Kontrolle:** Hidden-Unit-Reindexierungen dürfen eine ausgerichtete Prognose nicht ändern. Für spätere allgemeinere Verfahren permutationinvariante Features testen.
- **Zeitliche Persistenz:** Probe nach 1, 5 und 10 gemeinsamen C-Epochen trainieren und evaluieren. Ein Abfall zum Zufallsniveau würde eine endliche Gedächtnisdauer anzeigen; persistente Dekodierbarkeit wäre stärkere Evidenz.
- **Momentums-Kontrolle:** Primär alle Übergangsmomenta zurücksetzen; separat prüfen, ob fortgesetztes Momentum die Befunde verändert.
- **Negativkontrolle:** Eine künstlich zufällige Zuordnung von Historienlabels muss Chance-Leistung liefern. Keine run-spezifischen IDs, Dateinamen, Seed-Werte oder Datenordnungskennzeichen als Features.
- **Funktionale Variante (sekundär):** Gleiche final trainierte Netze auf einem feststehenden Satz von Inputs evaluieren; eine weitere Probe versucht das Historienlabel aus Outputs oder Hidden Activations zu dekodieren. So lässt sich unterscheiden, ob historische Information nur im Gewichtsraum oder auch im Funktionsverhalten sichtbar ist.

## Statistische Auswertung

Berichte Confusion Matrix, balancierte Accuracy, AUC, Effekt gegenüber 50 %, Seed-Block-Konfidenzintervalle und blockweisen Permutations-p-Wert. Testdaten und Hyperparameter vor dem abschließenden Testlauf einfrieren. Pro Seed-Blöcken aggregieren; die beiden Historien aus demselben Seed sind gepaarte Messungen. Beim Vergleich mehrerer C-Horizonte und Layer-Endpunkte das multiple Testen transparent behandeln und einen einzigen primären Endpunkt festlegen.

## Erwartete Dateien und Abbildungen

- `history_run_manifest.csv`: Seed, Initialisierungs-Hash, Aufgabenpermutationen, Aufgabenreihenfolge, Phasenlängen, Lernraten, RNG, Momentum-Reset, finaler Checkpoint.
- `probe_splits.json`: blockweise Train-/Val-/Test-Zuordnung, deterministische Probe-Seeds.
- `history_probe_results.csv`: Features, Layer, C-Horizont, Accuracy/AUC, Konfidenzintervalle, Permutations-p und Kontrollen.
- `history_decoding_by_C_duration.png/pdf`: Dekodierbarkeit nach unterschiedlich langem gemeinsamen C-Training.
- `history_probe_ablation.png/pdf`: nur `W0`, nur `W1`, beide, rein funktionale/diagnostische Baselines.
- `history_decoding_report.md`: Ergebnisse und klare Unterscheidung zwischen verbleibender Gewichtsraumspur und eventuell verbleibender funktionaler Differenz.

## Interpretation

Falls eine Probe in zuvor ungesehenen Seed-Blöcken `A→B→C` von `B→A→C` unterscheiden kann, obwohl die aktuelle Aufgabe C und die Endleistung vergleichbar sind, wäre dies **direkte Evidenz für rekonstruierbare Information über die Reihenfolge vergangener Lernereignisse in finalen Gewichten**. Das beweist keine Rekonstruktion *jedes* SGD-Schrittes und keinen Attraktor. Ein Nullresultat für eine lineare Probe bedeutet zunächst nur, dass dieses Label mit den gewählten Features und Modellen nicht linear dekodierbar war.

## Referenzen

- Krauss et al. (2026): https://doi.org/10.3389/fcogn.2026.1960468
- Schürholt et al. (2021), *Hyper-Representations: Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction*: https://proceedings.neurips.cc/paper_files/paper/2021/file/89562dccfeb1d0394b9ae7e09544dc70-Paper.pdf
- Horwitz et al. (2025), *Learning on Model Weights using Tree Experts*: https://openaccess.thecvf.com/content/CVPR2025/papers/Horwitz_Learning_on_Model_Weights_using_Tree_Experts_CVPR_2025_paper.pdf
