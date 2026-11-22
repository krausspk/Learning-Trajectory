# 08 · Trajektorienrekonstruktion und Ereignisdekodierung

Ausführung der Aufgabenstellung [04_Trajektorien_Rekonstruktion_und_Ereignisdekodierung.md](experiment_instructions.md) in einem eigenen Unterordner. Originaltraining, Checkpoints und Ergebnisse außerhalb dieses Ordners werden nur gelesen.

[Ergebnisbericht](trajectory_decoding_report.md) · [Figurengalerie](plots/index.html) · [Regressionsmesswerte](trajectory_regression_results.csv) · [Ereignisprobes](event_decoding_results.csv)

## Design

Die Exploration nutzt die 30 bestehenden Hauptläufe mit 21 Checkpoints. Sie dient der technischen Entwicklung und erlaubt wegen der kleinen, heterogenen Stichprobe keine starke Generalisierungsbehauptung.

Zwei getrennte kontrollierte MNIST-Kohorten enthalten je 200 unabhängige Initialisierungsseeds mit zwei gepaarten Historien, insgesamt 800 neu trainierte Netze:

- `one_bit`: A→B→C versus B→A→C, insgesamt sieben Epochen und 15 Checkpoints pro Lauf.
- `multi_bit`: vier AB/BA-Paare, gefolgt von C; zwei komplementäre Vier-Bit-Historien pro Seed, insgesamt 13 Epochen und 27 Checkpoints pro Lauf.

A verwendet die ursprünglichen zehn Ziffernlabels, B `(y+1) mod 10`, C `(y+3) mod 10`. Jedes A/B-Fenster dauert eine volle Epoche; C dauert fünf Epochen. Beide Arme teilen Initialisierung, Bilder und Batchplan. Die Null-Biases bleiben eingefroren; Momentum wird an jedem Phasenübergang zurückgesetzt. Alle Modelle verwenden 784–64–10, ReLU und SGD mit Lernrate 0,1, Momentum 0,9 und Batchgröße 256.

Je kontrollierter Kohorte liegen 120 Seeds im Trainings-, 40 im Validierungs- und 40 im Testsplit. Beide Arme und sämtliche Checkpoints eines Seeds bleiben zusammen. Referenz, Skalierung und PCA werden auf Trainingsdaten bestimmt. Eine ausschließlich aus Finalgewichten bestimmte Hidden-Unit-Permutation bleibt über die gesamte Trajektorie eines Modells gleich.

Die primäre Regressionsprobe ist Ridge auf acht PCA-Zielkoordinaten; Startzustand t₀, beide Gewichtsmatrizen, absolute Position ist der primäre Endpunkt. Verschiebungen, 16/32 Komponenten, einzelne Layer und ein kleines MLP ergänzen die Analyse. Ereignisse werden mit linearen Ridge-Klassifikatoren dekodiert; das früheste Aufgabenpaar und beide Gewichtsmatrizen bilden den primären Ereignisendpunkt. Diagnoseprobes, zufällige Labels/Zielkoordinaten, Seed-Block-Bootstraps und Permutationstests sind gesondert ausgewiesen. Ein bekannter Endpunkt ist kein Rekonstruktionserfolg.

Die MLP-Eingabe ist ein ausschließlich auf finalen Trainingsgewichten bestimmtes PCA-Embedding mit bis zu 32 Dimensionen; bei 18 explorativen Trainingsläufen wird es auf 17 Dimensionen begrenzt. Die Verschiebungskontrolle `mean_train[z(t)] − z(T)` trennt die Information des bekannten Endpunkts von der Vorhersage eines individuellen früheren Zustands. Ihr MSE entspricht genau der absoluten Trainingsmittelwertbaseline.

Beide kontrollierten Curricula und Seed-Listen wurden vor kontrollierter Auswertung festgelegt. Die Ein-Bit-Kohorte wird zuerst analysiert; das Vier-Bit-Training ist nicht an einen positiven Ein-Bit-Befund gekoppelt. Änderungen aufgrund kontrollierter Testergebnisse wurden nicht vorgenommen.

## Erneut ausführen

Die vorhandene Python-Umgebung und die bereits heruntergeladenen MNIST-Daten des übergeordneten Projekts werden verwendet. Befehle vom Projektstamm aus:

```bash
bash 08_Trajectory_Decoding/run_all.sh 4
```

Das Skript führt Tests, Exploration, beide kontrollierten Trainings-/Analysestufen, die nachträgliche C-Machbarkeitsdiagnose, den unabhängigen abschließenden Audit und die Berichterstellung aus. Die optionale Zahl ist die Zahl paralleler Trainingsprozesse. Die Diagnose ist deskriptiv; sie ändert keine Trainings- oder Probeentscheidungen. Einzelne Schritte:

```bash
.venv/bin/python -m pytest -q 08_Trajectory_Decoding/tests
.venv/bin/python 08_Trajectory_Decoding/code/analyze.py --cohort exploratory
.venv/bin/python 08_Trajectory_Decoding/code/train_cohorts.py --cohort one_bit --workers 6
.venv/bin/python 08_Trajectory_Decoding/code/analyze.py --cohort one_bit
.venv/bin/python 08_Trajectory_Decoding/code/train_cohorts.py --cohort multi_bit --workers 6
.venv/bin/python 08_Trajectory_Decoding/code/analyze.py --cohort multi_bit
.venv/bin/python 08_Trajectory_Decoding/code/diagnose_feasibility.py \
  --pair-score 08_Trajectory_Decoding/analysis/one_bit/event_both.npz \
  --pair-score 08_Trajectory_Decoding/analysis/multi_bit/event_both.npz
.venv/bin/python 08_Trajectory_Decoding/code/validate_results.py
.venv/bin/python 08_Trajectory_Decoding/code/audit_outputs.py
.venv/bin/python 08_Trajectory_Decoding/code/reporting.py
```

Passende, vollständige Trainingsarchive werden anhand ihrer Fingerprints validiert und wiederverwendet. Die Analysen lassen sich nach abgeschlossenem Training zusammen mit `analyze.py --cohort all` erneut berechnen. Standardmäßig wird die große temporäre Datei ausgerichteter Trajektorien nach jeder Kohorte entfernt; `--keep-aligned` behält sie. Die gespeicherten PCA-Basen, Vorhersagen, Permutationen, Splits und Quellcheckpointarchive bleiben erhalten.

Für ein kurzes technisches Training mit gesondertem Ausgabeziel stehen `--limit-blocks` und `--output` zur Verfügung. Solche Probeläufe ersetzen keine vollständige kontrollierte Kohorte. Die Analyse erwartet die in ihrem eigenen Ordner gespeicherten vollständigen Kohorten.

## Dateien

| Pfad | Inhalt |
|---|---|
| `training_plan.json`, `analysis_plan.json` | Eingefrorene Trainings- und Analyseentscheidungen |
| `trajectory_run_manifest.csv` | Run-ID, unabhängiger Seed, historischer Arm und Bitvektor, Curriculum, Hashes, Endleistung, Archivpfade |
| `checkpoints/{one_bit,multi_bit}/` | Gewichte und Momentum an jedem halben/vollen Epochenschritt sowie abschließender PyTorch-Zustand |
| `trajectory_probe_splits.json` | Vollständige Seed-/Run-Splits und Trainingsreferenzen |
| `analysis/{cohort}/` | Train-only-PCA, Ausrichtungspermutationen, Testvorhersagen, Ereignislabels und Permutationsverteilungen |
| `trajectory_regression_results.csv` | R², MSE, MAE, Baselines, Unsicherheiten und Varianten je früherem Zeitpunkt |
| `event_decoding_results.csv` | Accuracy, AUC, Unsicherheiten, Permutationstests, Layer- und Diagnostikkontrollen |
| `final_performance_controls.csv` | Innerhalb von Seeds gepaarte C-Accuracy-/Loss-Differenzen nach Bit, Kohorte und Split; deskriptive Bootstrap-CIs, kein Äquivalenznachweis |
| `validation.json` | Unabhängiger vollständiger Archiv-, Split- und Originalintegritäts-Audit |
| `provenance/multi_bit_feasibility_final.json/.csv` | Nachträgliche vollständige C-Machbarkeitsdiagnose, einschließlich Aktivierungen auf festen 1.024 Testbildern; keine kausale Aussage |
| `provenance/multi_bit_feasibility_final_pair_success_intervals.csv` | Exakte Intervalle für vollständig richtige Seed-Paare; andere Zielgröße als Accuracy pro einzelnem Lauf |
| `run_all.sh` | Vollständiger reproduzierbarer Ablauf einschließlich Tests und Validierung |
| `trajectory_R2_vs_lag.png/.pdf` | Trajektoriengüte nach zeitlichem Abstand |
| `event_decoding_vs_age.png/.pdf` | Dekodierbarkeit der vier Aufgabenpaare nach Alter |
| `trajectory_examples.png/.pdf` | Wahre und vorhergesagte Pfade ausschließlich aus dem Testsplit |
| `trajectory_decoding_report.md` | Messwerte, Interpretation und Grenzen |
| `code/`, `tests/`, `logs/`, `provenance/` | Rechenweg, unabhängige Prüfungen, Protokolle und Fingerprints |

Die Wissenschaftsaussage betrifft rekonstruierbare Koordinaten und Reihenfolgelabels in diesem kontrollierten Curriculum. Eine exakte, eindeutige Inversion der vollständigen SGD-Historie wird damit nicht nachgewiesen.

Die unabhängige Ergebnisprüfung in `provenance/output_audit.json` rechnet 60 primäre Regressionszeitpunkte und 30 lineare Ereignisendpunkte einschließlich Permutations-p-Werten und Holm-Korrektur nach. Der finale Quelltextstand liegt unter `provenance/final_source_snapshot/`; der frühere Stand zum Zeitpunkt des Archivaudits bleibt separat erhalten.
