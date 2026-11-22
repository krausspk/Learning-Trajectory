# Learning Trajectory – Reproduktion

Vollständig neu berechnete Reproduktion von Krauss et al. (2026), **Task-dependent clustering in permutation-aligned neural network weights**, DOI [10.3389/fcogn.2026.1960468](https://doi.org/10.3389/fcogn.2026.1960468).

Das primäre Erfolgskriterium ist die qualitative Übereinstimmung mit dem Paper. Berichtete Zahlen werden ausschließlich zum Vergleich verwendet. Alle Trainings- und Analyseergebnisse entstehen aus den heruntergeladenen Originaldatensätzen; es werden keine Paper-Zahlen als Messwerte eingesetzt.

## Ergebnisse lesen

- [Reproduktionsbericht mit Abbildungen](outputs/REPRODUCTION_REPORT.md)
- [Vollständiger numerischer Vergleich](outputs/summary/paper_comparison.csv)
- [Qualitative Befundprüfung](outputs/summary/qualitative_checks.json)
- [Technische und mathematische Validierung](outputs/validation/checks.json)
- [Abbildungsübersicht](docs/figure_inventory.md)
- [Versuchsprotokoll und offene Implementierungsdetails](docs/reproduction_spec.md)

## Ausführen

Voraussetzung: `uv`, Netzwerkzugriff für die erstmalige Einrichtung und ungefähr 4 GB freier Speicher für Daten und Ergebnisse, zusätzlich die Python-Umgebung. Die Referenzumgebung wird lokal unter `.venv` eingerichtet. Das Projekt benötigt keine GPU.

```bash
bash scripts/setup.sh
bash scripts/reproduce.sh --workers 4
```

Das Training umfasst **65 vollständige Läufe**: 30 Hauptläufe, 20 Läufe der Aufgaben-Kontrolle und 15 Frühtrainingsläufe. Datensätze werden nur beim ersten Durchlauf heruntergeladen. Vollständige, passende Trainingsarchive werden validiert und wiederverwendet; Analyse-Checkpoints werden nur bei passenden Quellcode-/Eingabefingerabdrücken wiederverwendet. Die Validierung und Abbildungserstellung werden erneut ausgeführt. Bestehende, abweichend konfigurierte Trainingsarchive werden nicht stillschweigend überschrieben.

Einzelne Phasen:

```bash
.venv/bin/python -m learning_trajectory.cli download
.venv/bin/python -m learning_trajectory.cli train --workers 4
.venv/bin/python -m learning_trajectory.cli analyze --workers 4
.venv/bin/python -m learning_trajectory.cli validate --workers 4
.venv/bin/python -m learning_trajectory.cli sensitivity
.venv/bin/python -m learning_trajectory.cli figures
.venv/bin/python -m learning_trajectory.cli report
.venv/bin/python -m learning_trajectory.cli manifest
.venv/bin/python -m learning_trajectory.cli status
.venv/bin/python -m pytest -q
python3 scripts/audit_results.py
```

Für einen vollständig unabhängigen erneuten Durchlauf ein neues Ziel wählen:

```bash
bash scripts/reproduce.sh --output outputs_replication_2 --workers 4
```

Die Modellkonfiguration folgt dem Paper: 784→64→10 (Parität: 2 Ausgänge), ReLU, fest auf null gesetzte Biases, SGD mit Lernrate 0,1 und Momentum 0,9, Batchgröße 256, zehn Epochen beziehungsweise zwei Epochen in der Frühkohorte. Alle Seed-Listen und Analysezeitpunkte stehen im Code und in den Konfigurationsdateien.

## Gespeicherte Forschungsdaten

| Pfad | Inhalt |
|---|---|
| `data/{dataset}/` | Original-IDX-Dateien, gzip-Downloads, Quellen, SHA-256 und Normalisierung |
| `outputs/checkpoints/{main,task,early}/*.npz` | 1.290 Checkpoints insgesamt: `W0`, `W1`, Momentum-Matrizen, feste Biases, Checkpoint-/Update-Indizes, Testgenauigkeit, Test-Loss und Metadaten |
| Gleichnamige `*.pt` | An jedem Checkpoint Modell-/Optimiererzustand, Torch-/NumPy-/Python-RNG, aktuelle Epoche, Batchposition, vollständige Bildreihenfolge und Generatorzustand |
| `outputs/manifests/` | Run-Konfiguration, Archivpfade und Hashes beider Checkpointformate |
| `outputs/probes/` | Exakte gemeinsame Probe-Matrizen, Labels, Quelldatensatz, Bildindizes und Hashes |
| `outputs/activations/{cohort}/` | Hidden-Aktivierungen auf dem festgelegten Probe-Set an jedem analysierten Checkpoint; Run-Reihenfolge liegt bei |
| `outputs/alignments/{cohort}/` | Alle verwendeten paarweisen Permutationen, Zuordnungsmethode und Run-Indizes |
| `outputs/pairs/{cohort}/` | Vollständige paarweise Messwerte, ein CSV pro Analysezeitpunkt, mit Provenienz-Receipt |
| `outputs/summary/` | Effekte, Konfidenzintervalle, Jackknife-Replikate, Training, Frühdynamik, Vorzeichenstabilität, Gauge- und Paper-Vergleiche |
| `outputs/validation/` | Invarianz, Umnummerierung, Integrität, unabhängige Nachrechnungen, synthetisches Netzwerk inklusive Gewichten/Permutationen |
| `outputs/sensitivity/` | Zusätzliche nachträgliche Prüfung der frühen Assignment-Variante auf identischen gespeicherten Gewichten; ersetzt keine Hauptanalyse |
| `outputs/figures/` | Alle Hauptfiguren 1–6 und Zusatzfiguren S1–S10 als PNG und SVG |
| `outputs/tables/` | Tabellen S1–S9 mit veröffentlichten und neu berechneten Werten |
| `outputs/provenance/` | Umgebungsinformationen und eingefrorene Kopie der verwendeten Quelltexte/Konfigurationen |
| `outputs/artifact_manifest.json` | SHA-256-Inventar sämtlicher Ergebnisdateien |

Haupt- und Aufgabenläufe speichern je 21 Checkpoints `E=0…20`; `E` ist eine halbe Epoche. Die Frühkohorte speichert die Updates `0,1,2,4,8,16,32,48,64,96,128,160,192,235,352,470`. Die Aufgabenanalysen nutzen wie im Paper nur `E=0,1,2,5,10,20`; die übrigen gespeicherten Trainingscheckpoints bleiben für spätere Analysen erhalten.

Die Inferenz verwendet vollständige Runs als unabhängige Einheiten, bei der Aufgaben-Kontrolle vollständige gepaarte Seed-Blöcke. Netzwerkpaare werden nicht als unabhängige Stichproben behandelt.

Die technische Validierung setzt außerdem einen tatsächlich gespeicherten M1-Checkpoint nach einer halben Epoche fort. Nach den verbleibenden 117 Updates werden beide Gewichtsmatrizen bitweise mit dem archivierten nächsten Checkpoint verglichen.

## Später weiterarbeiten

Gewichte und Verlauf laden:

```python
import json
import numpy as np

with np.load('outputs/checkpoints/main/M1.npz', allow_pickle=False) as run:
    metadata = json.loads(str(run['metadata'].item()))
    W0 = run['W0'][10]  # 784 × 64 bei E10
    W1 = run['W1'][10]  # 64 × 10
    accuracy = run['accuracy'].copy()
```

Modell und Momentum für ein Folgeexperiment wiederherstellen:

```python
from learning_trajectory.checkpoints import restore_checkpoint

model, optimizer, state = restore_checkpoint(
    'outputs/checkpoints/main/M1.pt', checkpoint=10
)
# state['epoch'], state['batch_position'], state['epoch_order']
# spezifizieren exakt, wo die ursprüngliche Trajektorie fortgesetzt wird.
```

Die Gewichte verwenden im NPZ die Paper-Orientierung `W0[input, hidden]`, `W1[hidden, output]`; im PyTorch-State-Dictionary die üblichen transponierten Linear-Gewichte. Eine gespeicherte Alignment-Permutation `p` bildet Netzwerk B auf A ab: `B0[:, p]`, `B1[p, :]`. Bei neuen Aufgaben, Störungen oder Hyperparametern die Originalarchive beibehalten und Ergebnisse in einem neuen Verzeichnis speichern.

## Dokumentierte Rekonstruktionsentscheidungen

Einige Details fehlen in der Publikation: die genaue Ableitung der Shuffle-Seeds, der Halbepochen-Batch bei ungerader Batchzahl, die NumPy-RNG-API für das Probe-Set, die Gewichtsdarstellung des PIF-Benchmark-Kandidaten sowie Seeds und Initialisierung des synthetischen Tests. Die gewählten Konventionen stehen in den Metadaten und der [Spezifikation](docs/reproduction_spec.md). Das Projekt behauptet daher keine bitidentische Wiederholung unveröffentlichter Originalskripte.

Python 3.11/Linux und Matplotlib 3.7.5 ersetzen Python 3.9.5/Windows und Matplotlib 3.4.3. NumPy 1.23.5, SciPy 1.10.1 und PyTorch 2.8.0 CPU entsprechen den numerischen Hauptbibliotheken des Papers. Training verwendet float32; Vergleiche und Statistik verwenden float64. Alle installierten Versionen sind in `requirements.lock` festgehalten.
