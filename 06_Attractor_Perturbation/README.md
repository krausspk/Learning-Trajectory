# 06 · Attraktorprüfung durch Gewichtsraumstörung

Funktionale Eignung am Endpunkt, ohne gesicherten strukturellen Rückkehrnachweis.

Ausführung der Vorgabe [02_Attraktor_Nachweis_durch_Stoerung.md](provenance/experiment_specification.md) in diesem eigenständigen Unterordner. Originalarchive unter `../outputs/` dienen nur als Quellen.

## Ergebnisse

- [Schlussfolgerung](attractor_conclusion.md)
- [Basin-Analyse mit Methodik, Kontrollen und Grenzen](basin_analysis.md)
- [Recovery-Kurven als PNG](recovery_curves.png) · [PDF](recovery_curves.pdf)
- [Regionzugehörigkeit und Rückkehr als PNG](return_probability.png) · [PDF](return_probability.pdf)
- [Messwerte je Zeitpunkt](recovery_metrics.csv), [Endpunkte](endpoint_metrics.csv), [Seed-Level-Intervalle](endpoint_summary.csv)
- [Fork-Manifest](perturbation_manifest.csv), [Konfiguration](config.json), [vorab eingefrorene Referenzschwellen](region_thresholds.json)

## Reproduzieren

Vom Projektverzeichnis aus, mit der vorhandenen Python-Umgebung:

```bash
.venv/bin/python 06_Attractor_Perturbation/code/run.py prepare
.venv/bin/python 06_Attractor_Perturbation/code/run.py references --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/run.py calibrate
.venv/bin/python 06_Attractor_Perturbation/code/run.py pilot --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/run.py analyze --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/run.py verify
```

Vorhandene Läufe werden nur bei passenden Fingerabdrücken wiederverwendet. Konfiguration und Schwellen werden nicht anhand der Störresultate angepasst. `checkpoints/` enthält separate Fork-Gewichte sowie Optimizer-/RNG-Zustände. `code/` enthält die vollständige lokale Implementierung; `provenance/` dokumentiert Vorgabe und Quellen; `validation/` enthält die Prüfungen.

## Umfang

300 gestörte Hauptläufe, 10 unabhängige MNIST-Ausgangsnetze, drei Störstärken, fünf Richtungen, zwei Momentum-Protokolle und zehn weitere Epochen; zusätzlich ungestörte und SGD-Rausch-Kontrollen. 10 unabhängige Referenznetze bestimmen die Region. Die vollständig ausgeführte Erweiterung über drei Datensätze, drei Störzeitpunkte, drei Schichtvarianten und vier nichttriviale Amplituden wird im [erweiterten Ergebnisbericht](expanded_analysis.md) separat ausgewertet. Die Zahlen dieses Pilotabschnitts beziehen sich ausschließlich auf die fünf Richtungen des MNIST-Piloten.


## Erweiterung auf alle Datensätze und Zeitpunkte

Zusätzlich zum Pilot wurden **2.160 Störläufe plus 180 Baselines** über drei Datensätze, drei Checkpoints, drei Schichtvarianten, vier Störstärken und beide Momentum-Protokolle vollständig ausgeführt. Jede Zelle enthält zehn Ausgangsnetze mit einer Richtung je Netz; die fünf Richtungen des Pilots bleiben separat ausgewertet. 0 numerische Ausfälle.

Bei gemeinsamer Störung beider Matrizen liegen 0/72 deskriptive r-Intervalle vollständig unter 1, 72/72 vollständig darüber. Damit ist die Aussage bedingungsspezifisch; eine allgemeine Rückkehr zur individuellen Kontrolltrajektorie oder eine universelle Basin-Grenze folgt daraus nicht. [Erweiterter Ergebnisbericht mit Abbildungen](expanded_analysis.md).

Alle 2160/2160 gestörten Ausgangszustände waren bereits innerhalb der festen strukturellen Region. Eine strukturelle Rückkehr nach unmittelbarem Austritt ist damit in dieser Erweiterung nicht getestet. Die 115 Rückkehren in die kombinierte Region betreffen funktionale Kriterien. [Matching-Robustheit](alignment_robustness.md).


## Abschlussprüfungen

- 22 Tests bestanden: [Testprotokoll](validation/unit_tests.log).
- 80 Pilot-/Erweiterungs-Wiederholungen an 1.200 Zuständen bitgleich: [Audit](validation/pilot_expanded_overlap.json).
- 60 Wiederholungen der Originaltrajektorie über alle 30 Ausgangsnetze bitgleich: [Audit](validation/original_trajectory_replays.json).
- 2.430 Archive der Erweiterung einschließlich Referenzen geprüft: [Integrität](validation/expanded_checks.json).
- Alle 421 Artefakte der Original-Reproduktion unverändert: [Audit](validation/original_reproduction_audit.json).
- [Gesamtmanifest aller 2.720 Versuchsfortsetzungen](all_perturbation_manifest.csv).

## Erweiterung reproduzieren

Vom Projektverzeichnis aus, nach abgeschlossenem Pilot und `code/validation.py`:

```bash
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py prepare
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py references --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py calibrate
.venv/bin/python 06_Attractor_Perturbation/code/expanded.py train --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py analyze --workers 8
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py verify
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py report
.venv/bin/python 06_Attractor_Perturbation/code/finalize.py manifest
python3 scripts/audit_results.py --output 06_Attractor_Perturbation
```

Die Original-Reproduktion bleibt unverändert. Eigene vollständige Optimizerzustände liegen jeweils unter `checkpoints/`; SHA-256-Receipts und das abschließende Artefaktinventar sichern Herkunft und Vollständigkeit. Die Ausführungsunterbrechung des Pilots und seine vollständige Wiederaufnahme sind separat unter `provenance/execution_interruption.json` dokumentiert.
