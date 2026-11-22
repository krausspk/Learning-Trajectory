# 07 · Aufgabenhistorie aus Endgewichten dekodieren

Umsetzung von `03_Aufgabenhistorie_aus_Endgewichten_dekodieren.md` als neues, isoliertes Experiment. Der eingefrorene Plan steht in [plan.json](plan.json), die ursprüngliche Aufgabenbeschreibung unter [provenance/experiment_specification.md](provenance/experiment_specification.md).

200 unabhängige Initialisierungen werden jeweils mit den Historien **A→B→C** und **B→A→C** trainiert. A ist die MNIST-Ziffernidentität, B verschiebt die Labels um eins und C um drei modulo zehn. Beide Arme haben dieselbe Initialisierung und Bildreihenfolge. A und B dauern je eine Epoche; C dauert zehn Epochen. Gespeichert und ausgewertet werden C1, C5 und C10.

Der Hauptversuch setzt SGD-Momentum an beiden Phasenübergängen zurück. Eine zusätzliche Kohorte mit denselben 200 Seed-Paaren führt Momentum fort. Insgesamt entstehen 800 Trainingsläufe mit jeweils 60.000 Trainingsbildern pro Epoche. Die 120/40/40 Train-/Validierungs-/Testblöcke und alle Hyperparameter wurden vor Trainingsbeginn festgelegt. Originaldaten und Ergebnisse der Reproduktion werden ausschließlich gelesen.

Der primäre Endpunkt ist die balancierte Testgenauigkeit einer linearen L2-logistischen Probe aus beiden kanonischen, ausgerichteten Gewichtsmatrizen nach C10 im Reset-Versuch. Referenz, Zentrierung, Skalierung und Kovariatenkorrektur werden ausschließlich auf Trainingsblöcken bestimmt. Regularisierung wird auf den Validierungsblöcken ausgewählt. 999 vollständige blockweise Labelpermutationen prüfen den primären Endpunkt; Nebenendpunkte erhalten ebenfalls 999 Permutationen und eine Holm-Korrektur. Konfidenzintervalle resampeln die vollständigen Test-Seed-Paare.

Ergebnisse:

- [Ergebnisbericht](history_decoding_report.md)
- [Ergebnistabelle](history_probe_results.csv)
- [Verlauf über die C-Trainingsdauer](history_decoding_by_C_duration.png)
- [Vergleich der Gewichtsschichten und Kontrollen](history_probe_ablation.png)
- [Trainingsmanifest](history_run_manifest.csv)

Das Experiment wird in der bestehenden lokalen Python-Umgebung ausgeführt. `code/prepare.py` hat den Plan bereits eingefroren und darf nicht über einen vorhandenen Plan schreiben. Die abgeschlossenen Seed-Archive können mit derselben Konfiguration geprüft und wiederverwendet werden. Ausführung und Ergebnisprüfungen werden in `provenance/` dokumentiert.

Vom Projektverzeichnis aus:

```bash
bash 07_History_Decoding/run.sh 4
```

Die Zahl legt die parallelen Trainingsprozesse fest. Das Skript prüft vorhandene vollständige Trainingsarchive, führt die Probe-Auswertung erneut aus und erzeugt Bericht, Abbildungen und Integritätsnachweise. Änderungen an Code, Plan oder Daten werden durch die Archivfingerabdrücke erkannt; abweichende Trainingsarchive werden nicht überschrieben.

Die Methodenprüfung führte vor der ersten Probe-Auswertung zu einer dokumentierten Änderung: 999 statt 199 Permutationen pro Nebenendpunkt, damit die Auflösung der p-Werte für die Holm-Korrektur ausreicht. Originalplan, geänderter Plan und Begründung bleiben in `provenance/` erhalten.

Die vollständige Kohorte enthält 800 Historien und 2.400 gespeicherte C-Checkpoints. Ein gesonderter Zeitmesslauf und die nach einer Prozessunterbrechung wiederholten Analysen sind unter `provenance/` erhalten; sie sind nicht zusätzliche unabhängige Versuchseinheiten. Plan und Zufallsseeds blieben bei der Fortsetzung unverändert.
