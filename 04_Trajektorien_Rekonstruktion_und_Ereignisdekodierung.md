# Experiment 4 — Rekonstruktion früherer Trajektorien und zeitlich lokalisierter Ereignisse aus Endgewichten

## Kernfrage

Wie viel **zeitlich aufgelöste Information** über einen individuellen Trainingsverlauf ist in den **finalen Gewichten allein** rekonstruierbar? Anders als Experiment 1 (RSA der Anfangs-/Endgeometrie) und Experiment 3 (ein binäres Gesamthistorienlabel) soll hier die dekodierbare Vergangenheit als Funktion des *zeitlichen Abstands vom Endzustand* gemessen werden.

Zwei komplementäre Zielgrößen:

1. **Kontinuierliche Trajektorienrekonstruktion:** Frühere niedrigdimensionale Gewichtsraumkoordinaten vorhersagen.
2. **Zeitlich lokalisierte Ereignisrekonstruktion:** Für einzelne frühere Zeitfenster vorhersagen, welche Aufgabe bzw. welcher Trainingsimpuls dort präsentiert wurde.

**Codex-Auftrag:** Erst nach abgeschlossener und validierter Reproduktion von Krauss et al. (2026), [DOI 10.3389/fcogn.2026.1960468](https://doi.org/10.3389/fcogn.2026.1960468), starten. Eigene Ausgaben und neue kontrollierte Trainingsläufe; keine Änderungen an Originalergebnissen. Bereits verfügbare 21 Checkpoints können für einen explorativen Vorversuch genutzt werden, bilden mit nur 30 unabhängigen Runs aber **keine ausreichende Stichprobe** für eine belastbare generalisierende Probe.

## Teil A — Frühere Koordinaten im Gewichtsraum dekodieren

### Datengrundlage und Ausrichtung

- Primär ein homogenes Ensemble gleicher Architektur und gleicher finaler Aufgabe (beispielsweise die neuen C-finalen Runs aus Experiment 3) mit **deutlich mehr unabhängigen Seeds** als die Originalkohorte verwenden.
- Alle Zwischengewichte `W(t_0), …, W(t_T)` im gleichen zeitlichen Raster speichern; zusätzlich genaue Update-Indizes, Lernrate, Trainingseingriffe und Run-IDs.
- Für jedes Modell eine kanonische ReLU-Gauge und eine **aus den Finalgewichten allein** bestimmte Hidden-Unit-Permutation gegen eine im Trainingssplit ausgewählte und eingefrorene Referenz anwenden. Anschließend genau diese finale Permutation **unverändert auf sämtliche früheren Checkpoints derselben Trajektorie** anwenden. Beim Testen keinerlei frühe Gewichte oder Historiendaten zur Ausrichtung heranziehen.
- Train-/Validierungs-/Test-Splits nach **vollständigen unabhängigen Runs/Seeds**; keine zufällige Aufteilung von Checkpoints desselben Runs. Bei gepaarten Historien beide Arme desselben Seeds gemeinsam halten.

### Zielvariablen der Probe

Die ausgerichteten Trajektorien der **Trainingsruns** in einen niedrigdimensionalen PCA-Raum projizieren. PCA-Basis, Zentrierung, Normalisierung und Anzahl Komponenten ausschließlich am Trainingssplit fitten (z. B. vorab geprüfte Dimensionen 8/16/32). Für jeden Run entstehen zeitabhängige Koordinaten `z(t)`.

Die Probe erhält nur den finalen Gewichtsvektor `W(T)` oder dessen feste train-only-Embedding-Darstellung und sagt frühere Koordinaten voraus:

\[
\hat z(t_k)=P_{\theta,k}(W(T)),\qquad t_k<T.
\]

Primär pro Zeitpunkt einen **linear regularisierten Ridge-Regressor** (oder einen gemeinsamen linearen Multi-Output-Regressor) trainieren. Sekundär ein kleines MLP mit identischen Splits. Als schwierigere Variante die zurückliegenden **Verschiebungen** `z(t_k)-z(T)` vorhersagen, damit die Probe nicht nur die bekannte finale Position wiedergibt.

### Bewertungsmaße und Baselines

- Hold-out-`R²`, MSE/MAE für jede Zielzeit, sowohl absolut als auch für Verschiebungen `z(t_k)-z(T)`.
- Gegen mindestens zwei Baselines vergleichen: **Trainingsmittelwert** des früheren Zustands sowie **Persistenzbaseline** `z(t_k)≈z(T)`; das Letztere ist für späte Zeiten eine starke triviale Vorhersage.
- Optional Abstand zwischen rekonstruierter und wahrer **gesamter niedrigdimensionaler Trajektorie**, Weglänge und Wendepunkten sowie paarweises RSA der rekonstruierten Trajektorien untereinander.
- Früheste Zeitpunkte und der Startzustand `t_0` sind besonders wichtig. Hohe Vorhersagegüte nur wenige Schritte vor `T` kann durch die Kontinuität des Gradientenverfahrens erklärt werden und ist weniger aussagekräftig.
- Den Endpunkt `t=T` **nicht als Rekonstruktionsziel in den primären Score aufnehmen**; er ist als Probe-Eingabe bekannt.

## Teil B — Zeitlich lokalisierte Ereignisse dekodieren

### Balanciertes Ereignis-Curriculum

Ein kontrolliertes Curriculum mit beispielsweise acht kurzen aufeinanderfolgenden Zeitfenstern vor einer gemeinsamen finalen C-Phase erzeugen. Wie in Experiment 3 A und B als unterschiedlich permutierte zehnklassige MNIST-Labelregeln definieren; die Netze nutzen damit immer dieselbe Input- und Output-Dimensionalität.

Die acht Zeitfenster in vier aufeinanderfolgende **Paare** aufteilen. Jedes Paar enthält **genau einmal A und einmal B**; ein unabhängig gezogenes Bit entscheidet ausschließlich über die Reihenfolge:

- `b_k=0` → in Paar `k`: A, dann B.
- `b_k=1` → in Paar `k`: B, dann A.

Eine Trajektorie kann zum Beispiel `AB | BA | BA | AB | C...` sein, mit Zielvektor `b=(0,1,1,0)`. Durch dieses Design sehen alle Netze insgesamt **gleich häufig A und B**. Die Bits kodieren daher die zeitliche **Reihenfolge**, nicht die bloße Zahl von A-/B-Darbietungen. Anschließend alle Netze im identischen C-Curriculum auf vergleichbare Endleistung bringen. Zufallsinitialisierung, Bilder, Phasenlängen, Lernraten, Batchplan und Momentumbehandlung wie in Experiment 3 paarweise kontrollieren.

### Dekodieraufgabe

Für jedes Ereignisfenster `k` einen binären linearen Klassifikator trainieren:

\[
\hat b_k=P_{\theta,k}(W(T)).
\]

Die Probe erhält **nur** die finalen Gewichte nach der C-Phase. Je Paar/Bit gelten 50 % Chance bei balancierten Labels. Die Vorhersagegüte kann über die zeitliche Distanz zu `T` aufgetragen werden und ergibt ein **Gedächtnisprofil der früheren Aufgabenreihenfolge**.

Vorteil gegenüber Experiment 3: Ein guter Decoder nur für das letzte Aufgabenpaar, aber nicht für frühe Paare, wäre Evidenz für begrenzte zeitliche Persistenz. Eine robuste Dekodierung früher Paare würde auf längerfristige historische Spuren hindeuten.

## Methodische Kontrollen

- **Nur Endgewichte:** In den Probe-Features weder Zeitindex, Trainingslog, Lernratenverlauf, Aufgabenlabels noch frühere Gewichte verwenden. Verhindern, dass die Nutzung der Checkpoints bei der PCA-/Alignment-Vorbereitung Informationen in den Testsplit leakt.
- **Initialisierungs-Kontrolle:** Verschiedene Geschichtssequenzen bei gleicher Initialisierung ausführen. Seed-Blöcke komplett aus dem Testtraining heraushalten.
- **Finale Leistung:** C-Testaccuracy und Loss beider/aller Historien dokumentieren; Probes allein auf diesen Diagnostika, Layernormen und ggf. festen Hidden-Aktivierungsstatistiken als Baselines trainieren.
- **Zeitfenster-Balance:** Jedes Paar enthält stets genau ein A und ein B; Bits und Positionen über Seeds ausgeglichen. Ein Decoder darf nicht allein die Gesamthäufigkeit von A/B nutzen.
- **Statistik:** Labelpermutation auf unabhängigem Seed-/Historienblock, keine Behandlung mehrfacher Zeitpunkte desselben Runs als unabhängige Samples. Für mehrere `t_k`/Bits primären Zeitpunkt vorab festlegen und zusätzliche Tests mittels Holm oder geeigneter Familie kontrollieren.
- **Robustheit:** Kanonische Gauge, permutationkonsistente Eingaben, alternative Feature-Ebenen (`W0`, `W1`, beide), lineare versus kleine nichtlineare Probe und ggf. alternative PCA-Dimensionen.
- **Spezielle Negativkontrolle:** Wird eine frühere Position durch willkürliche unabhängig zufällige Targets ersetzt oder werden vergangene Labels über Run-Identitäten permutiert, muss die Vorhersage gegen ihre jeweilige Baseline fallen.

## Experimentelle Abfolge

1. **Exploration mit bestehenden Checkpoints:** Nichts Neues trainieren; PCA-/Ridge-Pipeline, Leakage-Checks und Visualisierungen auf den alten Runs entwickeln. Wegen `n=30` keine starken Generalisierungsbehauptungen.
2. **Kontrollierte größere Kohorte:** Mit dem A/B/C-Design aus Experiment 3 viele unabhängige Startzustände und ihre vollständigen Zwischenzustände erzeugen, zuerst Teil A und die ein-Bit-Aufgabenhistorie validieren.
3. **Multifenster-Sequenzen:** Vier balancierte AB/BA-Paare + finale C-Phase; Probe für die vier historischen Bits trainieren und Gedächtnisprofil auswerten.
4. **Transfer:** Optional auf andere Zielaufgaben, Datensätze und rekurrente Netze erweitern; dies ist **nicht** Teil des primären Nachweises.

## Ergebnisdateien

- `trajectory_run_manifest.csv`: alle Seeds, vollständige `A/B`-Bitsequenzen, Phasenlängen, Checkpoint-IDs, RNG, gemeinsame finale Aufgabe und Endperformance.
- `trajectory_probe_splits.json`: reproduzierbare gruppierte Splits, train-only-PCA-/Alignment-Referenz.
- `trajectory_regression_results.csv`: Zeitpunkt, PCA-Dimension, Layer, Modellklasse, MSE, R², Baselines, CI.
- `trajectory_R2_vs_lag.png/pdf`: Vorhersagbarkeit früherer Trajektorienpositionen gegen zeitlichen Abstand zum Endzustand.
- `event_decoding_vs_age.png/pdf`: Accuracy/AUC einzelner historischer Bits nach ihrem Alter, einschließlich Chance und Unsicherheiten.
- `trajectory_examples.png/pdf`: wahre versus vorhergesagte niedrigdimensionale Pfade für **ausschließlich zurückgehaltene** Runs.
- `trajectory_decoding_report.md`: saubere Trennung zwischen dekodierbaren Eigenschaften einzelner Vergangenheitsschritte, teilweise rekonstruierter Pfadgeometrie und einer **nicht nachgewiesenen** exakten Rekonstruktion der vollständigen hochdimensionalen Trainingshistorie.

## Interpretation

Positive Ergebnisse der Probe für frühe Zustandskoordinaten oder frühe Aufgabenereignisse würden zeigen, dass finale Gewichte noch **rekonstruierbare, zeitlich strukturierte historische Information** enthalten. Das würde die Historical Trace Hypothesis wesentlich direkter stützen als Endpunkt-Clustering allein. Es wäre jedoch kein Beweis, dass sich eine vollständige, eindeutige SGD-Trajektorie aus einem Endzustand invertieren lässt: Unterschiedliche Vergangenheiten können auf denselben oder nahezu denselben Endzustand abgebildet werden.

## Referenzen

- Krauss et al. (2026): https://doi.org/10.3389/fcogn.2026.1960468
- Schürholt et al. (2022), *Model Zoos: A Dataset of Diverse Populations of Neural Network Models*: https://arxiv.org/abs/2209.14764
- Schürholt et al. (2021), *Hyper-Representations*: https://proceedings.neurips.cc/paper_files/paper/2021/file/89562dccfeb1d0394b9ae7e09544dc70-Paper.pdf
