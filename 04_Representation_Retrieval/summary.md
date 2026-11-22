# Punkt 3: Initialisierungs-Retrieval aus Repräsentationen

Die vorab festgelegte rohe Hidden-CKA erkennt am Ende des Trainings die zugehörige Initialisierung **nicht konsistent besser als die Zufallszuordnung**. Der gleichgewichtete mittlere Rang über die drei Datensätze beträgt 5.400 (95%-Run-Jackknife-Intervall [4.761, 6.039], einseitiges Permutations-p=0.366863); Top-1 beträgt 6.67%. Das ist ein ausbleibender Nachweis in diesem Test, kein Äquivalenztest und kein Beweis, dass die Repräsentationen keinerlei historische Information enthalten.

## Festgelegtes Experiment

Es wurden ausschließlich die 30 bestehenden Netze der Hauptkohorte verwendet: zehn je MNIST, Fashion-MNIST und KMNIST, jeweils alle 21 gespeicherten Halbepochen-Checkpoints E=0…20. Für jedes Dataset und jeden Checkpoint wurden sämtliche 100 Vergleiche zwischen aktueller Query i und Initialisierungskandidat j bei E=0 berechnet. Kein neues Training, keine Gewichtsraumdistanz und keine Neuronenzuordnung gingen in diese Auswertung ein.

Die gemeinsame gehashte Probe enthält 3.000 Beispiele, 1.000 pro Dataset, in unveränderter Reihenfolge. Primär wird 1−CKA der rohen Hidden-Aktivierungen verglichen. Die biased centered linear CKA ist

`CKA(X,Y) = ||XcᵀYc||²_F / (||XcᵀXc||_F · ||YcᵀYc||_F)`.

Die Zentrierung erfolgt über die Probezeilen jeder Feature-Spalte. Zwei vorab festgelegte Robustheiten verwenden (a) zusätzlich jede nichtkonstante Hidden-Spalte mit Populations-SD=1 und (b) rohe Logits `H @ W1`, deren feste Null-Biases geprüft wurden. Konstante Spalten bleiben null; eine vollständig varianzlose Repräsentation führt zu einem Fehler. Rohe Hidden-CKA ist nicht invariant gegen beliebige positive Skalierung einzelner Hidden-Neuronen; die Standardisierungsvariante kontrolliert genau diese Empfindlichkeit. Logits sind unveränderte Ausgabewerte, keine Softmax-Wahrscheinlichkeiten.

## Primäre Endpunkte bei E=20

Kleinere Ränge sind besser. Bei zehn Kandidaten beträgt der Zufallsmittelrang 5,5 und die Top-1-Chance 10%.

| Dataset | Mittlerer Rang | 95%-Run-JK-Intervall | p, Rang kleiner | Holm-p | Top-1 |
|---|---:|---|---:|---:|---:|
| MNIST | 5.5 | [4.570, 6.430] | 0.541746 | 1.000000 | 0% |
| Fashion-MNIST | 4.8 | [3.519, 6.081] | 0.075392 | 0.226177 | 10% |
| KMNIST | 5.9 | [4.820, 6.980] | 0.845315 | 1.000000 | 10% |

Keine der drei primären Dataset-Hypothesen besteht die Holm-Korrektur. E=0 liefert Selbst-Retrieval mit Rang 1 und ist lediglich ein Sanity-Check ohne p-Werte. Die Verlaufsgrafik zeigt die Abnahme der Identifizierbarkeit nach Beginn des Trainings; andere Checkpoints wurden nicht zu neuen primären Endpunkten erklärt.

## Aggregierte Robustheiten

Die Zusammenfassung mittelt die drei Dataset-Statistiken gleichgewichtet. Die Zusatzvarianten und Top-1/MRR sind explorativ; die unadjustierten p-Werte dieser Tabelle bilden keine zusätzliche bestätigende Testfamilie.

| Repräsentation | Mittlerer Rang | 95%-Run-JK-Intervall | p, Rang kleiner | Top-1 | MRR |
|---|---:|---|---:|---:|---:|
| Rohe Hidden-CKA (primär) | 5.400 | [4.761, 6.039] | 0.366863 | 6.67% | 0.2822 |
| Standardisierte Hidden-CKA | 5.533 | [5.106, 5.960] | 0.608439 | 10.00% | 0.2926 |
| Logit-CKA | 5.600 | [5.208, 5.992] | 0.736326 | 13.33% | 0.3006 |

Weder die standardisierte Hidden-CKA noch die Logit-CKA liefern hier einen konsistenten Vorteil über die Datensätze. Die detaillierten Einzelwerte und Verläufe bleiben in den CSV-Dateien erhalten; es wurde keine Variante anhand eines positiven Ergebnisses ausgewählt.

## Statistische Einheit und Grenzen

Die Netzwerke sind die Replikationseinheiten, nicht die 100 Matrixzellen. Je Dataset wurden 10.000 vollständige bijektive Zuordnungen der Kandidatenidentitäten getestet, mit den festen Seeds 2026112001–2026112003. Die Nullzuordnung verwendet jeweils dieselbe globale Kandidatenpermutation für alle Query-Zeilen. Der gerichtete primäre Test fragt nach kleinerem mittleren Rang. Holm korrigiert die drei rohen Hidden-Endpunkte bei E=20. Unsicherheit stammt aus dem gepaarten Delete-one-Run-Jackknife: Query-Zeile und zugehörige Kandidatenspalte werden gemeinsam entfernt, die Kandidatenränge neu berechnet und für die neun verbleibenden Kandidaten kalibriert. Die Aggregation verwendet unabhängige Permutationen und getrennte Jackknife-Zentrierung je Dataset.

CKA misst die Ähnlichkeit der Repräsentationen auf dieser festen Probe. Eine erfolgreiche Erkennung würde weder beweisen, dass historische Information für die Klassifikation genutzt wird, noch eine kausale Rolle der Initialisierung identifizieren. Das hier ausbleibende Retrieval schließt Information in anderen Merkmalen, Proben oder Verfahren nicht aus. Der Kandidatenraum umfasst nur zehn bekannte Initialisierungen pro Dataset; dies ist keine Rekonstruktion einer unbekannten Initialisierung. Die vorhandene Hauptkohorte koppelt Initialisierungs- und Trainings-/Shuffle-Zufall über die ursprünglichen Run-Seeds und isoliert diese Ursachen nicht experimentell. Die spätere faktorielle Untersuchung ist davon getrennt.

## Nachvollziehbare Artefakte

- [Vorab festgelegte Definitionen](analysis_plan.json), [Distanzdefinitionen](data/distance_definitions.json), [Quell- und Rechenvalidierung](data/validation.json).
- [Dataset-Ergebnisse](retrieval_by_dataset.csv), [Aggregation](aggregate.csv), [einzelne Query-Ränge](per_query_ranks.csv), [Inferenzdiagnostik](inference_diagnostics.json).
- [Berechnung](code/representations.py), [Tests](tests/test_representations.py), [unverändert wiederverwendete Inferenzquelle](provenance/inference_source.json).
- [Grafikübersicht](index.html). PNG und PDF zeigen dieselben berechneten Werte; Intervalle werden nur zur Darstellung auf zulässige Wertebereiche begrenzt, die unveränderten Grenzen stehen in den CSVs.
