# 01_RSA: Persistiert die Geometrie der Initialisierungen?

Das vorgegebene Kriterium eines robusten, über die drei Datensätze reproduzierbaren positiven Zusammenhangs ist nicht erfüllt.

## Beobachtung

Die neue Analyse verwendet ausschließlich die abgeschlossene und geprüfte Hauptkohorte der Paper-Reproduktion: 30 gespeicherte Netze, zehn je Datensatz, 21 Checkpoints von E0 bis E20 (zehn Epochen). Es wurde nicht neu trainiert. Die optionale separate Frühtrainingskohorte wurde nicht einbezogen.

Primärer Endpunkt ist die Spearman-Korrelation der 45 paarweisen Anfangs- und Endabstände **innerhalb jedes Datensatzes**. Die Abstände sind die gemeinsame W0/W1-Frobenius-Distanz nach positiver ReLU-Gauge-Normierung und global optimaler gemeinsamer Hungarian-Zuordnung. Dieselbe Definition gilt an jedem Checkpoint.

| Datensatz | Spearman ρ | 95%-JK-Intervall | p positiv | p positiv, Holm | p zweiseitig |
| --- | --- | --- | --- | --- | --- |
| MNIST | 0.1165 | [-0.568, 0.801] | 0.2935 | 0.5869 | 0.5961 |
| Fashion-MNIST | -0.1806 | [-0.573, 0.212] | 0.8401 | 0.8401 | 0.3138 |
| KMNIST | 0.2764 | [-0.152, 0.705] | 0.0490 | 0.1470 | 0.0911 |

Über die drei Datensätze ergibt die gleichgewichtete Fisher-z-Zusammenfassung eine zurücktransformierte Korrelation von **0.073**, ein 95%-Netzwerk-Jackknife-Intervall von [-0.235, 0.380] und einseitiges Permutations-p = 0.2535. Das ist eine sekundäre Zusammenfassung; die Einzelergebnisse bleiben entscheidend.

Bereits bei E1 (einer halben Epoche) sinkt die primäre RSA auf MNIST 0.425, Fashion-MNIST 0.367 und KMNIST -0.211. Danach unterscheiden sich die Verläufe; Fashion-MNIST wechselt im späteren Training das Vorzeichen. Ein gemeinsames stabiles positives Plateau wird damit nicht beobachtet.

Die normierte Gesamtgeometrie (Cosine bzw. monoton äquivalente normierte Frobenius-Distanz) zeigt am Endpunkt nur schwache Zusammenhänge: MNIST 0.052, Fashion-MNIST 0.057, KMNIST 0.130. Die Anfangsdistanzen korrelieren dagegen deutlich mit der Summe der initialen kanonischen Gesamtnormen (ρ = 0.602 bis 0.791). Auch Schichten und festgehaltene Alignments liefern datensatzabhängige Ergebnisse; sie stützen keine einheitliche positive Persistenz über alle drei Datensätze.

## Statistische Evidenz

Je Datensatz wurden **10.000 Identitätspermutationen** gerechnet: gleichzeitige Zeilen- und Spaltenpermutation der Vergleichs-RDM, bei festgehaltener Anfangs-RDM. Die Seeds sind MNIST 2026111601, Fashion-MNIST 2026111602 und KMNIST 2026111603. p = (1 + Anzahl mindestens so extremer Nullwerte) / 10.001; kleinster darstellbarer p-Wert ist 0,000100. Die gerichtete Hypothese testet einen positiven Zusammenhang; zweiseitige Werte sind zusätzlich angegeben. Holm korrigiert die Familie der drei primären Datensatztests.

Die Replikationseinheiten sind **zehn Netzwerke je Datensatz, nicht 45 unabhängige Paare**. Der Jackknife entfernt jeweils ein Netz mit allen neun zugehörigen Paaren und berechnet die Ränge auf den verbleibenden 36 Paaren neu. Normalapproximierte Intervalle mit nur zehn Netzen sind grob; sie sind weder simultane Konfidenzbänder noch ein Äquivalenztest. Die gespeicherten Intervallgrenzen sind unbeschnitten; Figuren begrenzen sie auf den Korrelationsbereich.

Für E0 gilt überall ρ = 1; dafür werden keine inferenziellen p-Werte berichtet. Die Nullmittelwerte der drei primären Endpunkte liegen bei MNIST -0.0024, Fashion-MNIST -0.0013, KMNIST +0.0011. Eine einzelne Identitätspermutation muss nicht exakt null ergeben. Die Nullverteilung und ihr Monte-Carlo-Standardfehler sind gespeichert.

## Interpretation

Die beobachtete paarweise Anfangsgeometrie liefert unter dem festgelegten primären Protokoll keine konsistente positive Vorhersage der Endgeometrie über alle drei Aufgaben. Das Ergebnis schließt historische Information in anderen Merkmalen der Endgewichte nicht aus. Es zeigt weder die Rekonstruierbarkeit einer vollständigen Lerntrajektorie noch einen dynamischen Attraktor.

Die Anfangsabstände sind stark konzentriert: ihre Variationskoeffizienten liegen zwischen 0.58% und 0.75%. Numerische Rangstabilität wurde unabhängig mit float32- und float64-Rechnung geprüft; das ersetzt keine Replikation mit neuen Seeds. Metrik-, Schicht-, Norm- und Alignment-Kontrollen sind explorativ und in [rsa_robustness.md](rsa_robustness.md) vollständig dokumentiert. Signifikante Einzelkontrollen wären keine Bestätigung über unabhängige Datensätze und kein Grund, das primäre Ergebnis zu ersetzen.

## Ergebnisse öffnen und weiterverwenden

- [Alle Abbildungen mit PNG/PDF-Links](index.html)
- [Vollständige RSA-Tabelle](rsa_by_dataset.csv): 3276 Zeilen, 52 Varianten × 21 Checkpoints × drei Datensätze.
- [Alle Endpunktvarianten](rsa_endpoints_all_metrics.csv), [Fisher-z-Zusammenfassung](aggregate.csv), [Robustheitsbericht](rsa_robustness.md).
- [Konfiguration und Provenienz](rsa_config.json), [Distanzdefinitionen](data/distance_definitions.json), [unabhängiger Rechenaudit](provenance/independent_audit.json).
- `data/*_distances.npz`: alle RDMs, Paarzuordnungen, Run-IDs, Seeds, Normen und Neuronenpermutationen.
- `data/*_inference.npz` und `data/*_controls.npz`: Identitätspermutationen, Nullverteilungen und Jackknife-Replikate.

Diese RSA ist eine zusätzliche Analyse nach der neuen Spezifikation, kein im Originalpaper berichtetes Experiment. Die ursprünglichen Trainings-, Ergebnis- und Figurendateien bleiben erhalten; [Prüfsummenvergleich](provenance/preservation_check.json).
