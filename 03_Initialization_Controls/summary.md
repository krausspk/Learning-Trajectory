# 03: Kontrollen zur Wiedererkennung der Initialisierung

Die vollständige Gewichtskonfiguration bleibt über Normierung und Neuronenumordnung hinweg wiedererkennbar.

## Zentrale Beobachtungen

Das Referenzprotokoll aus Punkt1 wird exakt reproduziert: alle 30 Endnetze erkennen ihre Initialisierung als nächsten Kandidaten. Auch rohe Gesamtgewichte und die Eingangsschicht W0 erreichen in den geprüften Basisprotokollen perfekte Wiedererkennung. Die Schicht W1 verhält sich dagegen stark abhängig vom Matching: In den gespeicherten Neuronenkoordinaten bleibt sie identifizierbar; nach gemeinsamer kanonischer W0/W1-Zuordnung ist ihre eigene Initialisierung häufig weit hinten. **Daraus folgt keine generelle Löschung der Initialisierungsinformation in W1.**

| dataset | layer | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- |
| mnist | W0 | 1.0000 | 1.0000 | 0.0001 |
| mnist | W1 | 7.8000 | 0.0000 | 0.9992 |
| mnist | total | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | W0 | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | W1 | 8.7000 | 0.0000 | 1.0000 |
| fashion_mnist | total | 1.0000 | 1.0000 | 0.0001 |
| kmnist | W0 | 1.0000 | 1.0000 | 0.0001 |
| kmnist | W1 | 6.0000 | 0.2000 | 0.7399 |
| kmnist | total | 1.0000 | 1.0000 | 0.0001 |

Die Kontrolle mit häufig aktiven Neuronen liefert weiterhin starke Wiedererkennung der Gesamtgewichte. Ebenso bleibt sie bei den großen kandidatbezogenen Residuen erhalten. Kleine Residuen oder geringe Aktivität allein erklären das Muster deshalb unter diesen konkreten Kostenprotokollen nicht. Die Kontrollen sind keine kausale Zerlegung der gespeicherten Information.

## Umfang und Statistik

42 vor der Kontrollrechnung festgelegte Varianten × 21 Checkpoints × drei Datensätze; jeweils zehn Query-Netze und zehn Initialisierungen. Das sind 264.600 Distanzwerte und 2.646 Datensatz-Schätzer. Zusätzlich sechs nachträglich ergänzte Normrang-Kontrollen (378 Schätzer), unten begründet. Hauptkohorte und Checkpoints bleiben identisch zu Punkt1. Es wurde nicht neu trainiert und kein funktionaler Aktivierungs-Fingerprint zwischen Anfang und Ende berechnet.

Alle Kontrolltests sind **explorativ**. 10.000 globale bijektive Herkunftslabel-Permutationen je Datensatz, Seeds2026111801/02/03 für die 42 Kontrollen, Seeds2026111901/02/03 für die Zusatzkontrolle. Die Mittelwerte der Ränge werden einseitig gegen kleinere Werte getestet. Es gibt hier keine neue konfirmatorische Holm-Familie; Einzel-p-Werte über viele Varianten sind keine unabhängigen Bestätigungen. E0 bleibt eine Sanity-Prüfung ohne p-Werte.

Die Unsicherheit stammt aus dem gepaarten Netzwerk-Jackknife: Query-Zeile und Initialisierungsspalte desselben Runs entfernen, verbleibende Kandidaten neu rangieren, gegen die bei neun Kandidaten geänderte Zufallserwartung normieren und auf die Zehn-Kandidaten-Skala zurücktransformieren. Die Null-Mappings, Einzelquery-Ergebnisse, Nullarrays und Jackknife-Replikate sind gespeichert. **Bei perfekter Wiedererkennung können die Jackknife-Intervalle zu [1,1] kollabieren; das garantiert keine perfekte Wiedererkennung neuer Netze.**

## Koordinaten, Gauge, Normierung

Vier Protokolle: rohe bzw. kanonische Gewichte, jeweils gespeicherte Koordinaten (`identity`) oder gemeinsames optimales W0/W1-Matching (`optimal`). Zwei Kosten: Frobenius sowie normalisierte Frobenius-Distanz. Bei letzterer normiert ein **gemeinsamer Gesamtfaktor beide Schichten**; layerweise Werte sind Beiträge zu diesem Gesamtvektor, keine separat auf Eins normierten Layer. Die Neuronenzuordnung optimiert beide Schichten gemeinsam und bleibt bei ihrer separaten Auswertung dieselbe. `identity` ist ein Koordinatengedächtnis-Diagnostikum und nicht permutationsinvariant.

| dataset | metric | alignment | gauge | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- | --- |
| mnist | frobenius | identity | raw | 1.0000 | 1.0000 | 0.0001 |
| mnist | normalized_frobenius | identity | raw | 1.0000 | 1.0000 | 0.0001 |
| mnist | frobenius | optimal | raw | 1.0000 | 1.0000 | 0.0001 |
| mnist | normalized_frobenius | optimal | raw | 1.0000 | 1.0000 | 0.0001 |
| mnist | frobenius | identity | canonical | 1.0000 | 1.0000 | 0.0001 |
| mnist | normalized_frobenius | identity | canonical | 1.0000 | 1.0000 | 0.0001 |
| mnist | frobenius | optimal | canonical | 1.0000 | 1.0000 | 0.0001 |
| mnist | normalized_frobenius | optimal | canonical | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | frobenius | identity | raw | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | normalized_frobenius | identity | raw | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | frobenius | optimal | raw | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | normalized_frobenius | optimal | raw | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | frobenius | identity | canonical | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | normalized_frobenius | identity | canonical | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | frobenius | optimal | canonical | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | normalized_frobenius | optimal | canonical | 1.0000 | 1.0000 | 0.0001 |
| kmnist | frobenius | identity | raw | 1.0000 | 1.0000 | 0.0001 |
| kmnist | normalized_frobenius | identity | raw | 1.0000 | 1.0000 | 0.0001 |
| kmnist | frobenius | optimal | raw | 1.0000 | 1.0000 | 0.0001 |
| kmnist | normalized_frobenius | optimal | raw | 1.0000 | 1.0000 | 0.0001 |
| kmnist | frobenius | identity | canonical | 1.0000 | 1.0000 | 0.0001 |
| kmnist | normalized_frobenius | identity | canonical | 1.0000 | 1.0000 | 0.0001 |
| kmnist | frobenius | optimal | canonical | 1.0000 | 1.0000 | 0.0001 |
| kmnist | normalized_frobenius | optimal | canonical | 1.0000 | 1.0000 | 0.0001 |

Damit bleibt die Wiedererkennung der Gesamtgewichte auch nach Ausschalten der globalen Norm und Neuronenumordnung erhalten. Der Befund bezieht sich auf die zehn Kandidaten dieses Datensatzes, nicht auf die vollständige Rekonstruktion unbekannter Initialgewichte.

## Schichtkontrollen

| dataset | layer | alignment | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | W0 | identity | 1.0000 | 1.0000 | 0.0001 |
| mnist | W1 | identity | 1.0000 | 1.0000 | 0.0001 |
| mnist | total | identity | 1.0000 | 1.0000 | 0.0001 |
| mnist | W0 | optimal | 1.0000 | 1.0000 | 0.0001 |
| mnist | W1 | optimal | 7.8000 | 0.0000 | 0.9992 |
| mnist | total | optimal | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | W0 | identity | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | W1 | identity | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | total | identity | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | W0 | optimal | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | W1 | optimal | 8.7000 | 0.0000 | 1.0000 |
| fashion_mnist | total | optimal | 1.0000 | 1.0000 | 0.0001 |
| kmnist | W0 | identity | 1.0000 | 1.0000 | 0.0001 |
| kmnist | W1 | identity | 1.0000 | 1.0000 | 0.0001 |
| kmnist | total | identity | 1.0000 | 1.0000 | 0.0001 |
| kmnist | W0 | optimal | 1.0000 | 1.0000 | 0.0001 |
| kmnist | W1 | optimal | 6.0000 | 0.2000 | 0.7399 |
| kmnist | total | optimal | 1.0000 | 1.0000 | 0.0001 |

Die kanonische gemeinsame Zuordnung kann für die Gesamtgeometrie sinnvoll sein und zugleich die in W1 allein erkennbare Koordinatenkorrespondenz verändern. Das W1-Ergebnis ist deshalb **bedingt auf dieses Matching**. Hier wurde kein nachträgliches W1-Spezialmatching optimiert, um die Erfolgsquote zu erhöhen.

## Normen: ursprüngliche und ergänzende Kontrolle

Die geplante Normkontrolle verwendet |Norm(Query E20) − Norm(Kandidat E0)|, getrennt nach W0/W1/Gesamt und roh/kanonisch.

| dataset | layer | gauge | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | W0 | raw | 5.5000 | 0.1000 | 1.0000 |
| mnist | W1 | raw | 5.5000 | 0.1000 | 1.0000 |
| mnist | total | raw | 5.5000 | 0.1000 | 1.0000 |
| mnist | W0 | canonical | 5.5000 | 0.1000 | 1.0000 |
| mnist | W1 | canonical | 5.5000 | 0.1000 | 1.0000 |
| mnist | total | canonical | 5.5000 | 0.1000 | 1.0000 |
| fashion_mnist | W0 | raw | 5.5000 | 0.1000 | 1.0000 |
| fashion_mnist | W1 | raw | 5.5000 | 0.1000 | 1.0000 |
| fashion_mnist | total | raw | 5.5000 | 0.1000 | 1.0000 |
| fashion_mnist | W0 | canonical | 5.5000 | 0.1000 | 1.0000 |
| fashion_mnist | W1 | canonical | 5.5000 | 0.1000 | 1.0000 |
| fashion_mnist | total | canonical | 5.5000 | 0.1000 | 1.0000 |
| kmnist | W0 | raw | 5.5000 | 0.1000 | 1.0000 |
| kmnist | W1 | raw | 5.5000 | 0.1000 | 1.0000 |
| kmnist | total | raw | 5.5000 | 0.1000 | 1.0000 |
| kmnist | W0 | canonical | 5.5000 | 0.1000 | 1.0000 |
| kmnist | W1 | canonical | 5.5000 | 0.1000 | 1.0000 |
| kmnist | total | canonical | 5.5000 | 0.1000 | 1.0000 |

**Grenze dieser Regel:** Wenn alle Endnormen über allen Anfangsnormen liegen, erhält jeder Query dieselbe Kandidatenrangfolge. Bei einer bijektiven Herkunftszuordnung ist der mittlere Rang dann konstruktiv 5,5. Dieses Ergebnis allein kann keine allgemeine Abwesenheit von Norminformation belegen.

Nach Erkennen dieser Degeneration wurde deshalb eine zusätzliche, explizit **nachträgliche explorative Normrang-Kontrolle** gerechnet: Innerhalb jedes Checkpoints und Datensatzes werden die zehn Normen rangiert; der Kostenwert ist |Normrang(Query Et) − Normrang(Kandidat E0)|/9. Das entfernt gemeinsame Änderungen des Normniveaus und prüft die Erhaltung der Normrangfolge. Die Kalibrierung verwendet keine Herkunftslabels, ist aber transduktiv über das gesamte Ensemble. Beim Jackknife werden die Normränge unter den verbleibenden neun Netzen **neu kalibriert**; es werden nicht einfach Zeilen/Spalten aus den Zehn-Netz-Kosten entfernt.

| dataset | layer | gauge | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | W0 | raw | 5.1500 | 0.1000 | 0.3468 |
| mnist | W1 | raw | 3.8000 | 0.4000 | 0.0340 |
| mnist | total | raw | 5.1000 | 0.1000 | 0.3313 |
| mnist | W0 | canonical | 4.7000 | 0.1000 | 0.1871 |
| mnist | W1 | canonical | 4.7000 | 0.1000 | 0.1871 |
| mnist | total | canonical | 4.7000 | 0.1000 | 0.1871 |
| fashion_mnist | W0 | raw | 5.7500 | 0.1000 | 0.6123 |
| fashion_mnist | W1 | raw | 5.6000 | 0.1000 | 0.5380 |
| fashion_mnist | total | raw | 5.3000 | 0.2000 | 0.4100 |
| fashion_mnist | W0 | canonical | 6.2000 | 0.1000 | 0.7870 |
| fashion_mnist | W1 | canonical | 6.2000 | 0.1000 | 0.7870 |
| fashion_mnist | total | canonical | 6.2000 | 0.1000 | 0.7870 |
| kmnist | W0 | raw | 6.6000 | 0.1000 | 0.8952 |
| kmnist | W1 | raw | 7.1000 | 0.0000 | 0.9762 |
| kmnist | total | raw | 4.3500 | 0.3000 | 0.1034 |
| kmnist | W0 | canonical | 6.6500 | 0.1000 | 0.9097 |
| kmnist | W1 | canonical | 6.6500 | 0.1000 | 0.9097 |
| kmnist | total | canonical | 6.6500 | 0.1000 | 0.9097 |

Gesamtnorm-Rangkontrollen: MNIST (raw): Rang 5.10, p=0.3313; MNIST (canonical): Rang 4.70, p=0.1871; Fashion-MNIST (raw): Rang 5.30, p=0.4100; Fashion-MNIST (canonical): Rang 6.20, p=0.7870; KMNIST (raw): Rang 4.35, p=0.1034; KMNIST (canonical): Rang 6.65, p=0.9097. Auch diese Tests sind explorativ und keine Auswahl eines neuen primären Endpunkts. Quellen, Motivation und eigene Seeds stehen in `norm_rank_controls/analysis_note.json`.

Gleichgewichtete Zusammenfassung der Normrang-Kontrollen:

| gauge | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- |
| raw | 4.9167 | 0.2000 | 0.1289 |
| canonical | 5.8500 | 0.1000 | 0.7481 |

Die Normrang-Kontrolle reproduziert die nahezu perfekte Wiedererkennung der vollständigen Gewichte nicht. Einzelne unadjustierte Layer-Befunde sind bei dieser Anzahl explorativer Tests vorsichtig zu lesen; aus ihnen folgt kein konsistenter Normeffekt über alle drei Datensätze. Die kanonischen W0-, W1- und Gesamtnormen sind durch die Normangleichung mathematisch proportional und keine drei unabhängigen Kontrollen.

## Kleine und große Residuen

Nach kanonischer Gauge, gemeinsamer Gesamtnormierung und derselben optimalen Paarzuordnung werden **für jedes Kandidatenpaar separat** die absoluten Gewichtsresiduen berechnet. Pro Schicht wird das untere bzw. obere Viertel ausgewählt, einschließlich aller Grenzwertbindungen. Kosten sind RMS auf den ausgewählten Einträgen; Gesamtwerte poolen beide Layer. Schwellen und Mengen sind für alle Kandidatenpaare gespeichert.

| dataset | metric | layer | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | small_residual_rms | W0 | 1.0000 | 1.0000 | 0.0001 |
| mnist | small_residual_rms | W1 | 4.8000 | 0.2000 | 0.2266 |
| mnist | small_residual_rms | total | 1.0000 | 1.0000 | 0.0001 |
| mnist | large_residual_rms | W0 | 1.0000 | 1.0000 | 0.0001 |
| mnist | large_residual_rms | W1 | 8.3000 | 0.0000 | 0.9999 |
| mnist | large_residual_rms | total | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | small_residual_rms | W0 | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | small_residual_rms | W1 | 5.0000 | 0.1000 | 0.3196 |
| fashion_mnist | small_residual_rms | total | 1.1000 | 0.9000 | 0.0001 |
| fashion_mnist | large_residual_rms | W0 | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | large_residual_rms | W1 | 9.3000 | 0.0000 | 1.0000 |
| fashion_mnist | large_residual_rms | total | 1.0000 | 1.0000 | 0.0001 |
| kmnist | small_residual_rms | W0 | 1.0000 | 1.0000 | 0.0001 |
| kmnist | small_residual_rms | W1 | 7.0000 | 0.0000 | 0.9523 |
| kmnist | small_residual_rms | total | 1.1000 | 0.9000 | 0.0001 |
| kmnist | large_residual_rms | W0 | 1.0000 | 1.0000 | 0.0001 |
| kmnist | large_residual_rms | W1 | 6.7000 | 0.0000 | 0.9434 |
| kmnist | large_residual_rms | total | 1.0000 | 1.0000 | 0.0001 |

Das ist eine Sensitivität gegenüber verschiedenen Teilen der Residuenverteilung. Diese Residuen sind **nicht die tatsächlichen historischen Trainingsänderungen**: Das Matching und die Normierung verändern den Vergleich, und bei fremden Kandidaten gibt es keine entsprechende gemeinsame Trainingsgeschichte. Die Auswertung ist auch keine Intervention, die schwach veränderte Parameter kausal aus dem Netz entfernt.

Die tatsächlichen Änderungen der gespeicherten Rohgewichte vom eigenen E0 zu E20 wurden separat deskriptiv ausgewertet; folgende Werte sind Mittel über zehn Runs je Datensatz:

| dataset | layer | n_runs | median_abs_change_mean | p75_abs_change_mean | relative_frobenius_change_mean | fraction_exactly_unchanged_mean | fraction_change_le_1e_minus6_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fashion_mnist | W0 | 10 | 0.0454 | 0.0851 | 4.2882 | 0.0000 | 1.99e-06 |
| fashion_mnist | W1 | 10 | 0.1722 | 0.3141 | 3.8961 | 0.0000 | 0.0000 |
| kmnist | W0 | 10 | 0.0680 | 0.1165 | 5.0533 | 0.0000 | 1.2e-05 |
| kmnist | W1 | 10 | 0.2557 | 0.4162 | 4.9261 | 0.0000 | 0.0000 |
| mnist | W0 | 10 | 0.0263 | 0.0697 | 3.8671 | 0.0000 | 1.59e-05 |
| mnist | W1 | 10 | 0.2353 | 0.3839 | 4.4948 | 0.0000 | 0.0000 |

Kein gespeicherter Gewichtseintrag blieb am Endpunkt exakt unverändert. Die mittlere Frobenius-Bewegung je Datensatz und Schicht beträgt das 3.87- bis 5.05-Fache der jeweiligen Anfangsnorm. Die Wiedererkennung beruht damit nicht auf einem Bestand exakt eingefrorener Parameter. Das allein quantifiziert noch nicht, wie viel Information kleine gegenüber großen Änderungen tragen.

Diese historische Tabelle wird weder für Kandidatenkosten noch für Masken oder inferenzielle p-Werte verwendet. Damit erfährt der Wiedererkennungstest aus ihr nicht, welcher Kandidat der richtige ist.

## Query-Aktivität und inaktive Neuronen

Für jedes Query-Netz und jeden Checkpoint wird auf dem bereits vorhandenen festen Probe mit 3.000 Bildern pro Hidden-Neuron der Anteil positiver ReLU-Aktivierungen bestimmt. Das untere bzw. obere Aktivitätsquartil liefert die Maske; alle Grenzwertbindungen werden aufgenommen. Ausschließlich Query-Daten bestimmen die Maske, nicht Kandidatenaktivierungen oder tatsächliche Herkunft. Ein neues maskenspezifisches gemeinsames Hungarian-Matching minimiert die vollständige quadratische Distanz einschließlich Kandidatennormtermen für alle Kandidaten gleich.

| dataset | metric | layer | mean_rank | top1 | p_perm_less |
| --- | --- | --- | --- | --- | --- |
| mnist | activity_low_normalized_frobenius | W0 | 1.0000 | 1.0000 | 0.0001 |
| mnist | activity_low_normalized_frobenius | W1 | 5.3000 | 0.2000 | 0.4191 |
| mnist | activity_low_normalized_frobenius | total | 1.0000 | 1.0000 | 0.0001 |
| mnist | activity_high_normalized_frobenius | W0 | 1.0000 | 1.0000 | 0.0001 |
| mnist | activity_high_normalized_frobenius | W1 | 5.1000 | 0.3000 | 0.3152 |
| mnist | activity_high_normalized_frobenius | total | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | activity_low_normalized_frobenius | W0 | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | activity_low_normalized_frobenius | W1 | 6.8000 | 0.2000 | 0.9478 |
| fashion_mnist | activity_low_normalized_frobenius | total | 1.1000 | 0.9000 | 0.0001 |
| fashion_mnist | activity_high_normalized_frobenius | W0 | 1.0000 | 1.0000 | 0.0001 |
| fashion_mnist | activity_high_normalized_frobenius | W1 | 6.4000 | 0.0000 | 0.9026 |
| fashion_mnist | activity_high_normalized_frobenius | total | 1.4000 | 0.9000 | 0.0001 |
| kmnist | activity_low_normalized_frobenius | W0 | 1.1000 | 0.9000 | 0.0001 |
| kmnist | activity_low_normalized_frobenius | W1 | 4.7000 | 0.3000 | 0.1735 |
| kmnist | activity_low_normalized_frobenius | total | 1.0000 | 1.0000 | 0.0001 |
| kmnist | activity_high_normalized_frobenius | W0 | 1.0000 | 1.0000 | 0.0001 |
| kmnist | activity_high_normalized_frobenius | W1 | 5.3000 | 0.3000 | 0.4195 |
| kmnist | activity_high_normalized_frobenius | total | 1.0000 | 1.0000 | 0.0001 |

Anzahl ausgewählter Hidden-Neuronen am Endpunkt:

| dataset | subset | min | mean | max |
| --- | --- | --- | --- | --- |
| fashion_mnist | high | 16 | 16.0000 | 16 |
| fashion_mnist | inactive | 0 | 0.0000 | 0 |
| fashion_mnist | low | 16 | 16.1000 | 17 |
| kmnist | high | 16 | 16.0000 | 16 |
| kmnist | inactive | 0 | 0.0000 | 0 |
| kmnist | low | 16 | 16.0000 | 16 |
| mnist | high | 16 | 16.0000 | 16 |
| mnist | inactive | 0 | 0.0000 | 0 |
| mnist | low | 16 | 16.1000 | 17 |

Am Endpunkt sind insgesamt 0 Neuronen über die 30 Netze auf diesem Probe vollständig inaktiv. »Inaktiv« bezeichnet ausschließlich Nullaktivität auf diesem konkreten Probe, keine globale Aussage für alle möglichen Eingaben. Ein Test mit einer leeren Inaktivitätsmaske wäre undefiniert; stattdessen werden die nichtleeren Aktivitätsquartile und die tatsächlichen Inaktivitätszahlen berichtet. Die Ergebnisse häufiger aktiver Neuronen sind in dieser Analyse kein Vergleich funktionaler Anfangs-/Endrepräsentationen und kein Beweis, dass der erkennbare Fingerabdruck für die Klassifikation benötigt wird.

## Fazit und Grenzen

Punkt1 zeigt die Wiedererkennbarkeit der eigenen Initialisierung; Punkt2 bestätigt ihre Robustheit für mehrere Gesamtgewichts-Darstellungen und lokalisiert deutliche Unterschiede zwischen Schichten und Zuordnungen. Die schwache Ensemble-RSA aus01 bleibt davon unberührt. Beide Ergebnisse beantworten unterschiedliche Fragen.

Initialisierung und Batch-Reihenfolge sind weiterhin durch denselben Run-Seed gekoppelt. Diese Auswertung trennt ihre kausalen Beiträge nicht. Für eine solche Trennung wäre das vorgeschlagene gekreuzte Training erforderlich; es wurde nicht beauftragt und nicht gestartet. Keine Aussage über eine vollständige Trajektorienrekonstruktion oder einen dynamischen Attraktor folgt aus diesen Ergebnissen.

## Reproduktion und Dateien

- [Figurenübersicht](index.html), [alle Endpunktkontrollen inklusive Normrang-Zusatz](all_control_endpoints.csv).
- `retrieval_by_dataset.csv` enthält die 42 geplanten Varianten; `norm_rank_controls/retrieval_by_dataset.csv` die sechs nachträglichen Normrang-Varianten. Beide enthalten sämtliche Zeitpunkte und ungeglättete Ergebnisse.
- `data/distance_definitions.json`: vollständige Kosten-, Matching-, Normierungs- und Maskendefinitionen.
- `data/*_retrieval.npz`: alle Kosten, neuronalen Zuordnungen, Normen, Residualschwellen und -mengen; `data/*_query_masks.npz`: jede Query-Maske und Aktivitätsrate.
- `data/*_inference.npz`: sämtliche Nullstatistiken, bijektive Herkunftsmappings und Jackknife-Replikate. Entsprechende Dateien für Normränge unter `norm_rank_controls/data/`.
- Prüfsummenbelege unter `provenance/`; das Punkt1-Referenzprotokoll stimmt in allen Matrizen mit maximalem Fehler 0 überein.

Wiederholung: `.venv/bin/python 03_Initialization_Controls/code/run.py`. Alle Ergebnisse liegen ausschließlich hier. Original-Reproduktion, Experiment01 und der vor Beginn dieser Kontrolle eingefrorene Punkt1 bleiben unverändert.
