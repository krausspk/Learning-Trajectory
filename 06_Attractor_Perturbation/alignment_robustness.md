# Sekundäre Robustheit gegenüber der Alignment-Methode

Keine der drei Methoden ergibt eine Zelle mit einem vollständig unter 1 liegenden Verhältnisintervall. Git Re-Basin und PIF zeigen in jeweils 222 von 222 Zellen größere End- als Anfangsabstände; Alpha-Matching bestätigt dies in 221 Zellen.

Die zusätzlichen Matching-Methoden waren vorab vorgesehen. Diese gemeinsame Zusammenfassung wurde nach Vorliegen der Ergebnisse erstellt; sie verändert weder die primäre Analyse noch deren Kriterien.

Ausgewertet werden 300 gestörte Pilotläufe in sechs Zellen und 2.160 gestörte Erweiterungsläufe in 216 Zellen. Jede Zelle umfasst zehn unabhängige Ausgangsnetze. Im Pilot werden fünf Richtungen zuerst innerhalb eines Ausgangsnetzes gemittelt, in der Erweiterung liegt eine Richtung pro Ausgangsnetz vor. Baselines und Kontrollen mit anderer Batchreihenfolge sind hier ausgeschlossen.

Für jede Matching-Methode gilt r = Endabstand / Abstand unmittelbar nach der Störung. Die Abstände sind kanonische Frobeniusnormen über beide Gewichtsmatrizen. Die Tabellen zählen Zellen, deren punktweises 95%-Seed-Bootstrapintervall vollständig unter beziehungsweise über 1 liegt (10.000 Resamples). Fehlende Werte werden innerhalb eines Seeds nicht ausgelassen.

| Stufe | Matching | Zellen | KI vollständig < 1 | KI vollständig > 1 | KI umfasst 1 | Undefiniert |
|---|---|---:|---:|---:|---:|---:|
| Erweiterung | Git Re-Basin | 216 | 0 | 216 | 0 | 0 |
| Erweiterung | PIF/Hungarian | 216 | 0 | 216 | 0 | 0 |
| Erweiterung | Alpha-Matching | 216 | 0 | 215 | 1 | 0 |
| Pilot | Git Re-Basin | 6 | 0 | 6 | 0 | 0 |
| Pilot | PIF/Hungarian | 6 | 0 | 6 | 0 | 0 |
| Pilot | Alpha-Matching | 6 | 0 | 6 | 0 | 0 |

Das Alpha-Matching-Intervall für fashion_mnist, E20, Schicht both, ε=0.2, Momentum reset umfasst 1: Mittelwert 1.095, 95%-KI [0.959; 1.198].

Git Re-Basin minimiert bei dieser Architektur den euklidischen Abstand der kanonischen Gewichte exakt über die gemeinsame Neuronenpermutation. PIF/Hungarian optimiert die Zuordnung anhand komprimierter Neuronenmerkmale; es minimiert den vollständigen Gewichtsabstand nicht exakt. Auch das Alpha-Matching optimiert ein anderes Zuordnungskriterium. Unterschiedliche Anfangs- und Endzuordnungen können die relativen Abstände deutlich verändern. Alternative Kontraktionsbefunde sind deshalb keine unabhängigen Replikationen derselben Quotientenmetrik.

Zusätzlich wird die Änderung des Alpha-Vorzeichenmaßes ausgewertet: ΔAlpha = Alpha am Ende − Alpha unmittelbar nach der Störung. Ein Alpha-Verhältnis wird nicht verwendet. Alpha gewichtet Vorzeichenabweichungen mit den Gewichtsbeträgen; es ist keine euklidische Distanz. Negative Änderungen bedeuten weniger gewichtete Vorzeichenabweichung, nicht automatisch eine strukturelle Rückkehr.

| Stufe | ΔAlpha-KI vollständig < 0 | ΔAlpha-KI vollständig > 0 | KI umfasst 0 | Undefiniert |
|---|---:|---:|---:|---:|
| Pilot | 0 | 4 | 2 | 0 |
| Erweiterung | 21 | 184 | 11 | 0 |

Die Intervalle sind explorativ und nicht für multiple Vergleiche korrigiert. Ihre Grenzen sind keine simultanen Aussagen über alle Zellen. Eine Übereinstimmung mehrerer Matchings beweist weder eine Attraktorstruktur noch eine Basin-Grenze; ein richtungsabhängiger Befund bleibt insbesondere in der Erweiterung möglich.

[Einzelwerte](alignment_robustness.csv) · [Seed-Mittelwerte und Intervalle](alignment_robustness_summary.csv) · [Reproduktionsskript](code/alignment_robustness.py)
