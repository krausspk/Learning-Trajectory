# Attraktor-Schlussfolgerung

**Gesamtbefund:** Keine allgemeine Rückkehr zur individuellen Kontrolltrajektorie. In der Erweiterung zeigen 0/216 Zellen über alle Schichtvarianten ein deskriptives 95%-Intervall für r vollständig unter 1; 216/216 liegen vollständig darüber. Die Ergebnisse gelten für die untersuchten Bedingungen. Ein strenger mathematischer Attraktor oder eine universelle Basin-Grenze ist damit nicht nachgewiesen.

## MNIST-Pilot

**Funktionale Eignung am Endpunkt, ohne gesicherten strukturellen Rückkehrnachweis.**

In allen untersuchten Störstärke-/Momentum-Zellen liegt selbst die untere 95%-Intervallgrenze von r über 1. Nach zehn weiteren Epochen sind die Forks von ihrer individuellen Kontrolltrajektorie weiter entfernt als unmittelbar nach der Störung. H1 wird für diesen Horizont nicht unterstützt. Deskriptiv zeigen 300/300 Störläufe während der gespeicherten Zwischenstände zumindest vorübergehend r<1 (kleinster gemessener Wert 0.9456). Vorübergehende Kontraktion und fehlende Kontraktion am Zehn-Epochen-Endpunkt sind deshalb getrennte Befunde.

180/300 unmittelbar nach der Störung außerhalb des festen Zielbereichs liegende Forks erreichen ihn am Ende. Insgesamt gehören 180/300 gestörte Endpunkte zu dieser Region. In 0 Fällen lag die ungestörte Ausgangskonfiguration bereits im Zielbereich, die Störung führte hinaus und die Fortsetzung wieder hinein. Nur diese engere Kategorie weist einen beobachteten störungsbedingten Austritt mit anschließender Rückkehr aus. Hier lag jedoch keine ungestörte Ausgangskonfiguration bereits im festen kombinierten Zielbereich; das strengere störungsbedingte Rückkehrkriterium ist damit nicht prüfbar. Alle 300 gestörten Ausgangszustände lagen bereits innerhalb der festen strukturellen Grenze. Die Intervention erzeugte keinen sofortigen strukturellen Austritt; eine Rückkehr nach einem solchen Austritt wurde in diesem Raster daher nicht getestet. Das ist kein Gegenbeweis gegen H2. Eintritt in die kombinierte Region betrifft hier Funktionskriterien oder bereits bestehende strukturelle Nähe und belegt keine strukturelle Attraktion.

Die vorab festgelegte Regel für individuelle Nähe verwendet 10 % der typischen Distanz zeitlich passender unabhängiger Referenznetze plus Funktionskriterien. Diese zeitabhängige Umgebung wird in 0 Fällen nach anfänglichem Verlassen wieder erreicht. Bloße Endpunktnähe ist keine lokale Rückkehr; für eine Rückstelltendenz ist zusätzlich Kontraktion gegenüber der anfänglichen Stördistanz relevant. Eine Basin-Grenze ist mit drei Amplituden und zehn Ausgangsnetzen nicht lokalisiert.

Bei der größten getesteten Störung ε=0.20 verändert sich die mittlere Accuracy unmittelbar von 97.218 % auf 96.607 %, der Loss von 0.12048 auf 0.14728. Nach zehn weiteren Epochen liegen die gestörten Forks bei 97.612 % und Loss 0.13056; die zugehörigen ungestörten Fortsetzungen bei 97.567 % und Loss 0.13281. Dies beschreibt beobachtete Funktionsänderungen und ihre Erholung; kleine Störungen können bereits unmittelbar danach kaum messbaren Leistungsverlust zeigen. Die strenge Referenzschwelle für Funktionsmitgliedschaft und die direkte Erholung relativ zur eigenen Baseline sind verschiedene Kriterien.

| Momentum | ε | r am Ende, 95%-KI | Task-Region, 95%-KI | Fester Zielbereich: außen → innen | Störungsbedingte Rückkehr | Individuelle Nähe, 95%-KI |
|---|---:|---:|---:|---:|---:|---:|
| A · Momentum fortgesetzt | 0.01 | 16.134 [14.964; 17.294] | 72.0 % [48.0 %; 92.0 %] | 72.0 % [48.0 %; 92.0 %] | 0.0 % [0.0 %; 0.0 %] | 0.0 % [0.0 %; 0.0 %] |
| A · Momentum fortgesetzt | 0.05 | 3.344 [3.121; 3.550] | 62.0 % [38.0 %; 84.0 %] | 62.0 % [38.0 %; 84.0 %] | 0.0 % [0.0 %; 0.0 %] | 0.0 % [0.0 %; 0.0 %] |
| A · Momentum fortgesetzt | 0.20 | 1.165 [1.128; 1.202] | 34.0 % [12.0 %; 58.0 %] | 34.0 % [12.0 %; 58.0 %] | 0.0 % [0.0 %; 0.0 %] | 0.0 % [0.0 %; 0.0 %] |
| B · Momentum zurückgesetzt | 0.01 | 15.646 [14.196; 16.916] | 74.0 % [52.0 %; 92.0 %] | 74.0 % [52.0 %; 92.0 %] | 0.0 % [0.0 %; 0.0 %] | 6.0 % [0.0 %; 18.0 %] |
| B · Momentum zurückgesetzt | 0.05 | 3.305 [3.056; 3.527] | 76.0 % [52.0 %; 96.0 %] | 76.0 % [52.0 %; 96.0 %] | 0.0 % [0.0 %; 0.0 %] | 0.0 % [0.0 %; 0.0 %] |
| B · Momentum zurückgesetzt | 0.20 | 1.161 [1.120; 1.196] | 42.0 % [22.0 %; 64.0 %] | 42.0 % [22.0 %; 64.0 %] | 0.0 % [0.0 %; 0.0 %] | 0.0 % [0.0 %; 0.0 %] |

Die festen Zielbereichsanalysen, der störungsbedingte Austritt mit Rückkehr und die Unterscheidung zwischen anfänglicher und späterer lokaler Nähe wurden als **sekundäre Ergänzung** am 2026-11-16T12:38:46.173839+01:00 aufgenommen: während des Pilottrainings, vor jeglicher Pilot-Endpunkt- oder Trajektorienauswertung. Sie waren nicht Teil der vor Trainingsstart festgeschriebenen primären Analyse. Die primären Schwellen, Läufe und Trainingsprotokolle blieben unverändert. [Zeitpunkt und Begründung](analysis_amendment.json).

Die Intervalle resampeln Ausgangsnetze nach Mittelung ihrer Störrichtungen. Eine niedrige Loss-Barriere oder gute Accuracy allein gilt nicht als dynamische Rückkehr. Keine dieser Auswertungen beweist einen mathematischen Attraktor oder eine universelle Basin-Struktur.

Untersucht wurden 300 gestörte MNIST-Forks aus 10 unabhängigen Ausgangsnetzen bei E20 mit zehn zusätzlichen Epochen. Die vollständig ausgeführte Erweiterung über drei Datensätze, drei Störzeitpunkte, drei Schichtvarianten und vier nichttriviale Amplituden wird im [erweiterten Ergebnisbericht](expanded_analysis.md) separat ausgewertet. Die Zahlen dieses Pilotabschnitts beziehen sich ausschließlich auf die fünf Richtungen des MNIST-Piloten.

[Methodik, Kontrollen und vollständige Einordnung](basin_analysis.md) · [Endpunktmesswerte](endpoint_metrics.csv) · [Konfidenzintervalle](endpoint_summary.csv)


## Erweiterung auf alle Datensätze und Zeitpunkte

Zusätzlich zum Pilot wurden **2.160 Störläufe plus 180 Baselines** über drei Datensätze, drei Checkpoints, drei Schichtvarianten, vier Störstärken und beide Momentum-Protokolle vollständig ausgeführt. Jede Zelle enthält zehn Ausgangsnetze mit einer Richtung je Netz; die fünf Richtungen des Pilots bleiben separat ausgewertet. 0 numerische Ausfälle.

Bei gemeinsamer Störung beider Matrizen liegen 0/72 deskriptive r-Intervalle vollständig unter 1, 72/72 vollständig darüber. Damit ist die Aussage bedingungsspezifisch; eine allgemeine Rückkehr zur individuellen Kontrolltrajektorie oder eine universelle Basin-Grenze folgt daraus nicht. [Erweiterter Ergebnisbericht mit Abbildungen](expanded_analysis.md).

Alle 2160/2160 gestörten Ausgangszustände waren bereits innerhalb der festen strukturellen Region. Eine strukturelle Rückkehr nach unmittelbarem Austritt ist damit in dieser Erweiterung nicht getestet. Die 115 Rückkehren in die kombinierte Region betreffen funktionale Kriterien. [Matching-Robustheit](alignment_robustness.md).
