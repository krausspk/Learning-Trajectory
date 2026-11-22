# Reproduktionsspezifikation

Stand: 16. November 2026. Quelle: Krauss et al. (2026), *Task-dependent clustering in permutation-aligned neural network weights*, DOI 10.3389/fcogn.2026.1960468. Die lokale Hauptdatei ist `fcogn-5-1960468.pdf`. Das zusätzlich beschaffte Supplement liegt in `references/supplement.pdf` (19 Seiten). Die beiden extrahierten Volltexte liegen neben den Quellen. `configs/paper_targets.json` enthält 170 berichtete numerische Vergleichswerte mit Quellenstellen.

Die Nutzerpräzisierung erlaubt neue numerische Werte. Ziel ist eine möglichst genaue methodische Reproduktion **aller tatsächlich durchgeführten Experimente und Auswertungen**, ohne Ergebnisse aus dem Paper in Messdaten einzusetzen oder Seeds auf deren Übereinstimmung zu optimieren. Die berichteten Konfidenzintervalle sind keine Toleranzgrenzen für neue Läufe. Abweichungen werden einschließlich ihrer Größenordnung dokumentiert. Ein neues Experiment gilt auch dann als durchgeführt, wenn ein behaupteter Effekt nicht erneut auftritt; seine Interpretation muss das tatsächliche Ergebnis wiedergeben.

## Quellen und Verfügbarkeit

- Hauptartikel: <https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2026.1960468/full>.
- Verlagsmetadaten des Supplements: <https://www.frontiersin.org/api/v4/articles/1960468/supplemental-data>. Gesicherte Antwort: `references/supplemental_metadata.json`.
- Verlagsdatei: <https://public-pages-files-2025.frontiersin.org/articles/1960468/file/Data_Sheet_1.pdf/1960468_data-sheet_1/2>. Angezeigter Name: `Data Sheet 1.pdf`, Version 2; PDF-Erstellungsdatum 23. September 2026.
- Artikel und Supplement nennen kein öffentliches Repository. Die Datenerklärung sagt lediglich eine Bereitstellung der Rohdaten durch die Autoren zu. Der Download des Supplements enthält Tabellen und Abbildungen, keine Skripte oder Checkpoints. Diese Feststellung bezieht sich auf die geprüften Quellen und ist kein Nachweis, dass nirgendwo Originalcode existiert.
- SHA-256-Fingerabdrücke beider PDF-Dateien sind im Target-Register gespeichert. Die Artikel-HTML-Seite wurde zur Quellenprüfung gesichert.

## Verbindlicher Versuchsaufbau

| Eigenschaft | Beschreibung aus dem Paper |
|---|---|
| Datensätze | MNIST, Fashion-MNIST, KMNIST; jeweils originale 60.000 Trainings- und 10.000 Testbilder |
| Eingaben | 28×28 Graustufen, flach als 784 Merkmale; float32, `/255`, danach `(x-mean)/sd` |
| Normalisierung | MNIST (0,1307; 0,3081), Fashion-MNIST (0,2860; 0,3530), KMNIST (0,1918; 0,3483) |
| Augmentation | Keine |
| Netzwerk | 784→64→10; eine ReLU-Hidden-Schicht; Logits ohne Output-Aktivierung |
| Initialisierung | PyTorch-Standardinitialisierung beider Linear-Schichten; Biases auf null setzen, von Optimierung ausschließen, nach Updates kontrollieren |
| Optimierung | Cross-Entropy; SGD, LR 0,1, Momentum 0,9, Batchgröße 256; übrige Optionen nicht abweichend spezifiziert |
| Training | 10 Epochen; pro Epoche deterministisch aus Run-Seed abgeleitete neue Bildreihenfolge; Testdaten nicht mischen |
| Checkpoints | E0 Initialisierung bis E20 Ende Epoche 10; halbe Epochen, 21 Checkpoints |
| Hauptkohorte | 10 Läufe je Dataset: MNIST Seeds101–110, Fashion Seeds201–210, KMNIST Seeds301–310 |
| Paarzahlen | Pro Checkpoint 435 ungeordnete Paare: 135 innerhalb Dataset, 300 zwischen Datasets |
| Genauigkeit | Test-Accuracy und mittlere Test-Cross-Entropy bei jedem Checkpoint |

Die Hauptanalyse ist kanonische, schichtbalancierte Alpha-Dissimilarität mit Alpha-Exponent 2, nach direkter Zuordnung anhand vollständiger kanonischer Neuronenprofile. Alle 21 Checkpoints und alle 435 Paare werden ausgewertet. Ein gemeinsames Hidden-Permutationsmapping muss gleichzeitig Spalten von W0 und Zeilen von W1 verändern.

### Aufgaben-Kontrolle

20 zusätzliche Netzwerke in 10 gepaarten Seed-Blöcken 401–410. Alle erhalten MNIST, dieselbe Normalisierung und Trainingskonfiguration. Ein Netz pro Block lernt Ziffernidentität mit 10 Logits, das andere Parität `digit % 2` mit 2 Logits. Initiale W0 und Bildreihenfolge müssen innerhalb des Blocks bitweise identisch sein. Die Output-Layer sind unterschiedlich groß.

Gewichtsvergleich nur für W0, mit incoming-weight-only Hungarian-Assignment pro Metrik; keine Zweischicht-Kanonisierung und kein Whole-Network-Git-Re-Basin quer über Aufgaben. CKA ebenfalls berechnen. Checkpoints für die strukturelle Auswertung: E∈{0,1,2,5,10,20}; Trainingsevaluierung weiterhin alle 21. Probe: erste 200 Testbilder jeder Ziffer in ursprünglicher IDX-Reihenfolge, insgesamt 2.000. Innerhalb-Aufgabe besteht aus 90 Paaren (45 Identität und 45 Parität), zwischen Aufgaben aus 100 Paaren (10 Same-Seed und 90 Different-Seed). Zusätzlich diese vier Untergruppen separat beschreiben. Resampling-Einheit ist immer der ganze Seed-Block.

### Frühe SGD-Kohorte

15 zusätzliche Netzwerke, fünf je Dataset, gleiche Architektur und Optimierung: Seeds701–705,801–805,901–905. Zwei komplette Epochen (235 Mini-Batches/Epoche), gespeicherte Updatezahlen:

`0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 235, 352, 470`.

Bei allen 16 Checkpoints alle 105 Netzwerkpaare auswerten: Euclidean, SND, raw Alpha, canonical Alpha und Hidden-1−CKA. Weight-Views: W0, W1, balanced und size-weighted. Der gemeinsame 3.000-Bilder-Probe wird wiederverwendet. Zusätzlich jede Schicht relativ zur Initialisierung untersuchen:

- Relative Frobenius-Verschiebung `||W(s)-W(0)||F / ||W(0)||F`.
- RMS-Änderung pro Parameter `||W(s)-W(0)||F / sqrt(number_of_weights)`.
- Mittelwerte und 95%-Student-t-Intervalle über alle 15 Läufe.
- Funktioneller Fortschritt je Dataset `(mean_accuracy(s)-mean_accuracy(0))/(mean_accuracy(470)-mean_accuracy(0))`.
- Zeitpunkt 90%: erstes gespeichertes Update mit mittlerem Fortschritt über die drei Datasets mindestens 0,9. Zusätzlich je Dataset bestimmen.
- Struktureller Beginn: erster gespeicherter Zeitpunkt >0, dessen vollständiges 95%-Run-Jackknife-Intervall des Between-minus-within-Kontrasts positiv ist. Es wird kein nichtgespeicherter Update interpoliert.

Die ursprünglichen Zeitpunkte 2/4/64 sind Ergebnisse, keine Vorschrift. Die Hypothese einer zeitlich nachgelagerten Konsolidierung nur dann stützen, wenn ihre im Paper angegebenen zeitlichen Bedingungen in den neuen Daten erfüllt sind. Die Nähe zur Initialisierung gilt im Paper nur für die allerersten Updates, nicht zwangsläufig für die ganze dargestellte Kurve.

## Gewichte, Ähnlichkeit und Zuordnung

W0 hat Form784×64, W1 Form64×K. Pro Hidden-Neuron sind u die eingehenden und v die ausgehenden Gewichte, q=(u,v). Für einen Assignment-Index p wird B als `W0_B[:,p], W1_B[p,:]` verglichen.

- **Euclidean:** RMS der Elementdifferenzen, nicht unnormalisierte L2-Norm.
- **SND:** Bei a=b=0 Beitrag0; bei gleichem, von null verschiedenem Vorzeichen `abs((a-b)/(a+b))`; sonst1. Nennerbetrag≤1e−12 erhält1. Mittel über Elemente. Die Betragsstriche sind in reiner Textkonvertierung leicht zu übersehen.
- **Alpha:** Gewicht `abs(a*b)**2`, normalisiert über die verglichenen Array-Elemente; aufsummiertes Gewicht der Vorzeichenabweichungen. Nenner≤1e−12 liefert0. Keine mathematische Metrik; keine Aussage über Betragsähnlichkeit bei identischen Vorzeichen.
- **Aggregation:** Balanced=(d0+d1)/2; size-weighted=(50176*d0+640*d1)/50816. Im Task-Control nur W0.
- **PIF:** Pro u und v Summe, Zahl von Nichtnullwerten, Zahl positiver Werte und L2-Norm, insgesamt8 Features. Jede Dimension über die128 Neuronen beider Netze standardisieren. Euclidean-PIF-Cost und Hungarian ergeben initiale Reihenfolge. In dichten Netzen konstante Nichtnull-Counts dürfen keine Division durch null verursachen.
- **Metric-specific refinement:** Hungarian auf dissimilarity(qA,qB) mit voller eingehender und ausgehender Information. Jede Metrik hat ihre eigene Zuordnung. Lokal normierte Alpha-Costs minimieren nicht zwingend Matrix-Alpha.
- **Kanonische Gauge:** Pro nichtdegeneriertem Neuron a=||u||2, b=||v||2, s=sqrt(b/a); u*=s, v/=s. Beide Normen sind danach sqrt(a*b). Diese Gauge entfernt positive neuronweise ReLU-Skalierung. Anschließend direkte canonical-Alpha-Profilzuordnung; PIF-Vorsortieren ist bei einer einzelnen versteckten Schicht für den optimalen Assignment-Wert entbehrlich.
- **Benchmark-PIF:** Aus PIF-Assignment und direktem canonical-Alpha-Profile-Assignment den Kandidaten mit kleinerer finaler parameter-count-weighted canonical-Alpha-Dissimilarität auswählen. Als Benchmark-Ergebnis wird unter anderem layer-balanced canonical Alpha gezeigt. Diese beiden Aggregationen sind verschieden; Fig3 ist nicht dieselbe Definition wie Fig1/2.
- **Git Re-Basin:** Hungarian maximiert Summe des Similarity-Matrix `W0_A.T@W0_B + W1_A@W1_B.T`. Ein Hidden-Permutationsblock bedeutet globales Optimum dieses Objectives, keine iterative oder zufällige Optimierung nötig.

## Probe, CKA und unabhängige Aktivierungsvalidierung

Haupt-Probe: NumPy-Seed5.052.026, ohne Zurücklegen100 Testbilder je Klasse je Dataset auswählen, Dataset-Reihenfolge und Indizes speichern. Insgesamt3.000 Bilder; jedes nur einmal mit den Konstanten seines Ursprungs-Datasets normalisieren. Alle Netze erhalten exakt dieselbe numerische Matrix. Hidden-Aktivierung `ReLU(X@W0)` ohne Output-Schicht.

Lineares CKA: H-Spalten über Bilder zentrieren. `||HA.T@HB||F²/(||HA.T@HA||F*||HB.T@HB||F)`; berichtete Dissimilarität1−CKA; Roundoff auf[0,1] begrenzen. Haupt-CKA bei E∈{0,1,2,5,10,20}. Kein Matching für CKA.

Pearson und Spearman zwischen Hidden-1−CKA und raw-Alpha jeweils für W0,W1,balanced, an den sechs Checkpoints und getrennt über alle/within/between Paare. Für W0-Alpha bleibt die Zuordnung aus beiden Schichten; das ist nicht der Task-Control-Algorithmus.

Aktivierungskorrespondenz: 64 Pearson-Korrelationen entsprechender Hidden-Neuronen über Probe-Bilder mitteln. Bei allen21 Haupt-Checkpoints unaligned, Benchmark-PIF, Git Re-Basin vergleichen. Aktivierungen dürfen keine Assignment-Costs beeinflussen. Tote/konstante Neuronen müssen eine dokumentierte numerische Behandlung erhalten.

## Statistik

Haupt-/Early-Dataset-Kontrast: gepoolte Mean-Between minus gepoolte Mean-Within. Netzwerkpaare sind abhängig. Für jede Dataset-Stratum g jeden vollständigen Run r löschen und sämtliche inzidenten Paare entfernen; Statistik erneut berechnen. Varianz = Summe_g ((n_g−1)/n_g)*Summe_r(theta_g,-r−mean_r(theta_g,-r))². Normale95%-Intervalle estimate±1,95996*SE.

Task: pro Replikat beide Netze eines Seed-Blocks löschen; Varianz=(B−1)/B*Summe_b(theta_-b−mean(theta_-b))², B10. Korrelationen ebenfalls run-/blockweise resamplen, nicht die Paarwerte bootstrappen. Jackknife-Replikate und ausgeschlossene IDs speichern.

Run-Beiträge für S3: jedes Netz seine mittlere Distanz zu anderen Datasets minus seine mittlere Distanz zu anderen Runs des eigenen Datasets. Diese Werte sind deskriptiv, kein zweites Testverfahren. Sign-Retention und finale Accuracy/Loss-Verteilungen ebenfalls deskriptiv.

## Weitere vollständig zu wiederholende Analysen

### ReLU-Stresstest

Alle30 Hauptnetze bei E20; C∈{1,1.25,1.5,2,4,8}. Exponenten sind64 gleichmäßig verteilte Werte von−1 bis1, separat pro Netz deterministisch permutiert; Faktoren c=C**z haben geometrischen Mittelwert1. W0-Spalten mal c, korrespondierende W1-Zeilen durch c. Für jede Transformation erneutes raw-Alpha-Matching und Messung aller435 Paare und vier Views. Je C: mittlere absolute Änderung und Korrelation gegenüber C1, Between-minus-within-Kontrast mit Run-Jackknife, Anteil des Originalkontrasts. Funktionskontrolle auf256 Probe-Bildern: maximale Logitabweichung und Klassenübereinstimmung; Elementvergleich der kanonisierten Gewichte. Zusätzlich canonical Alpha für ganze Haupt-Trajektorie berechnen. Float64 für symmetriebezogene numerische Prüfungen verhindert das Verwechseln von float32-Roundoff mit strukturellen Effekten.

### Matching-Stabilität und funktionelle Validierung

Alle435 Paare×21 Checkpoints: raw total weight inner product, raw cosine, raw whole-network RMSE, canonical Alpha pro Schicht/balanced und Aktivierungskorrespondenz für identity, Benchmark-PIF und Git. Assignment-Overlap zwischen beiden Matchern mitteln. Auf30 vorab festgelegten E20-Paaren jeweils10 Zufallsumbenennungen von B: erneut matchen, Mapping in Originalindizes zurückkomponieren, Assignment- und Objective-Gleichheit sowie erhaltene Logits prüfen. S9 zeigt finale Aktivierungskorrelationsverteilungen und Anteile verbesserter Paare (PIF>unmatched, Git>unmatched, Git>PIF), über alle/within/between getrennt.

### Sign-Retention

Pro Hauptlauf, Layer und allen21 Checkpoints Anteil der Gewichte mit gleichem Vorzeichen wie beim selben Netz E0. Keine neuronweise Neuzuordnung zwischen E0 und Et. Dataset-Mittelkurven und finale Run-Verteilungen von1−retention; E1/E20-Tabelle. Dies liefert keine kausale Lottery-Ticket-Evidenz.

### Synthetischer Multilayer-Test

Architektur20→12→10→8→6. Ein Zufallsnetz kopieren, drei Hidden-Layer unabhängig permutieren und jeder Weight-Matrix iid Gaußrauschen mit SD=30% ihrer Gewichts-SD hinzufügen. Mit Euclidean,SND,Alpha-Profilkosten die Kopie iterativ ausrichten. Generalisierte Mehrschicht-Zigzag-Sweeps backward/forward; Abbruch bei unveränderten Reihenfolgen, Verbesserung des besten Network-Objectives<1e−12 oder maximal30 Passes. Vorher/nachher Objectives, Permutationsrecovery, Zahl Sweeps und Funktionsinvarianz durch reines Permutieren speichern. Das zugrunde liegende Netzwerk wird nicht trainiert; keine tiefen unabhängigen Trainingsexperimente daraus ableiten.

## Vollständige Abbildungs- und Tabellenabdeckung

| Kennung | Inhalt | Benötigte Daten |
|---|---|---|
| Fig1 | Design; finales canonical balanced Alpha within/between | Hauptkohorte, alle Paare, Jackknife |
| Fig2 | Canonical-Verlauf, layerweise Endkontraste, Euclidean/SND | Hauptkohorte21 Checkpoints |
| Fig3 | Alignment-Gewichtskosinus, canonical Alpha, Aktivierungskorrelation | Benchmark3 Methoden |
| Fig4 | Identity/Parity-Design; Aufgabenverlauf und Endkontrast | Task6 Checkpoints |
| Fig5 | Alpha/CKA-Kontrastverläufe, Gruppenmittel, Paar-Scatter | Main6 Checkpoints |
| Fig6 | Erste64Updates: Accuracy, Verschiebung, Strukturkontrast | Early-Kohorte |
| S1 / TableS1 | Run-Accuracy/Loss-Verläufe und Endverteilungen | Hauptkohorte |
| S2 / TableS2 | Drei Raw-Metriken×vierViews mit within/between-Verläufen | Hauptkohorte |
| S3 | Finale Run-Kontrastbeiträge je Metrik/View/Dataset | Hauptkohorte |
| S4 / TableS3 | Task-Accuracy, alle Metric-Kontraste, Seed-Untergruppen | Task-Kohorte |
| S5 / TableS4 | Alpha/CKA-Korrelationen, Schichten, Konditionierung | Hauptkohorte |
| S6 / TableS5 | ReLU-Skalierung,4Views, Funktion/Gauge-Fehler | Gauge-Kohorte |
| S7 / TableS6 | Alle470 frühen Updates als16Checkpoints, alle Metriken | Early-Kohorte |
| S8 / TableS7 | Matcher-Verläufe, Assignment-Overlap, Renumber-Tests | Benchmark |
| S9 | Aktivierungskorrelationsverteilungen und Verbesserungsanteile | Benchmark |
| S10 / TableS8 | Sign-Retention-Verläufe und Endverteilungen | Hauptkohorte |
| S11 / TableS9 | Synthetische tiefe Matching-Reduktion | Eigene deterministische Konstruktion |

Die exakten gezeichneten Koordinaten aller Kurven sind nicht tabellarisch veröffentlicht; sie werden neu berechnet. Kein Screenshot-Abzeichnen als Ersatz für Trainingsdaten.

## Offen gebliebene Implementierungsentscheidungen

Diese Punkte müssen in den erzeugten Run-Konfigurationen und im Ergebnisbericht explizit stehen. Sie sind keine Gründe, die berechenbaren Experimente anzuhalten.

1. Formel zur Ableitung der DataLoader-/Epochen-Seeds aus Run-Seed nicht angegeben; ebenso nicht klar, ob Generator pro Epoche reseeded oder kontinuierlich genutzt wurde.
2. Halbe Epoche bei235 Mini-Batches: genaue Grenzbehandlung (117/118 Updates bzw. exakt30.000 Beispiele) fehlt.
3. Standard-PyTorch-Initializer sagt nicht ausdrücklich, ob `bias=False` von Anfang an oder Bias-Initialisierung mit anschließendem Nullsetzen verwendet wurde. Dies verändert RNG-Verbrauch und somit nachfolgende Gewichte.
4. NumPy-Probe-Seed ist angegeben, Generatorfamilie (`RandomState` vs `default_rng`), Klassen-/Dataset-Schleifenreihenfolge und sortierte/unsortierte Auswahl nicht vollständig festgelegt.
5. Die Benchmark-Beschreibung benennt PIF- und canonical-Alpha-Kandidaten, aber nicht an jeder Stelle eindeutig die Gauge, in der die PIF-Featurestandardisierung erfolgt. Die gewählte Interpretation transparent dokumentieren.
6. PIF-Standardabweichung: ddof und Nullvarianzbehandlung nicht spezifiziert. Konstante Features sind aufgrund dichter Gewichte sicher vorhanden.
7. Genaue Seeds für ReLU-Exponentenpermutation sowie Auswahl der30Renumber-Paare und ihrer10Permutationen fehlen.
8. Konstante/tote Neuronen und degeneriertes CKA: Grenzfallkonvention nicht spezifiziert; robuste numerische Regel verwenden und ihre Häufigkeit melden.
9. Synthetisches Netz: Gewichtsverteilung, Seed, Permutations-/Noise-Seeds, Noise-Order und Whole-Network-Aggregation nicht angegeben. Diese Konstruktion daher methodisch und qualitativ, nicht bitweise reproduzierbar.
10. Originalumgebung: Windows64, Python3.9.5, NumPy1.23.5, SciPy1.10.1, PyTorch2.8.0+cpu, Matplotlib3.4.3. Neue Plattform und tatsächliche Versionen aufzeichnen; deterministische Läufe innerhalb dieser Umgebung testen. Hardware-/BLAS-Roundoff lässt sich nicht aus tabellierten Fehlergrenzen rekonstruieren.
11. Die Originalmanifeste, Analysepläne, Probe-Indizes, Checkpoints und Roh-Paartabellen werden im Paper beschrieben, sind im heruntergeladenen Supplement aber nicht enthalten.

## Artefakte und Abnahmekriterien

Vor Berechnungen Analyseplan und vollständige Konfiguration schreiben. Für jeden Run: Seed, Task/Dataset, Checkpoint-Updates, Gewichte, Testhistorie, Bias-Checks, Fingerabdruck und Versionsdaten. Probe-Dateien mit Herkunftsindizes und Fingerabdruck. Vollständige Pair-Tables, Summaries mit Jackknife-Replikaten, Gauge-/Matching-/Synthetic-Validation,16wissenschaftliche Abbildungen (6Hauptfiguren+10Supplementfiguren) und9Supplementtabellen. Alle Vergleichswerte aus neuer Berechnung markieren; `paper_targets.json` bleibt getrennt.

Numerische Prüfung: vollständige Formen/Checkpoints, finiteness, Symmetrie der Distanzmatrizen und Null-Diagonale; unabhängige Direktberechnung ausgewählter Metriken und CKA; Recovery bekannter Permutationen; erhaltene Logits; Kanonisierung nach neuronweiser Skalierung; identische W0/Trainingorder in gepaarten Tasks. Die Vollständigkeit der Experimentfamilien ist unabhängig von der Richtung eines gemessenen Effekts zu prüfen.

Die Outlook-Vorschläge (Perturbation/Rückkehr, Task-Switches, Basin-Mapping, tiefe unabhängig trainierte Netze, Modell-Merging und Loss-Interpolation) wurden im Paper **nicht** durchgeführt. Sie gehören zur späteren Aufbauphase, nicht zur Reproduktion vorhandener Ergebnisse.

## Nachträglicher Sensitivitätscheck: frühes Alignment

Dieser zusätzliche Check wurde **nach Sichtung einer auffälligen numerischen Abweichung** konzipiert. Er gehört nicht zum ursprünglichen Analyseplan und ersetzt keine Hauptanalyse. Der Haupttext nennt im frühen Experiment kanonische Alpha-Dissimilarität, unterscheidet dort jedoch nicht ausdrücklich zwischen direktem Profil-Assignment und der an anderer Stelle beschriebenen Benchmark-Kandidatenauswahl. Beide Verfahren bewerten dieselbe kanonische, schichtbalancierte Alpha-Größe, können aber andere Neuronen zuordnen.

`python -m learning_trajectory.sensitivity` hält sämtliche neu trainierten Gewichte unverändert und berechnet an allen 16 frühen Checkpoints alle 105 Paare unter drei Zuordnungen: direktes canonical-Alpha-Profil-Assignment, die dokumentierte Benchmark-Kandidatenauswahl und Git Re-Basin. Die Auswertung speichert 5.040 Paarwerte, alle Permutationen,192 Gruppenschätzungen (within/between/contrast/all),2.880 vollständige Run-Jackknife-Replikate und einen Diagnosebericht getrennt in `outputs/sensitivity/`. Die 1.680 direkten Paarwerte werden gegen die unveränderte Hauptanalyse erneut geprüft; maximale Abweichung 4,996×10⁻¹⁶.

Bei Update 4 ergeben sich für den Between-minus-within-Kontrast:

| Zuordnung | Kontrast | 95%-Run-Jackknife-Intervall |
|---|---:|---:|
| Direktes canonical-Alpha-Profil-Assignment | 0,016252 | [0,014447; 0,018057] |
| Benchmark-Kandidatenauswahl | 0,125960 | [0,094900; 0,157020] |
| Git Re-Basin | 0,020511 | [0,017221; 0,023800] |
| Berichteter Vergleichswert (Paper 3.9/Fig. 6) | 0,1322 | [0,1072; 0,1571] |

Der erste positive Kontrast nach dem im Paper definierten Zeitpunktkriterium liegt in den neuen Läufen bei Update 2 für direktes Assignment und Benchmark-Kandidatenauswahl sowie Update 4für Git Re-Basin. Der funktionelle 90%-Zeitpunkt der unveränderten frühen Kohorte liegt bei Update 64.

Die Größenordnung des frühen Gewichts-Kontrasts hängt hier stark von der Zuordnung ab. Die Benchmark-Variante liegt näher am publizierten Update 4-Wert; daraus folgt **weder ein nachgewiesener Fehler im Paper noch Gewissheit über die tatsächlich verwendete Originalimplementierung**. Keine Seeds, Gewichte, Lernraten oder Zuordnungsregeln wurden auf den berichteten Wert optimiert. Der Befund dient der transparenten Einordnung der Abweichung und zeigt, weshalb direkte und benchmarkbasierte Werte bei späteren Experimenten getrennt gehalten werden müssen.
