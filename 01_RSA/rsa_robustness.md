# 01_RSA: Robustheit und genaue Auswertungsdefinitionen

## Stichprobe und Erhaltung der Originaldaten

30 unabhängige Hauptkohorten-Netze; 10 pro Datensatz; je 45 abhängige Within-Dataset-Paare; 135 Paare insgesamt ohne Between-Dataset-Pooling. 21 Checkpoints, also 630 Netzwerkzustände und 2.835 Paar-Checkpoint-Kombinationen. 52 fest definierte Varianten ergeben 147.420 Distanzwerte und 3276 RSA-Schätzer. Alle Netzwerkidentitäten stammen aus `outputs/manifests/main.json`, sortiert nach Seed innerhalb jedes Datensatzes. Null-Biases, Shapes, Checkpointfolge, Metadaten, NPZ-/PT-Hashes und 21 vorhandene Originalanalyse-Receipts wurden geprüft.

Die Reproduktion war vor Start abgeschlossen: alle 17 primären qualitativen Prüfungen bestanden; alle 65 Trainingsläufe waren in einer unabhängigen Wiederholung bitgleich. Diese Zusatzanalyse schreibt ausschließlich in `01_RSA`. Die alte Pipeline wurde nicht überschrieben. Es wurden keine neuen Trainingsläufe und keine optionale separate Frühkohorten-Replikation erzeugt.

## Kanonische Gauge und primäre Distanz

Pro Hidden Unit wird s = sqrt(||out||₂ / ||in||₂) verwendet, dann W0[:,h] *= s und W1[h,:] /= s. Das erhält die ReLU-Funktion. Keine der 40,320 geprüften Neuron-Checkpoint-Konfigurationen hatte eine Nullnorm. Deterministische Sonderfallkonvention: nur eingehend nichtnull -> s=1/||in||; nur ausgehend nichtnull -> s=||out||; beide null -> s=1. Für einseitige Nullnormen ist die Normangleichung mit endlicher positiver Skalierung nicht möglich; die aktive Seite wird stattdessen auf Eins normiert. Diese Konvention wurde getestet, aber in den vorliegenden Daten nicht benötigt.

Die gemeinsame Hungarian-Zuordnung maximiert A0ᵀB0 + A1B1ᵀ auf den kanonischen Gewichten. Für dieses einzelne Hidden Layer ist sie das globale Optimum der **gemeinsamen** quadratischen Frobenius-Distanz. Berichtet wird die Quadratwurzel. Auch layerweise Werte verwenden dieselbe gemeinsame Zuordnung, keine getrennten Layer-Optima. Die Gauge, Distanzformel und Matching-Zielfunktion wechseln über E0..E20 nicht.

## Schichten und Metrikrobustheit

| dataset | layer | rho | ci_low | ci_high | p_perm_greater |
| --- | --- | --- | --- | --- | --- |
| mnist | total | 0.1165 | -0.5678 | 0.8007 | 0.2935 |
| mnist | W0 | 0.0866 | -0.4397 | 0.6128 | 0.3641 |
| mnist | W1 | 0.0539 | -0.5564 | 0.6642 | 0.3572 |
| mnist | balanced | 0.0296 | -0.5808 | 0.6401 | 0.4137 |
| fashion_mnist | total | -0.1806 | -0.5728 | 0.2115 | 0.8401 |
| fashion_mnist | W0 | -0.4099 | -0.9141 | 0.0943 | 0.9575 |
| fashion_mnist | W1 | 0.4804 | 0.1161 | 0.8446 | 0.0007 |
| fashion_mnist | balanced | 0.4679 | 0.1192 | 0.8165 | 0.0009 |
| kmnist | total | 0.2764 | -0.1522 | 0.7051 | 0.0490 |
| kmnist | W0 | 0.2631 | -0.2715 | 0.7977 | 0.0682 |
| kmnist | W1 | -0.0016 | -0.5141 | 0.5110 | 0.5079 |
| kmnist | balanced | -0.0011 | -0.5202 | 0.5181 | 0.5049 |

Layer-balancierte Frobenius-Distanz = sqrt(0,5 × (||ΔW0||² / 50.176 + ||ΔW1||² / 640)). Das ist die quadratische Mittelung der beiden Layer-RMS-Werte; die Zuordnung optimiert weiterhin die gemeinsame ungewichtete Distanz. Cosine-Dissimilarität = 1 − <A,B>/(||A|| ||B||); normierte Frobenius-Distanz = sqrt(2 × Cosine-Dissimilarität). Letztere beiden sind bei derselben Zuordnung streng monoton äquivalent und liefern dieselbe Spearman-RSA: sie sind keine zwei unabhängigen Robustheitsbelege.

Alpha (Exponent 2) und SND verwenden exakt die zuvor reproduzierten metric-spezifischen Profilzuordnungen. Kanonische Alpha nutzt die direkte Profil-Hungarian-Lösung; rohe Alpha und SND die reproduzierte PIF-/Profilpipeline. Es werden keine PIF/Alpha-Benchmark-Kandidaten an deren Stelle eingesetzt. Alle Werte wurden aus den Gewichten neu ausgewertet und stimmen mit den vorhandenen Paartabellen bis maximal 6.66e-16 überein. Alpha ist keine mathematische Metrik; Alpha, SND und Frobenius definieren unterschiedliche Geometrien.

| dataset | metric | layer | alignment | gauge | rho | ci_low | ci_high | p_perm_greater |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | frobenius | total | optimal | canonical | 0.1165 | -0.5678 | 0.8007 | 0.2935 |
| mnist | frobenius | W0 | optimal | canonical | 0.0866 | -0.4397 | 0.6128 | 0.3641 |
| mnist | frobenius | W1 | optimal | canonical | 0.0539 | -0.5564 | 0.6642 | 0.3572 |
| mnist | frobenius | balanced | optimal | canonical | 0.0296 | -0.5808 | 0.6401 | 0.4137 |
| mnist | cosine | total | optimal | canonical | 0.0515 | -0.3523 | 0.4553 | 0.3581 |
| mnist | cosine | W0 | optimal | canonical | 0.1539 | -0.0824 | 0.3901 | 0.1421 |
| mnist | cosine | W1 | optimal | canonical | -0.0302 | -0.4858 | 0.4255 | 0.5739 |
| mnist | normalized_frobenius | total | optimal | canonical | 0.0515 | -0.3523 | 0.4553 | 0.3581 |
| mnist | normalized_frobenius | W0 | optimal | canonical | 0.1539 | -0.0824 | 0.3901 | 0.1421 |
| mnist | normalized_frobenius | W1 | optimal | canonical | -0.0302 | -0.4858 | 0.4255 | 0.5739 |
| mnist | frobenius | total | optimal | raw | 0.1552 | -0.2187 | 0.5291 | 0.2141 |
| mnist | frobenius | W0 | optimal | raw | -0.0260 | -0.5823 | 0.5304 | 0.5700 |
| mnist | frobenius | W1 | optimal | raw | 0.0561 | -0.3673 | 0.4795 | 0.3403 |
| mnist | frobenius | balanced | optimal | raw | 0.0514 | -0.3622 | 0.4650 | 0.3510 |
| mnist | cosine | total | optimal | raw | 0.0572 | -0.2948 | 0.4092 | 0.3699 |
| mnist | cosine | W0 | optimal | raw | -0.0734 | -0.6353 | 0.4885 | 0.6448 |
| mnist | cosine | W1 | optimal | raw | 0.1092 | -0.4120 | 0.6304 | 0.2252 |
| mnist | normalized_frobenius | total | optimal | raw | 0.0572 | -0.2948 | 0.4092 | 0.3699 |
| mnist | normalized_frobenius | W0 | optimal | raw | -0.0734 | -0.6353 | 0.4885 | 0.6448 |
| mnist | normalized_frobenius | W1 | optimal | raw | 0.1092 | -0.4120 | 0.6304 | 0.2252 |
| mnist | alpha2 | W0 | alpha_profile | canonical | 0.0789 | -0.0983 | 0.2561 | 0.3109 |
| mnist | alpha2 | W1 | alpha_profile | canonical | -0.0043 | -0.6779 | 0.6692 | 0.4979 |
| mnist | alpha2 | balanced | alpha_profile | canonical | 0.0664 | -0.4142 | 0.5470 | 0.3413 |
| mnist | alpha2 | size_weighted | alpha_profile | canonical | 0.0657 | -0.0992 | 0.2306 | 0.3417 |
| mnist | alpha2 | W0 | alpha_profile | raw | -0.1609 | -0.6245 | 0.3028 | 0.8094 |
| mnist | alpha2 | W1 | alpha_profile | raw | 0.1306 | -0.5187 | 0.7798 | 0.2196 |
| mnist | alpha2 | balanced | alpha_profile | raw | 0.0552 | -0.3068 | 0.4172 | 0.3762 |
| mnist | alpha2 | size_weighted | alpha_profile | raw | -0.1543 | -0.6017 | 0.2931 | 0.7993 |
| mnist | snd | W0 | snd_profile | raw | 0.0738 | -0.2952 | 0.4428 | 0.3881 |
| mnist | snd | W1 | snd_profile | raw | 0.0852 | -0.2082 | 0.3787 | 0.2964 |
| mnist | snd | balanced | snd_profile | raw | 0.1071 | -0.2326 | 0.4468 | 0.2485 |
| mnist | snd | size_weighted | snd_profile | raw | 0.0837 | -0.3109 | 0.4782 | 0.3738 |
| fashion_mnist | frobenius | total | optimal | canonical | -0.1806 | -0.5728 | 0.2115 | 0.8401 |
| fashion_mnist | frobenius | W0 | optimal | canonical | -0.4099 | -0.9141 | 0.0943 | 0.9575 |
| fashion_mnist | frobenius | W1 | optimal | canonical | 0.4804 | 0.1161 | 0.8446 | 0.0007 |
| fashion_mnist | frobenius | balanced | optimal | canonical | 0.4679 | 0.1192 | 0.8165 | 0.0009 |
| fashion_mnist | cosine | total | optimal | canonical | 0.0568 | -0.4968 | 0.6104 | 0.3396 |
| fashion_mnist | cosine | W0 | optimal | canonical | -0.0726 | -0.5563 | 0.4111 | 0.6807 |
| fashion_mnist | cosine | W1 | optimal | canonical | 0.4922 | 0.1967 | 0.7877 | 0.0007 |
| fashion_mnist | normalized_frobenius | total | optimal | canonical | 0.0568 | -0.4968 | 0.6104 | 0.3396 |
| fashion_mnist | normalized_frobenius | W0 | optimal | canonical | -0.0726 | -0.5563 | 0.4111 | 0.6807 |
| fashion_mnist | normalized_frobenius | W1 | optimal | canonical | 0.4922 | 0.1967 | 0.7877 | 0.0007 |
| fashion_mnist | frobenius | total | optimal | raw | -0.1976 | -0.6188 | 0.2236 | 0.7973 |
| fashion_mnist | frobenius | W0 | optimal | raw | 0.1530 | -0.3707 | 0.6766 | 0.2318 |
| fashion_mnist | frobenius | W1 | optimal | raw | 0.1112 | -0.4359 | 0.6583 | 0.2622 |
| fashion_mnist | frobenius | balanced | optimal | raw | 0.0859 | -0.4764 | 0.6482 | 0.3139 |
| fashion_mnist | cosine | total | optimal | raw | 0.1781 | -0.3534 | 0.7096 | 0.1794 |
| fashion_mnist | cosine | W0 | optimal | raw | 0.3101 | -0.1459 | 0.7661 | 0.0194 |
| fashion_mnist | cosine | W1 | optimal | raw | 0.1211 | -0.4178 | 0.6599 | 0.2379 |
| fashion_mnist | normalized_frobenius | total | optimal | raw | 0.1781 | -0.3534 | 0.7096 | 0.1794 |
| fashion_mnist | normalized_frobenius | W0 | optimal | raw | 0.3101 | -0.1459 | 0.7661 | 0.0194 |
| fashion_mnist | normalized_frobenius | W1 | optimal | raw | 0.1211 | -0.4178 | 0.6599 | 0.2379 |
| fashion_mnist | alpha2 | W0 | alpha_profile | canonical | -0.0967 | -0.7169 | 0.5235 | 0.7250 |
| fashion_mnist | alpha2 | W1 | alpha_profile | canonical | -0.1735 | -0.6423 | 0.2952 | 0.8102 |
| fashion_mnist | alpha2 | balanced | alpha_profile | canonical | 0.0271 | -0.3711 | 0.4254 | 0.4372 |
| fashion_mnist | alpha2 | size_weighted | alpha_profile | canonical | -0.0972 | -0.7159 | 0.5214 | 0.7270 |
| fashion_mnist | alpha2 | W0 | alpha_profile | raw | -0.1470 | -0.4287 | 0.1347 | 0.8120 |
| fashion_mnist | alpha2 | W1 | alpha_profile | raw | -0.2804 | -0.7983 | 0.2376 | 0.9525 |
| fashion_mnist | alpha2 | balanced | alpha_profile | raw | -0.1856 | -0.6661 | 0.2948 | 0.8369 |
| fashion_mnist | alpha2 | size_weighted | alpha_profile | raw | -0.1556 | -0.4597 | 0.1485 | 0.8223 |
| fashion_mnist | snd | W0 | snd_profile | raw | 0.4372 | -0.1539 | 1.0282 | 0.0148 |
| fashion_mnist | snd | W1 | snd_profile | raw | -0.1734 | -0.7762 | 0.4294 | 0.8693 |
| fashion_mnist | snd | balanced | snd_profile | raw | -0.0872 | -0.5787 | 0.4043 | 0.7227 |
| fashion_mnist | snd | size_weighted | snd_profile | raw | 0.4621 | -0.0840 | 1.0081 | 0.0071 |
| kmnist | frobenius | total | optimal | canonical | 0.2764 | -0.1522 | 0.7051 | 0.0490 |
| kmnist | frobenius | W0 | optimal | canonical | 0.2631 | -0.2715 | 0.7977 | 0.0682 |
| kmnist | frobenius | W1 | optimal | canonical | -0.0016 | -0.5141 | 0.5110 | 0.5079 |
| kmnist | frobenius | balanced | optimal | canonical | -0.0011 | -0.5202 | 0.5181 | 0.5049 |
| kmnist | cosine | total | optimal | canonical | 0.1300 | -0.4037 | 0.6638 | 0.2085 |
| kmnist | cosine | W0 | optimal | canonical | -0.0736 | -0.5614 | 0.4141 | 0.6873 |
| kmnist | cosine | W1 | optimal | canonical | -0.0642 | -0.5486 | 0.4202 | 0.6632 |
| kmnist | normalized_frobenius | total | optimal | canonical | 0.1300 | -0.4037 | 0.6638 | 0.2085 |
| kmnist | normalized_frobenius | W0 | optimal | canonical | -0.0736 | -0.5614 | 0.4141 | 0.6873 |
| kmnist | normalized_frobenius | W1 | optimal | canonical | -0.0642 | -0.5486 | 0.4202 | 0.6632 |
| kmnist | frobenius | total | optimal | raw | 0.1958 | -0.0736 | 0.4652 | 0.1373 |
| kmnist | frobenius | W0 | optimal | raw | -0.2502 | -0.6539 | 0.1535 | 0.9357 |
| kmnist | frobenius | W1 | optimal | raw | 0.4178 | 0.1335 | 0.7021 | 0.0031 |
| kmnist | frobenius | balanced | optimal | raw | 0.4204 | 0.1238 | 0.7170 | 0.0029 |
| kmnist | cosine | total | optimal | raw | -0.0729 | -0.5380 | 0.3923 | 0.6591 |
| kmnist | cosine | W0 | optimal | raw | -0.2426 | -0.6973 | 0.2121 | 0.9546 |
| kmnist | cosine | W1 | optimal | raw | 0.4063 | 0.0799 | 0.7327 | 0.0042 |
| kmnist | normalized_frobenius | total | optimal | raw | -0.0729 | -0.5380 | 0.3923 | 0.6591 |
| kmnist | normalized_frobenius | W0 | optimal | raw | -0.2426 | -0.6973 | 0.2121 | 0.9546 |
| kmnist | normalized_frobenius | W1 | optimal | raw | 0.4063 | 0.0799 | 0.7327 | 0.0042 |
| kmnist | alpha2 | W0 | alpha_profile | canonical | 0.0751 | -0.3402 | 0.4904 | 0.3171 |
| kmnist | alpha2 | W1 | alpha_profile | canonical | -0.2871 | -0.6435 | 0.0694 | 0.9490 |
| kmnist | alpha2 | balanced | alpha_profile | canonical | 0.0856 | -0.3262 | 0.4974 | 0.2840 |
| kmnist | alpha2 | size_weighted | alpha_profile | canonical | 0.0640 | -0.3550 | 0.4830 | 0.3439 |
| kmnist | alpha2 | W0 | alpha_profile | raw | -0.0282 | -0.6058 | 0.5494 | 0.5796 |
| kmnist | alpha2 | W1 | alpha_profile | raw | -0.1088 | -0.7230 | 0.5054 | 0.7667 |
| kmnist | alpha2 | balanced | alpha_profile | raw | -0.2145 | -0.6253 | 0.1964 | 0.9079 |
| kmnist | alpha2 | size_weighted | alpha_profile | raw | -0.0321 | -0.6114 | 0.5471 | 0.5877 |
| kmnist | snd | W0 | snd_profile | raw | 0.1414 | -0.2787 | 0.5614 | 0.1747 |
| kmnist | snd | W1 | snd_profile | raw | -0.0275 | -0.5479 | 0.4928 | 0.5697 |
| kmnist | snd | balanced | snd_profile | raw | -0.0049 | -0.5092 | 0.4995 | 0.5126 |
| kmnist | snd | size_weighted | snd_profile | raw | 0.1354 | -0.2189 | 0.4898 | 0.1799 |


## Alignment-Sensitivität und Zykluskonsistenz

`optimal` berechnet die Paarpermutation an jedem Checkpoint neu. `fixed_E0` und `fixed_E20` halten für jedes Paar dessen Anfangs- bzw. Endzuordnung über **alle** Checkpoints fest, einschließlich der Referenz-RDM bei E0. `fixed_E20` ist eine vom Endzustand informierte Sensitivitätskontrolle, keine zusätzliche unabhängige Bestätigung.

| dataset | alignment | rho | ci_low | ci_high | p_perm_greater |
| --- | --- | --- | --- | --- | --- |
| mnist | optimal | 0.1165 | -0.5678 | 0.8007 | 0.2935 |
| mnist | fixed_E0 | 0.2986 | -0.1153 | 0.7124 | 0.0449 |
| mnist | fixed_E20 | 0.1559 | -0.4384 | 0.7502 | 0.2395 |
| fashion_mnist | optimal | -0.1806 | -0.5728 | 0.2115 | 0.8401 |
| fashion_mnist | fixed_E0 | 0.3585 | 0.0109 | 0.7061 | 0.0128 |
| fashion_mnist | fixed_E20 | 0.1663 | -0.3796 | 0.7122 | 0.1727 |
| kmnist | optimal | 0.2764 | -0.1522 | 0.7051 | 0.0490 |
| kmnist | fixed_E0 | -0.0294 | -0.6245 | 0.5657 | 0.5485 |
| kmnist | fixed_E20 | 0.3202 | -0.0919 | 0.7322 | 0.0167 |

Anteil gegenüber E0 veränderter Einzelneuronenzuordnungen am Endpunkt:

| dataset | change_fraction_vs_E0_mean | change_fraction_vs_E0_min | change_fraction_vs_E0_max |
| --- | --- | --- | --- |
| fashion_mnist | 0.9351 | 0.8281 | 1.0000 |
| kmnist | 0.9101 | 0.8125 | 0.9688 |
| mnist | 0.9205 | 0.8281 | 0.9844 |

Paarweise optimale Zuordnungen sind nicht automatisch über alle Netze zykluskonsistent. Für jedes der 120 Netzwerk-Tripel je Datensatz wird P(B→C)∘P(A→B) mit P(A→C) verglichen. Die orientierten Kompositionen sind separat getestet.

| dataset | checkpoint | mean_fraction_consistent | exactly_consistent_triplets | n_triplets |
| --- | --- | --- | --- | --- |
| fashion_mnist | 0 | 0.1577 | 0 | 120 |
| fashion_mnist | 20 | 0.2898 | 0 | 120 |
| kmnist | 0 | 0.1339 | 0 | 120 |
| kmnist | 20 | 0.2654 | 0 | 120 |
| mnist | 0 | 0.1470 | 0 | 120 |
| mnist | 20 | 0.2693 | 0 | 120 |

Die Distanz wird paarweise auf dem Raum kanonischer Gewichte modulo Hidden-Unit-Permutationen bestimmt; eine einzige globale Neuronenbeschriftung wird daraus nicht abgeleitet.

## Normkontrollen

Alle Roh- und kanonischen W0-/W1-/Gesamtnormen sind in `data/norms.csv` gespeichert. Die Kontrollen korrelieren Summe und absolute Differenz der initialen Paarnormen mit Anfangs- und Endabständen. Die partielle Spearman-Korrelation residualisiert die Rangvariablen gegen einen Intercept und beide rangtransformierten Normprädiktoren; der Jackknife rangiert nach Netzwerkausschluss neu. Beide Gauges und alle drei Layer-Aggregationen werden analysiert.

Partielle RSA für die Gesamtdistanz:

| dataset | gauge | rho | ci_low | ci_high | p_perm_greater |
| --- | --- | --- | --- | --- | --- |
| mnist | raw | 0.0484 | -0.4211 | 0.5179 | 0.3840 |
| mnist | canonical | 0.1844 | -0.1526 | 0.5214 | 0.1010 |
| fashion_mnist | raw | 0.1197 | -0.5425 | 0.7820 | 0.2758 |
| fashion_mnist | canonical | -0.0174 | -0.7292 | 0.6943 | 0.5401 |
| kmnist | raw | 0.1034 | -0.2765 | 0.4833 | 0.2701 |
| kmnist | canonical | 0.2112 | -0.2132 | 0.6355 | 0.0941 |

Run-Level-Korrelation initialer gegen finale Normen:

| dataset | gauge | layer | rho | ci_low | ci_high | p_perm_greater |
| --- | --- | --- | --- | --- | --- | --- |
| mnist | raw | W0 | 0.2606 | -0.4158 | 0.9370 | 0.2294 |
| mnist | raw | W1 | 0.4182 | -0.4219 | 1.2583 | 0.1180 |
| mnist | raw | total | 0.3212 | -0.3909 | 1.0333 | 0.1811 |
| mnist | canonical | W0 | 0.3818 | -0.3112 | 1.0748 | 0.1407 |
| mnist | canonical | W1 | 0.3818 | -0.3112 | 1.0748 | 0.1407 |
| mnist | canonical | total | 0.3818 | -0.3112 | 1.0748 | 0.1407 |
| fashion_mnist | raw | W0 | -0.1879 | -1.0532 | 0.6774 | 0.7118 |
| fashion_mnist | raw | W1 | 0.1152 | -0.7378 | 0.9681 | 0.3750 |
| fashion_mnist | raw | total | 0.1515 | -0.7104 | 1.0135 | 0.3470 |
| fashion_mnist | canonical | W0 | -0.0061 | -0.8297 | 0.8176 | 0.5121 |
| fashion_mnist | canonical | W1 | -0.0061 | -0.8297 | 0.8176 | 0.5121 |
| fashion_mnist | canonical | total | -0.0061 | -0.8297 | 0.8176 | 0.5121 |
| kmnist | raw | W0 | -0.4424 | -1.1773 | 0.2925 | 0.9074 |
| kmnist | raw | W1 | -0.5273 | -1.0834 | 0.0289 | 0.9428 |
| kmnist | raw | total | 0.3333 | -0.4217 | 1.0884 | 0.1753 |
| kmnist | canonical | W0 | -0.4182 | -0.9976 | 0.1613 | 0.8930 |
| kmnist | canonical | W1 | -0.4182 | -0.9976 | 0.1613 | 0.8930 |
| kmnist | canonical | total | -0.4182 | -0.9976 | 0.1613 | 0.8930 |

Die vollständigen Normkontrollen stehen in `norm_controls.csv`. Die Normprädiktoren bleiben bei der Identitätspermutation fest. Die daraus entstehenden partiellen p-Werte sind **deskriptive Sensitivitätstests**, kein allgemein exakter konditionaler Test unter beliebiger Kovariatenabhängigkeit. Die Normkontrollen verändern den primären Test nicht.

## Abstandskonzentration und Rangstabilität

| dataset | mean | std | CV (%) | min | max | rank_gap_min_positive | n_unique | tied_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | 4.7715 | 0.0334 | 0.6998 | 4.6958 | 4.8439 | 4.58e-05 | 45 | 0 |
| fashion_mnist | 4.7609 | 0.0358 | 0.7510 | 4.6882 | 4.8587 | 1.71e-05 | 45 | 0 |
| kmnist | 4.7413 | 0.0277 | 0.5840 | 4.6882 | 4.7886 | 3.79e-05 | 45 | 0 |

Streuung, exakte Rangbindungen und Rangabstände sind für alle Varianten an E0 und E20 in `distance_concentration.csv` gespeichert. Bei den Anfangsabständen ist die relative Streuung klein. Das begrenzt die Interpretation einer Rangordnung aus einem endlichen Initialisierungsensemble. Der Leave-one-network-Jackknife quantifiziert ihre Auswirkung auf die RSA; seine Einzelreplikate stehen in `rsa_jackknife.csv` und den NPZ-Dateien.

Unabhängig neu gerechnete primäre Abstände an E0/E20 sowie separate float32/float64-Sensitivität:

| dataset | checkpoint | n_pairs | max_abs_independent_distance_error | max_abs_reverse_distance_error | max_abs_squared_distance_algebra_error | float32_vs_float64_rank_rho | float32_vs_float64_max_abs_distance_error | float32_vs_float64_assignment_disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist | 0 | 45 | 0.0000 | 8.88e-16 | 1.07e-14 | 1.0000 | 4.79e-07 | 0.0000 |
| mnist | 20 | 45 | 0.0000 | 3.55e-15 | 1.71e-13 | 1.0000 | 1.27e-06 | 0.0000 |
| fashion_mnist | 0 | 45 | 0.0000 | 8.88e-16 | 1.07e-14 | 1.0000 | 4.19e-07 | 0.0000 |
| fashion_mnist | 20 | 45 | 0.0000 | 3.55e-15 | 1.71e-13 | 1.0000 | 1.05e-06 | 0.0000 |
| kmnist | 0 | 45 | 0.0000 | 8.88e-16 | 7.11e-15 | 1.0000 | 4.42e-07 | 0.0000 |
| kmnist | 20 | 45 | 0.0000 | 3.55e-15 | 2.27e-13 | 1.0000 | 2e-06 | 0.0000 |

Die float32-Kontrolle betrifft ausschließlich numerische Rechengenauigkeit. Sie prüft keine Stabilität unter neuen Zufallsinitialisierungen. Die Hauptanalyse arbeitet durchgehend float64 auf den gespeicherten float32-Gewichten.

## Optionale Nachbarschaftserhaltung

| dataset | k | retention | expected_uniform_retention | p_perm_greater | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| mnist | 1 | 0.3000 | 0.1111 | 0.1248 | -0.3663 | 0.9663 |
| mnist | 3 | 0.3667 | 0.3333 | 0.4163 | 0.0556 | 0.6777 |
| fashion_mnist | 1 | 0.2000 | 0.1111 | 0.3021 | -0.1920 | 0.5920 |
| fashion_mnist | 3 | 0.2333 | 0.3333 | 0.9105 | -0.1317 | 0.5984 |
| kmnist | 1 | 0.0000 | 0.1111 | 1.0000 | -0.3201 | 0.3201 |
| kmnist | 3 | 0.4333 | 0.3333 | 0.1595 | 0.0587 | 0.8080 |

Top-k-Retention ist der mittlere Anteil initialer k-Nachbarn, die am Ende noch zu den k-Nachbarn desselben Netzes zählen. Zufallserwartung k/9. Auch hier werden ganze Netzwerkidentitäten permutiert; ein ausgeschlossener Run entfernt dessen Zeile und Spalte, anschließend werden Nachbarn neu bestimmt. Diese Tests sind explorativ.

## Inferenz, Grenzen und Prüfbarkeit

Primäre Endpunkte: drei einseitige positive Identitätspermutationstests, Holm-korrigiert. Zusätzlich zweiseitige Tests. Alle anderen Metriken, Zeitpunkte, Schichten, Norm- und Nachbarschaftstests sind explorativ, ohne familienweite Fehlerkontrolle. 10.000 Permutationen je Datensatz, unabhängig zwischen Datensätzen; innerhalb eines Datensatzes werden dieselben Mappings für alle Varianten verwendet. Die Fisher-z-Zusammenfassung nimmt den gleichgewichteten Mittelwert der drei transformierten Korrelationen und permutiert die drei Strata unabhängig. E0 ist eine Identitätskontrolle ohne Fisher-z-Inferenz. Fisher-Grenzwerte werden nur rechnerisch auf ±(1−10⁻¹²) begrenzt.

Nullverteilungen werden float64 berechnet, p-Werte daraus bestimmt und Nullarrays platzsparend als float32 gespeichert; Identitätsmappings erlauben ihre exakte Neuberechnung. 3120 Nicht-E0-Nullverteilungen wurden auf Zentrierung geprüft; 0 lagen mit ihrem Mittel außerhalb von sechs Monte-Carlo-Standardfehlern um null (maximal |z|=3.586). Das ist eine Pipeline-Diagnose und kein weiterer Hypothesentest.

Vollständige Definitionen:

| Schlüssel | Definition | Zuordnung |
| --- | --- | --- |
| frobenius/total/optimal/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W0/optimal/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W1/optimal/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/balanced/optimal/canonical | sqrt(0.5*(//delta W0//_F^2 / n0 + //delta W1//_F^2 / n1)); RMS of layer RMS distances | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/total/optimal/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W0/optimal/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W1/optimal/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/total/optimal/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W0/optimal/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W1/optimal/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/total/fixed_E0/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W0/fixed_E0/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W1/fixed_E0/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/balanced/fixed_E0/canonical | sqrt(0.5*(//delta W0//_F^2 / n0 + //delta W1//_F^2 / n1)); RMS of layer RMS distances | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/total/fixed_E0/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W0/fixed_E0/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W1/fixed_E0/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/total/fixed_E0/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W0/fixed_E0/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W1/fixed_E0/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/total/fixed_E20/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W0/fixed_E20/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W1/fixed_E20/canonical | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/balanced/fixed_E20/canonical | sqrt(0.5*(//delta W0//_F^2 / n0 + //delta W1//_F^2 / n1)); RMS of layer RMS distances | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/total/fixed_E20/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W0/fixed_E20/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W1/fixed_E20/canonical | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/total/fixed_E20/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W0/fixed_E20/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W1/fixed_E20/canonical | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/total/optimal/raw | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W0/optimal/raw | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/W1/optimal/raw | sqrt(sum of squared elementwise differences) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| frobenius/balanced/optimal/raw | sqrt(0.5*(//delta W0//_F^2 / n0 + //delta W1//_F^2 / n1)); RMS of layer RMS distances | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/total/optimal/raw | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W0/optimal/raw | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| cosine/W1/optimal/raw | 1 - <A,B>/(//A//_F //B//_F) | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/total/optimal/raw | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W0/optimal/raw | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| normalized_frobenius/W1/optimal/raw | sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors | Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment) |
| alpha2/W0/alpha_profile/canonical | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's direct canonical-Alpha profile Hungarian |
| alpha2/W1/alpha_profile/canonical | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's direct canonical-Alpha profile Hungarian |
| alpha2/balanced/alpha_profile/canonical | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's direct canonical-Alpha profile Hungarian |
| alpha2/size_weighted/alpha_profile/canonical | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's direct canonical-Alpha profile Hungarian |
| alpha2/W0/alpha_profile/raw | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| alpha2/W1/alpha_profile/raw | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| alpha2/balanced/alpha_profile/raw | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| alpha2/size_weighted/alpha_profile/raw | sum /a*b/^2 * sign_disagreement / sum /a*b/^2 | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| snd/W0/snd_profile/raw | mean entrywise SND: 0 for both zero; /a-b///a+b/ for equal nonzero signs (denominator>1e-12); 1 otherwise | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| snd/W1/snd_profile/raw | mean entrywise SND: 0 for both zero; /a-b///a+b/ for equal nonzero signs (denominator>1e-12); 1 otherwise | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| snd/balanced/snd_profile/raw | mean entrywise SND: 0 for both zero; /a-b///a+b/ for equal nonzero signs (denominator>1e-12); 1 otherwise | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |
| snd/size_weighted/snd_profile/raw | mean entrywise SND: 0 for both zero; /a-b///a+b/ for equal nonzero signs (denominator>1e-12); 1 otherwise | Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian |


## Wiederholung

Vom Projektverzeichnis: `.venv/bin/python 01_RSA/code/run.py --stage all`. Einzelstufen: `distances`, `inference`, `plots`, `report`, `finalize`. Tests: `.venv/bin/python -m pytest -q 01_RSA/code/test_*.py`. `rsa_config.json` und `artifact_manifest.json` dokumentieren Quellcode, Quellen, Softwareversionen, Seeds und Ausgabeprüfsummen. Die Originaldateien werden gegen `provenance/source_inventory_before.json` geprüft.
