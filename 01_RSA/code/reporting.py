"""Data-bound German reports and local figure index for experiment 01_RSA."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd

DATASETS = ("mnist", "fashion_mnist", "kmnist")
NAMES = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST"}
PRIMARY = dict(metric="frobenius", layer="total", alignment="optimal", gauge="canonical")
FIGURES = [
    ("rsa_initial_vs_final", "Initialisierung gegen Endzustand", "Je Datensatz 45 abhängige Paarabstände aus zehn Netzen; primäre kanonische Frobenius-Distanz."),
    ("rsa_over_training", "RSA über das Training", "E0 bis E20; Bänder zeigen punktweise 95%-Intervalle aus dem Netzwerk-Jackknife."),
    ("distance_distributions", "Streuung der Abstände", "Anfangs- und Endverteilungen mit getrennten Achsenskalierungen, Standardabweichung und Variationskoeffizient."),
    ("rsa_robustness", "Metrik- und Schichtkontrollen", "Die verschiedenen Distanzdefinitionen beschreiben unterschiedliche Geometrien. Alle Werte stammen aus denselben gespeicherten Trainingsläufen."),
    ("alignment_sensitivity", "Feste und wechselnde Zuordnungen", "Für feste Anfangs- oder Endzuordnungen wird dieselbe Paarpermutation auch für die jeweilige Anfangs-RDM verwendet."),
    ("norm_controls", "Kontrolle initialer Gewichtsnormen", "Deskriptive partielle Rangkorrelation nach Kontrolle von Summe und absoluter Differenz initialer Gesamtnormen."),
]


def select(frame, **criteria):
    result = frame
    for key, value in criteria.items():
        result = result[result[key].eq(value)]
    return result


def fmt(value):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        return f"{value:.3g}" if 0 < abs(value) < 1e-4 else f"{value:.4f}"
    return str(value).replace("|", "/")


def table(frame, columns=None, labels=None):
    if columns is not None:
        frame = frame[columns]
    headers = labels or list(frame.columns)
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)) + "\n"


def make_reports(out: Path):
    out = Path(out)
    rsa = pd.read_csv(out / "rsa_by_dataset.csv")
    endpoint = select(rsa, checkpoint=20, **PRIMARY).copy()
    endpoint["dataset"] = pd.Categorical(endpoint.dataset, DATASETS, ordered=True)
    endpoint = endpoint.sort_values("dataset")
    endpoint["Dataset"] = endpoint.dataset.map(NAMES)
    endpoint["95%-JK-Intervall"] = [f"[{a:.3f}, {b:.3f}]" for a,b in zip(endpoint.ci_low, endpoint.ci_high)]
    agg = pd.read_csv(out / "aggregate.csv").query("checkpoint == 20").iloc[0]
    concentration = pd.read_csv(out / "distance_concentration.csv")
    initial = select(concentration, checkpoint=0, **PRIMARY).copy()
    initial["CV (%)"] = 100 * initial.cv
    norm = pd.read_csv(out / "norm_controls.csv")
    normpartial = select(norm, analysis="partial_rsa", layer="total")
    neighbors = pd.read_csv(out / "neighbor_retention.csv")
    validation = json.loads((out / "data/validation.json").read_text())
    diagnostics = json.loads((out / "inference_diagnostics.json").read_text())
    numerical = pd.read_csv(out / "numerical_stability.csv")
    cycle = pd.read_csv(out / "data/cycle_consistency_summary.csv")
    assignment = pd.read_csv(out / "data/assignment_change_summary.csv")
    end_csv = rsa[rsa.checkpoint.eq(20)].copy()
    end_csv.to_csv(out / "rsa_endpoints_all_metrics.csv", index=False)
    robust_positive = bool((endpoint.rho > 0).all() and (endpoint.p_holm_greater < .05).all())
    conclusion = ("Das vorgegebene Kriterium eines über alle drei Datensätze reproduzierbaren positiven Zusammenhangs ist erfüllt."
                  if robust_positive else "Das vorgegebene Kriterium eines robusten, über die drei Datensätze reproduzierbaren positiven Zusammenhangs ist nicht erfüllt.")
    first = select(rsa, checkpoint=1, **PRIMARY).set_index("dataset")
    cosine = select(rsa, checkpoint=20, metric="cosine", layer="total", alignment="optimal", gauge="canonical").set_index("dataset")
    norm_sum = select(norm, analysis="norm_predicts_distance", checkpoint=0, layer="total", gauge="canonical", predictor="norm_sum")
    primary_table = table(endpoint, ["Dataset", "rho", "95%-JK-Intervall", "p_perm_greater", "p_holm_greater", "p_perm_two_sided"],
                          ["Datensatz", "Spearman ρ", "95%-JK-Intervall", "p positiv", "p positiv, Holm", "p zweiseitig"])
    point_count = len(rsa)
    summary = f"""# 01_RSA: Persistiert die Geometrie der Initialisierungen?

{conclusion}

## Beobachtung

Die neue Analyse verwendet ausschließlich die abgeschlossene und geprüfte Hauptkohorte der Paper-Reproduktion: 30 gespeicherte Netze, zehn je Datensatz, 21 Checkpoints von E0 bis E20 (zehn Epochen). Es wurde nicht neu trainiert. Die optionale separate Frühtrainingskohorte wurde nicht einbezogen.

Primärer Endpunkt ist die Spearman-Korrelation der 45 paarweisen Anfangs- und Endabstände **innerhalb jedes Datensatzes**. Die Abstände sind die gemeinsame W0/W1-Frobenius-Distanz nach positiver ReLU-Gauge-Normierung und global optimaler gemeinsamer Hungarian-Zuordnung. Dieselbe Definition gilt an jedem Checkpoint.

{primary_table}
Über die drei Datensätze ergibt die gleichgewichtete Fisher-z-Zusammenfassung eine zurücktransformierte Korrelation von **{agg.rho:.3f}**, ein 95%-Netzwerk-Jackknife-Intervall von [{agg.ci_low:.3f}, {agg.ci_high:.3f}] und einseitiges Permutations-p = {agg.p_perm_greater:.4f}. Das ist eine sekundäre Zusammenfassung; die Einzelergebnisse bleiben entscheidend.

Bereits bei E1 (einer halben Epoche) sinkt die primäre RSA auf MNIST {first.loc['mnist','rho']:.3f}, Fashion-MNIST {first.loc['fashion_mnist','rho']:.3f} und KMNIST {first.loc['kmnist','rho']:.3f}. Danach unterscheiden sich die Verläufe; Fashion-MNIST wechselt im späteren Training das Vorzeichen. Ein gemeinsames stabiles positives Plateau wird damit nicht beobachtet.

Die normierte Gesamtgeometrie (Cosine bzw. monoton äquivalente normierte Frobenius-Distanz) zeigt am Endpunkt nur schwache Zusammenhänge: MNIST {cosine.loc['mnist','rho']:.3f}, Fashion-MNIST {cosine.loc['fashion_mnist','rho']:.3f}, KMNIST {cosine.loc['kmnist','rho']:.3f}. Die Anfangsdistanzen korrelieren dagegen deutlich mit der Summe der initialen kanonischen Gesamtnormen (ρ = {norm_sum.rho.min():.3f} bis {norm_sum.rho.max():.3f}). Auch Schichten und festgehaltene Alignments liefern datensatzabhängige Ergebnisse; sie stützen keine einheitliche positive Persistenz über alle drei Datensätze.

## Statistische Evidenz

Je Datensatz wurden **10.000 Identitätspermutationen** gerechnet: gleichzeitige Zeilen- und Spaltenpermutation der Vergleichs-RDM, bei festgehaltener Anfangs-RDM. Die Seeds sind MNIST 2026111601, Fashion-MNIST 2026111602 und KMNIST 2026111603. p = (1 + Anzahl mindestens so extremer Nullwerte) / 10.001; kleinster darstellbarer p-Wert ist 0,000100. Die gerichtete Hypothese testet einen positiven Zusammenhang; zweiseitige Werte sind zusätzlich angegeben. Holm korrigiert die Familie der drei primären Datensatztests.

Die Replikationseinheiten sind **zehn Netzwerke je Datensatz, nicht 45 unabhängige Paare**. Der Jackknife entfernt jeweils ein Netz mit allen neun zugehörigen Paaren und berechnet die Ränge auf den verbleibenden 36 Paaren neu. Normalapproximierte Intervalle mit nur zehn Netzen sind grob; sie sind weder simultane Konfidenzbänder noch ein Äquivalenztest. Die gespeicherten Intervallgrenzen sind unbeschnitten; Figuren begrenzen sie auf den Korrelationsbereich.

Für E0 gilt überall ρ = 1; dafür werden keine inferenziellen p-Werte berichtet. Die Nullmittelwerte der drei primären Endpunkte liegen bei {', '.join(f'{NAMES[r.dataset]} {r.null_mean:+.4f}' for r in endpoint.itertuples())}. Eine einzelne Identitätspermutation muss nicht exakt null ergeben. Die Nullverteilung und ihr Monte-Carlo-Standardfehler sind gespeichert.

## Interpretation

Die beobachtete paarweise Anfangsgeometrie liefert unter dem festgelegten primären Protokoll keine konsistente positive Vorhersage der Endgeometrie über alle drei Aufgaben. Das Ergebnis schließt historische Information in anderen Merkmalen der Endgewichte nicht aus. Es zeigt weder die Rekonstruierbarkeit einer vollständigen Lerntrajektorie noch einen dynamischen Attraktor.

Die Anfangsabstände sind stark konzentriert: ihre Variationskoeffizienten liegen zwischen {initial['CV (%)'].min():.2f}% und {initial['CV (%)'].max():.2f}%. Numerische Rangstabilität wurde unabhängig mit float32- und float64-Rechnung geprüft; das ersetzt keine Replikation mit neuen Seeds. Metrik-, Schicht-, Norm- und Alignment-Kontrollen sind explorativ und in [rsa_robustness.md](rsa_robustness.md) vollständig dokumentiert. Signifikante Einzelkontrollen wären keine Bestätigung über unabhängige Datensätze und kein Grund, das primäre Ergebnis zu ersetzen.

## Ergebnisse öffnen und weiterverwenden

- [Alle Abbildungen mit PNG/PDF-Links](index.html)
- [Vollständige RSA-Tabelle](rsa_by_dataset.csv): {point_count} Zeilen, 52 Varianten × 21 Checkpoints × drei Datensätze.
- [Alle Endpunktvarianten](rsa_endpoints_all_metrics.csv), [Fisher-z-Zusammenfassung](aggregate.csv), [Robustheitsbericht](rsa_robustness.md).
- [Konfiguration und Provenienz](rsa_config.json), [Distanzdefinitionen](data/distance_definitions.json), [unabhängiger Rechenaudit](provenance/independent_audit.json).
- `data/*_distances.npz`: alle RDMs, Paarzuordnungen, Run-IDs, Seeds, Normen und Neuronenpermutationen.
- `data/*_inference.npz` und `data/*_controls.npz`: Identitätspermutationen, Nullverteilungen und Jackknife-Replikate.

Diese RSA ist eine zusätzliche Analyse nach der neuen Spezifikation, kein im Originalpaper berichtetes Experiment. Die ursprünglichen Trainings-, Ergebnis- und Figurendateien bleiben erhalten; [Prüfsummenvergleich](provenance/preservation_check.json).
"""
    (out / "rsa_summary.md").write_text(summary, encoding="utf-8")
    definitions = json.loads((out / "data/distance_definitions.json").read_text())
    dtable = pd.DataFrame([{"Schlüssel": k, "Definition": v["formula"], "Zuordnung": v["assignment_objective"]} for k,v in definitions.items()])
    layer = select(end_csv, metric="frobenius", alignment="optimal", gauge="canonical")
    controls = end_csv[~end_csv.alignment.isin(["fixed_E0", "fixed_E20"])]
    fixed = select(end_csv, metric="frobenius", layer="total", gauge="canonical")
    norm_run = pd.read_csv(out / "norm_run_correlations.csv")
    robustness = f"""# 01_RSA: Robustheit und genaue Auswertungsdefinitionen

## Stichprobe und Erhaltung der Originaldaten

30 unabhängige Hauptkohorten-Netze; 10 pro Datensatz; je 45 abhängige Within-Dataset-Paare; 135 Paare insgesamt ohne Between-Dataset-Pooling. 21 Checkpoints, also 630 Netzwerkzustände und 2.835 Paar-Checkpoint-Kombinationen. 52 fest definierte Varianten ergeben 147.420 Distanzwerte und {point_count} RSA-Schätzer. Alle Netzwerkidentitäten stammen aus `outputs/manifests/main.json`, sortiert nach Seed innerhalb jedes Datensatzes. Null-Biases, Shapes, Checkpointfolge, Metadaten, NPZ-/PT-Hashes und 21 vorhandene Originalanalyse-Receipts wurden geprüft.

Die Reproduktion war vor Start abgeschlossen: alle 17 primären qualitativen Prüfungen bestanden; alle 65 Trainingsläufe waren in einer unabhängigen Wiederholung bitgleich. Diese Zusatzanalyse schreibt ausschließlich in `01_RSA`. Die alte Pipeline wurde nicht überschrieben. Es wurden keine neuen Trainingsläufe und keine optionale separate Frühkohorten-Replikation erzeugt.

## Kanonische Gauge und primäre Distanz

Pro Hidden Unit wird s = sqrt(||out||₂ / ||in||₂) verwendet, dann W0[:,h] *= s und W1[h,:] /= s. Das erhält die ReLU-Funktion. Keine der {validation['zero_norm_totals']['both_positive']:,} geprüften Neuron-Checkpoint-Konfigurationen hatte eine Nullnorm. Deterministische Sonderfallkonvention: nur eingehend nichtnull -> s=1/||in||; nur ausgehend nichtnull -> s=||out||; beide null -> s=1. Für einseitige Nullnormen ist die Normangleichung mit endlicher positiver Skalierung nicht möglich; die aktive Seite wird stattdessen auf Eins normiert. Diese Konvention wurde getestet, aber in den vorliegenden Daten nicht benötigt.

Die gemeinsame Hungarian-Zuordnung maximiert A0ᵀB0 + A1B1ᵀ auf den kanonischen Gewichten. Für dieses einzelne Hidden Layer ist sie das globale Optimum der **gemeinsamen** quadratischen Frobenius-Distanz. Berichtet wird die Quadratwurzel. Auch layerweise Werte verwenden dieselbe gemeinsame Zuordnung, keine getrennten Layer-Optima. Die Gauge, Distanzformel und Matching-Zielfunktion wechseln über E0..E20 nicht.

## Schichten und Metrikrobustheit

{table(layer, ['dataset','layer','rho','ci_low','ci_high','p_perm_greater'])}
Layer-balancierte Frobenius-Distanz = sqrt(0,5 × (||ΔW0||² / 50.176 + ||ΔW1||² / 640)). Das ist die quadratische Mittelung der beiden Layer-RMS-Werte; die Zuordnung optimiert weiterhin die gemeinsame ungewichtete Distanz. Cosine-Dissimilarität = 1 − <A,B>/(||A|| ||B||); normierte Frobenius-Distanz = sqrt(2 × Cosine-Dissimilarität). Letztere beiden sind bei derselben Zuordnung streng monoton äquivalent und liefern dieselbe Spearman-RSA: sie sind keine zwei unabhängigen Robustheitsbelege.

Alpha (Exponent 2) und SND verwenden exakt die zuvor reproduzierten metric-spezifischen Profilzuordnungen. Kanonische Alpha nutzt die direkte Profil-Hungarian-Lösung; rohe Alpha und SND die reproduzierte PIF-/Profilpipeline. Es werden keine PIF/Alpha-Benchmark-Kandidaten an deren Stelle eingesetzt. Alle Werte wurden aus den Gewichten neu ausgewertet und stimmen mit den vorhandenen Paartabellen bis maximal {validation['max_pipeline_dissimilarity_recompute_error']:.3g} überein. Alpha ist keine mathematische Metrik; Alpha, SND und Frobenius definieren unterschiedliche Geometrien.

{table(controls, ['dataset','metric','layer','alignment','gauge','rho','ci_low','ci_high','p_perm_greater'])}

## Alignment-Sensitivität und Zykluskonsistenz

`optimal` berechnet die Paarpermutation an jedem Checkpoint neu. `fixed_E0` und `fixed_E20` halten für jedes Paar dessen Anfangs- bzw. Endzuordnung über **alle** Checkpoints fest, einschließlich der Referenz-RDM bei E0. `fixed_E20` ist eine vom Endzustand informierte Sensitivitätskontrolle, keine zusätzliche unabhängige Bestätigung.

{table(fixed, ['dataset','alignment','rho','ci_low','ci_high','p_perm_greater'])}
Anteil gegenüber E0 veränderter Einzelneuronenzuordnungen am Endpunkt:

{table(assignment[assignment.checkpoint.eq(20)], ['dataset','change_fraction_vs_E0_mean','change_fraction_vs_E0_min','change_fraction_vs_E0_max'])}
Paarweise optimale Zuordnungen sind nicht automatisch über alle Netze zykluskonsistent. Für jedes der 120 Netzwerk-Tripel je Datensatz wird P(B→C)∘P(A→B) mit P(A→C) verglichen. Die orientierten Kompositionen sind separat getestet.

{table(cycle)}
Die Distanz wird paarweise auf dem Raum kanonischer Gewichte modulo Hidden-Unit-Permutationen bestimmt; eine einzige globale Neuronenbeschriftung wird daraus nicht abgeleitet.

## Normkontrollen

Alle Roh- und kanonischen W0-/W1-/Gesamtnormen sind in `data/norms.csv` gespeichert. Die Kontrollen korrelieren Summe und absolute Differenz der initialen Paarnormen mit Anfangs- und Endabständen. Die partielle Spearman-Korrelation residualisiert die Rangvariablen gegen einen Intercept und beide rangtransformierten Normprädiktoren; der Jackknife rangiert nach Netzwerkausschluss neu. Beide Gauges und alle drei Layer-Aggregationen werden analysiert.

Partielle RSA für die Gesamtdistanz:

{table(normpartial, ['dataset','gauge','rho','ci_low','ci_high','p_perm_greater'])}
Run-Level-Korrelation initialer gegen finale Normen:

{table(norm_run, ['dataset','gauge','layer','rho','ci_low','ci_high','p_perm_greater'])}
Die vollständigen Normkontrollen stehen in `norm_controls.csv`. Die Normprädiktoren bleiben bei der Identitätspermutation fest. Die daraus entstehenden partiellen p-Werte sind **deskriptive Sensitivitätstests**, kein allgemein exakter konditionaler Test unter beliebiger Kovariatenabhängigkeit. Die Normkontrollen verändern den primären Test nicht.

## Abstandskonzentration und Rangstabilität

{table(initial, ['dataset','mean','std','CV (%)','min','max','rank_gap_min_positive','n_unique','tied_pairs'])}
Streuung, exakte Rangbindungen und Rangabstände sind für alle Varianten an E0 und E20 in `distance_concentration.csv` gespeichert. Bei den Anfangsabständen ist die relative Streuung klein. Das begrenzt die Interpretation einer Rangordnung aus einem endlichen Initialisierungsensemble. Der Leave-one-network-Jackknife quantifiziert ihre Auswirkung auf die RSA; seine Einzelreplikate stehen in `rsa_jackknife.csv` und den NPZ-Dateien.

Unabhängig neu gerechnete primäre Abstände an E0/E20 sowie separate float32/float64-Sensitivität:

{table(numerical)}
Die float32-Kontrolle betrifft ausschließlich numerische Rechengenauigkeit. Sie prüft keine Stabilität unter neuen Zufallsinitialisierungen. Die Hauptanalyse arbeitet durchgehend float64 auf den gespeicherten float32-Gewichten.

## Optionale Nachbarschaftserhaltung

{table(neighbors, ['dataset','k','retention','expected_uniform_retention','p_perm_greater','ci_low','ci_high'])}
Top-k-Retention ist der mittlere Anteil initialer k-Nachbarn, die am Ende noch zu den k-Nachbarn desselben Netzes zählen. Zufallserwartung k/9. Auch hier werden ganze Netzwerkidentitäten permutiert; ein ausgeschlossener Run entfernt dessen Zeile und Spalte, anschließend werden Nachbarn neu bestimmt. Diese Tests sind explorativ.

## Inferenz, Grenzen und Prüfbarkeit

Primäre Endpunkte: drei einseitige positive Identitätspermutationstests, Holm-korrigiert. Zusätzlich zweiseitige Tests. Alle anderen Metriken, Zeitpunkte, Schichten, Norm- und Nachbarschaftstests sind explorativ, ohne familienweite Fehlerkontrolle. 10.000 Permutationen je Datensatz, unabhängig zwischen Datensätzen; innerhalb eines Datensatzes werden dieselben Mappings für alle Varianten verwendet. Die Fisher-z-Zusammenfassung nimmt den gleichgewichteten Mittelwert der drei transformierten Korrelationen und permutiert die drei Strata unabhängig. E0 ist eine Identitätskontrolle ohne Fisher-z-Inferenz. Fisher-Grenzwerte werden nur rechnerisch auf ±(1−10⁻¹²) begrenzt.

Nullverteilungen werden float64 berechnet, p-Werte daraus bestimmt und Nullarrays platzsparend als float32 gespeichert; Identitätsmappings erlauben ihre exakte Neuberechnung. {diagnostics['null_centering']['distributions']} Nicht-E0-Nullverteilungen wurden auf Zentrierung geprüft; {diagnostics['null_centering']['outside_6mcse']} lagen mit ihrem Mittel außerhalb von sechs Monte-Carlo-Standardfehlern um null (maximal |z|={diagnostics['null_centering']['max_abs_z']:.3f}). Das ist eine Pipeline-Diagnose und kein weiterer Hypothesentest.

Vollständige Definitionen:

{table(dtable)}

## Wiederholung

Vom Projektverzeichnis: `.venv/bin/python 01_RSA/code/run.py --stage all`. Einzelstufen: `distances`, `inference`, `plots`, `report`, `finalize`. Tests: `.venv/bin/python -m pytest -q 01_RSA/code/test_*.py`. `rsa_config.json` und `artifact_manifest.json` dokumentieren Quellcode, Quellen, Softwareversionen, Seeds und Ausgabeprüfsummen. Die Originaldateien werden gegen `provenance/source_inventory_before.json` geprüft.
"""
    (out / "rsa_robustness.md").write_text(robustness, encoding="utf-8")
    cards = []
    for name, title, caption in FIGURES:
        if (out / f"{name}.png").is_file():
            cards.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{html.escape(title)}"></a><p><a href="{name}.png">PNG</a> · <a href="{name}.pdf">Vektor-PDF</a></p></section>')
    rows = ''.join(f'<tr><td>{NAMES[str(r.dataset)]}</td><td>{r.rho:.3f}</td><td>{r.p_perm_greater:.4f}</td><td>{r.p_holm_greater:.4f}</td></tr>' for r in endpoint.itertuples())
    page = f'''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>01_RSA - Initialisierung und Endzustand</title><style>body{{max-width:1180px;margin:40px auto;padding:0 22px;font:17px/1.55 system-ui;color:#233142;background:#f6f8fa}}h1,h2{{line-height:1.25}}header,section{{background:white;padding:26px;margin:22px 0;border:1px solid #dce2e8;border-radius:12px}}img{{width:100%;height:auto}}a{{color:#215d96}}table{{border-collapse:collapse;width:100%;max-width:750px}}td,th{{padding:8px 14px;text-align:left;border-bottom:1px solid #dce2e8}}.lead{{font-size:20px;font-weight:600}}nav a{{margin-right:14px}}small{{color:#596879}}</style><header><h1>01_RSA: Initialisierung und Endzustand</h1><p class="lead">{html.escape(conclusion)}</p><p>30 Netze · 3 Datensätze · 21 Checkpoints · 10.000 Identitätspermutationen je Datensatz</p><table><tr><th>Datensatz</th><th>ρ bei E20</th><th>p positiv</th><th>p positiv, Holm</th></tr>{rows}</table><p><small>Primär: kanonische Gesamt-Frobenius-Distanz, optimale gemeinsame Neuronenzuordnung. Je Datensatz 10 Netze und 45 abhängige Paare.</small></p><nav><a href="rsa_summary.md">Ergebnisbericht</a><a href="rsa_robustness.md">Methoden und Kontrollen</a><a href="rsa_by_dataset.csv">Alle RSA-Werte (CSV)</a><a href="rsa_config.json">Konfiguration</a></nav></header>{''.join(cards)}<footer><p>Zusätzliche Analyse der neu berechneten Paper-Reproduktion. Alle Abbildungen beruhen auf gespeicherten eigenen Trainingsdaten. <a href="../figures/index.html">Originalpaper und Reproduktionsfiguren</a></p></footer></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "README.md").write_text("# 01_RSA\n\nStart: [Ergebnisbericht](rsa_summary.md) oder [Abbildungsgalerie](index.html).\n\nAlle Dateien dieser zusätzlichen Analyse liegen hier. Originalgewichte und Originalanalyse werden ausschließlich gelesen.\n\nWiederholung vom Projektordner: `.venv/bin/python 01_RSA/code/run.py --stage all`. Die frühere Reproduktion muss vollständig vorliegen. Details: [Robustheitsbericht](rsa_robustness.md), [Konfiguration](rsa_config.json), [Dateiinventar](artifact_manifest.json).\n", encoding="utf-8")
