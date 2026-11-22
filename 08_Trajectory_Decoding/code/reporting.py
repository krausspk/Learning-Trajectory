"""Render the held-out trajectory and event probe results, without fitting models.

The input tables, not figures, are the numerical source of truth. This module
never selects models or runs based on the final test performance.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT_NAMES = {"exploratory": "Exploration · 30 Bestandsläufe",
                "one_bit": "Ein Aufgabenpaar · A/B → C",
                "multi_bit": "Vier Aufgabenpaare · A/B → C"}
COLORS = {"ridge": "#2166ac", "mlp": "#762a83", "train_mean": "#777777",
          "persistence": "#d97706", "logistic": "#2166ac", "both": "#2166ac",
          "W0": "#4393c3", "W1": "#b2182b", "diagnostics": "#777777",
          "mean_early_minus_final": "#1b7837"}
LABELS = {"ridge": "Ridge", "mlp": "Kleines MLP", "train_mean": "Trainingsmittelwert",
          "persistence": "Persistenz", "both": "W0 + W1", "W0": "W0", "W1": "W1",
          "diagnostics": "Leistung + Normen", "performance": "Accuracy + Loss",
          "mean_early_minus_final": "Früher Mittelwert − z(T)"}


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelsize": 10,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "axes.grid": True,
                         "grid.alpha": .2, "grid.linewidth": .6})


def save_figure(fig, root: Path, stem: str):
    fig.savefig(root / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _primary_regression(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "layer" in result:
        result = result[result.layer.isin(["both", "W0+W1"])]
    if "pca_dim" in result:
        result = result[result.pca_dim == 8]
    if "model" in result:
        result = result[result.model.isin(["ridge", "train_mean", "persistence", "mean_early_minus_final"])]
    return result


def plot_regression(frame: pd.DataFrame, root: Path):
    primary = _primary_regression(frame)
    cohorts = [x for x in ("one_bit", "multi_bit") if x in set(primary.cohort)]
    if not cohorts:
        cohorts = sorted(primary.cohort.unique())
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(12, 4.1 * len(cohorts)), squeeze=False)
    for row, cohort in enumerate(cohorts):
        for column, target in enumerate(("absolute", "displacement")):
            ax = axes[row, column]
            data = primary[(primary.cohort == cohort) & (primary.target == target)]
            for model in ("ridge", "train_mean", "persistence", "mean_early_minus_final"):
                part = data[data.model == model].sort_values("lag_updates")
                if part.empty:
                    continue
                x, y = part.lag_updates.to_numpy(), part.r2.to_numpy()
                ax.plot(x, y, label=LABELS[model], color=COLORS[model],
                        lw=2 if model == "ridge" else 1.4,
                        ls="-" if model == "ridge" else "--", marker="o", ms=3)
                if model == "ridge" and {"r2_ci_low", "r2_ci_high"}.issubset(part.columns):
                    ax.fill_between(x, part.r2_ci_low.to_numpy(), part.r2_ci_high.to_numpy(),
                                    color=COLORS[model], alpha=.15)
            ax.axhline(0, color="#333333", lw=.7)
            symlog = len(data) and data.r2.min() < -5
            if symlog:
                ax.set_yscale("symlog", linthresh=1, linscale=1.5)
                low = min(data.r2.min(), data.r2_ci_low.min()) if "r2_ci_low" in data else data.r2.min()
                ax.set_ylim(float(low) * 1.35, 1.15)
            ax.set_title(COHORT_NAMES.get(cohort, cohort) + "\n" +
                         ("Frühere Position z(t)" if target == "absolute" else "Verschiebung z(t) − z(T)"))
            ax.set_xlabel("Zeitlicher Abstand zum Endzustand · SGD-Updates")
            ax.set_ylabel("Hold-out R²\nSymm. logarithmisch außerhalb [−1; 1]" if symlog else "Hold-out R²", fontsize=9 if symlog else 10)
            ax.legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("Wie viel der früheren Trajektorie ist aus Endgewichten vorhersagbar?", fontsize=15, y=1.01)
    fig.text(.5, -.015, "Train-only-PCA: 8 Dimensionen · Endpunkt T ausgeschlossen · Bänder: 95%-Seed-Block-Bootstrap-CI",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    save_figure(fig, root, "trajectory_R2_vs_lag")


def plot_events(frame: pd.DataFrame, root: Path):
    data = frame[frame.cohort == "multi_bit"].copy()
    if data.empty:
        return
    if "model" in data:
        data = data[data.model.isin(["logistic", "linear", "ridge_classifier"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, metric, label in zip(axes, ("accuracy", "auc"), ("Balancierte Accuracy", "ROC-AUC")):
        for feature in ("both", "W0", "W1", "diagnostics", "performance"):
            part = data[data.features == feature].sort_values("age_updates")
            if part.empty:
                continue
            x = part.age_updates.to_numpy()
            y = part[metric].to_numpy()
            low, high = f"{metric}_ci_low", f"{metric}_ci_high"
            errors = np.vstack([y-part[low].to_numpy(), part[high].to_numpy()-y]) if low in part and high in part else None
            ax.errorbar(x, y, yerr=errors, marker="o", ms=5, lw=1.7, capsize=3,
                        color=COLORS.get(feature, "#999999"), label=LABELS.get(feature, feature),
                        alpha=1 if feature == "both" else .8)
        ax.axhline(.5, color="#333333", ls="--", lw=1, label="Zufallsniveau")
        ax.set_ylim(.25, 1.02)
        pairs = data.drop_duplicates("bit").sort_values("age_updates")
        ax.set_xticks(pairs.age_updates.to_numpy(),
                      [f"{int(age)}\nPaar {int(bit)+1}" for age, bit in zip(pairs.age_updates, pairs.bit)])
        ax.set_xlabel("Alter seit Ende des Aufgabenpaars · SGD-Updates")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Dekodierbarkeit der vier früheren A/B-Reihenfolgen", fontsize=15)
    footer = "Finalgewichte nach fünf C-Epochen · Zurückgehaltene Seeds · 95%-Seed-Block-CI"
    diagnostic_path = root / "provenance/multi_bit_feasibility_final.json"
    if diagnostic_path.exists():
        feasibility = json.loads(diagnostic_path.read_text())
        if feasibility.get("C_accuracy", {}).get("mean", 1) < .8:
            footer += "\nC-Lernen häufig gescheitert: kein Nachweis unabhängig von aktueller Endleistung"
    fig.text(.5, -.015, footer,
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    save_figure(fig, root, "event_decoding_vs_age")


def plot_examples(paths: list[Path], root: Path):
    """Draw the first saved held-out examples; never select by reconstruction fit.

    Each archive provides true_z/pred_z [run,time,component], run_ids and updates.
    Final coordinates may be present in true_z only, but are marked as known.
    """
    loaded = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        values["cohort"] = str(values.get("cohort", path.parent.name if path.stem == "heldout_examples" else path.stem.replace("_examples", "")))
        loaded.append(values)
    if not loaded:
        return
    columns = min(3, max(len(x["run_ids"]) for x in loaded))
    fig, axes = plt.subplots(len(loaded), columns, figsize=(4.3 * columns, 4 * len(loaded)), squeeze=False)
    for row, values in enumerate(loaded):
        for col in range(columns):
            ax = axes[row, col]
            if col >= len(values["run_ids"]):
                ax.axis("off")
                continue
            truth, pred = values["true_z"][col], values["pred_z"][col]
            if "final_z" in values and len(truth) == len(pred) and np.allclose(truth[-1], values["final_z"][col]):
                # Saved archive appends the known endpoint; its prediction is not a reconstruction.
                pred = pred[:-1]
            ax.plot(truth[:, 0], truth[:, 1], "o-", ms=3, lw=1.6, color="#333333", label="Wahre Positionen")
            ax.plot(pred[:, 0], pred[:, 1], "o--", ms=3, lw=1.6, color=COLORS["ridge"], label="Aus W(T) vorhergesagt")
            ax.scatter(truth[0, 0], truth[0, 1], s=65, marker="s", facecolors="none", edgecolors="#333333", zorder=5,
                       label="Start t₀")
            if "final_z" in values:
                endpoint = values["final_z"][col]
                ax.scatter(endpoint[0], endpoint[1], s=105, marker="*", color="#d97706", zorder=6, label="Bekannter Endpunkt T")
            elif len(truth) > len(pred):
                ax.scatter(truth[-1, 0], truth[-1, 1], s=105, marker="*", color="#d97706", zorder=6, label="Bekannter Endpunkt T")
            ax.set_title(f"{values['run_ids'][col]}\n{COHORT_NAMES.get(values['cohort'], values['cohort'])}", fontsize=9)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Zurückgehaltene Trajektorien: wahre und rekonstruierte Pfade", fontsize=14)
    fig.text(.5, -.015, "Erste Testläufe in gespeicherter Reihenfolge · Darstellung der ersten 2 von 8 PCA-Koordinaten",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    save_figure(fig, root, "trajectory_examples")


def fmt(value, digits=3):
    if value is None or not np.isfinite(float(value)):
        return "–"
    return f"{float(value):.{digits}f}"


def _md_table(frame: pd.DataFrame, columns: list[str], labels: list[str], digits: dict[str, int] | None = None) -> str:
    present = [(key, label) for key, label in zip(columns, labels) if key in frame]
    lines = ["| " + " | ".join(label for _, label in present) + " |",
             "| " + " | ".join("---" for _ in present) + " |"]
    for _, row in frame.iterrows():
        fields = []
        for key, _ in present:
            value = row[key]
            fields.append(fmt(value, (digits or {}).get(key, 3)) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def write_report(root: Path, regression: pd.DataFrame, events: pd.DataFrame | None):
    manifest_path = root / "trajectory_run_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"bits": str}) if manifest_path.exists() else pd.DataFrame()
    primary = _primary_regression(regression)
    initial = primary[(primary.model == "ridge") & (primary["update"] == 0)].sort_values(["cohort", "target"])
    findings = []
    final_feasibility_path = root / "provenance/multi_bit_feasibility_final.json"
    snapshot_feasibility_path = root / "provenance/multi_bit_feasibility_snapshot.json"
    feasibility_path = final_feasibility_path if final_feasibility_path.exists() else snapshot_feasibility_path
    feasibility = json.loads(feasibility_path.read_text()) if feasibility_path.exists() else None
    if feasibility is not None and final_feasibility_path.exists():
        c = feasibility["C_accuracy"]
        if c["mean"] < .8 or c["min"] < .5:
            findings.append("**Die zentrale Voraussetzung der Vier-Bit-Kohorte ist nicht erfüllt:** "
                            "Nach der gemeinsamen C-Phase erreichen die Netze keine vergleichbare, erfolgreiche "
                            f"C-Endleistung. Über alle {feasibility['runs']} Läufe liegt die C-Accuracy "
                            f"im Mittel bei {100*c['mean']:.2f} %, mit einer Spanne von "
                            f"{100*c['min']:.2f} % bis {100*c['max']:.2f} %. "
                            "Die Ereignisprobes werden vollständig berichtet, erlauben unter diesen "
                            "Bedingungen aber keinen Nachweis historischer Information unabhängig "
                            "vom aktuellen Funktions- und Leistungszustand. Alle Läufe bleiben in der Analyse.")
    for cohort in ("one_bit", "multi_bit"):
        subset = primary[(primary.cohort == cohort) & (primary.target == "absolute") & (primary["update"] == 0)]
        r = subset[subset.model == "ridge"]
        if r.empty:
            continue
        r = r.iloc[0]
        means = subset[subset.model == "train_mean"]
        persistence = subset[subset.model == "persistence"]
        comparison = []
        if not means.empty:
            comparison.append(f"Trainingsmittelwert: MSE {fmt(means.iloc[0].mse, 6)}")
        if not persistence.empty:
            comparison.append(f"Persistenz: MSE {fmt(persistence.iloc[0].mse)}")
        interpretation = (" Die Probe ist schlechter als der Trainingsmittelwert; "
                          "die individuelle Startposition wird mit dieser primären Probe nicht besser rekonstruiert."
                          if not means.empty and r.mse >= means.iloc[0].mse else "")
        if not means.empty and r.mse < means.iloc[0].mse:
            gain = 100 * (1 - r.mse / means.iloc[0].mse)
            interpretation = (f" Die MSE liegt als Punktschätzer {gain:.2f} % unter dem Trainingsmittelwert. "
                              "Bei diesem nahezu null liegenden R² ist das keine präzise Startrekonstruktion. "
                              f"Der Zuordnungs-Permutationstest ergibt Holm-p = {fmt(r.get('p_holm'))}; "
                              "er prüft die Assoziation zwischen Vorhersage und individuellem Testverlauf, "
                              "nicht eine Verbesserung gegenüber der Mittelwertbaseline. "
                              "Eine belastbare Verbesserung der Rekonstruktionsgenauigkeit gegenüber "
                              "dieser Baseline ist damit nicht nachgewiesen.")
        findings.append(f"**{COHORT_NAMES[cohort]}:** Am Startzustand erreicht die primäre Ridge-Probe "
                        f"R² = {fmt(r.r2)} [95%-CI {fmt(r.get('r2_ci_low'))}; {fmt(r.get('r2_ci_high'))}] "
                        f"und MSE = {fmt(r.mse, 6)}. " + "; ".join(comparison) + "." + interpretation)
    if events is not None and not events.empty:
        for cohort, title in (("one_bit", "Ein-Bit-Aufgabenreihenfolge"), ("multi_bit", "Frühestes von vier Aufgabenpaaren")):
            e = events[(events.cohort == cohort) & (events.features == "both") &
                       (events.model == "ridge_classifier") & (events.bit == 0)]
            if e.empty:
                continue
            e = e.iloc[0]
            finding = (f"**{title}:** Accuracy = {fmt(e.accuracy)} "
                       f"[empirisches 95%-Bootstrap-CI {fmt(e.get('accuracy_ci_low'))}; {fmt(e.get('accuracy_ci_high'))}], "
                       f"AUC = {fmt(e.auc)}, Blockpermutations-p = {fmt(e.get('p_value'))}, "
                       f"Holm-p = {fmt(e.get('p_holm'))}.")
            diagnostic = events[(events.cohort == cohort) & (events.model == "ridge_classifier") &
                                (events.features == "diagnostics") & (events.bit == 0)]
            perf_only = events[(events.cohort == cohort) & (events.model == "ridge_classifier") &
                              (events.features == "performance") & (events.bit == 0)]
            if not diagnostic.empty:
                finding += f" Die Probe auf finaler Accuracy/Loss und Normen erreicht {100*diagnostic.iloc[0].accuracy:.1f} % Accuracy"
                if not perf_only.empty:
                    finding += f", Accuracy/Loss allein {100*perf_only.iloc[0].accuracy:.1f} %"
                significant_diagnostic = diagnostic.iloc[0].get("p_holm", 1) < .05
                if significant_diagnostic:
                    finding += (". Die Historie ist damit auch mit aktuellen Leistungs-/Normunterschieden "
                                "verbunden; die Gewichtsergebnisse isolieren keine davon unabhängige Erinnerung.")
                else:
                    finding += (". Diese Kontrollwerte sind zusammen mit ihren Unsicherheiten und "
                                "korrigierten p-Werten zu beurteilen; eine Unabhängigkeit von aktuellen "
                                "Funktionsunterschieden wird dadurch nicht nachgewiesen.")
            if e.accuracy == 1:
                finding += (" Die fehlerfreie Klassifikation dieser Teststichprobe ist kein Beleg "
                            "für perfekte Generalisierung; das empirische Bootstrap-Intervall "
                            "ist am Rand degeneriert.")
            findings.append(finding)
    if not manifest.empty:
        one = manifest[manifest.cohort == "one_bit"]
        if len(one):
            means = one.groupby("bits").final_accuracy.mean()
            if {"0", "1"}.issubset(means.index):
                findings.append(f"**Endleistung der Ein-Bit-Kohorte:** Die mittlere C-Accuracy liegt "
                                f"hier bei {100*one.final_accuracy.mean():.2f} %. Die beiden Historien unterscheiden "
                                f"sich um {100*(means['1']-means['0']):+.3f} Prozentpunkte "
                                "(Bit 1 minus Bit 0; gesamte Kohorte). Gleiche Endleistung ist auch "
                                "hier nicht durch einen Äquivalenztest belegt.")
    sections = [
        "# Experiment 4 · Trajektorien- und Ereignisdekodierung",
        "Ausgeführt im separaten Ordner `08_Trajectory_Decoding`. Die Analyse prüft, welche früheren "
        "PCA-Koordinaten und A/B-Reihenfolgen aus finalen Netzwerkgewichten auf bisher ungesehenen "
        "Initialisierungsseeds vorhersagbar sind. Die Ergebnisdateien enthalten tatsächlich berechnete "
        "Messwerte; die Originalreproduktion wird als unveränderte Eingabe verwendet.",
        "## Primäre Ergebnisse",
        "\n\n".join(findings) if findings else "Kontrollierte Ergebnisdateien liegen für diese Berichterstellung noch nicht vor.",
        "## Datengrundlage und vorab festgelegtes Vorgehen",
        "Die Exploration verwendet 30 vorhandene Läufe mit 21 Checkpoints aus drei Datensätzen. "
        "Sie dient ausschließlich der Entwicklung und Kontrolle der Pipeline; diese kleine, heterogene "
        "Kohorte erlaubt keinen belastbaren Nachweis allgemeiner Historienrekonstruktion.",
        "Die kontrollierten Kohorten verwenden MNIST und dieselbe Architektur 784–64–10 mit ReLU "
        "und eingefrorenen Null-Biases. Aufgaben: A = y, B = (y+1) mod 10 und C = (y+3) mod 10. "
        "Die Ein-Bit-Kohorte vergleicht A→B→C mit B→A→C. Die Vier-Bit-Kohorte enthält vier "
        "aufeinanderfolgende AB/BA-Paare und anschließend C. Pro unabhängiger Initialisierung "
        "werden zwei komplementäre Historien mit identischen Bildern und Batchfolgen trainiert. "
        "Jedes A/B-Fenster umfasst eine Epoche; C umfasst fünf Epochen. SGD: Lernrate 0,1, "
        "Momentum 0,9, Batchgröße 256. Momentum wird an jedem Phasenübergang zurückgesetzt.",
        "Beide kontrollierten Trainingsdesigns und ihre Seed-Listen wurden vor der Auswertung "
        "kontrollierter Ergebnisse festgelegt. Die Ein-Bit-Kohorte wurde zuerst analysiert; "
        "die Erzeugung der Vier-Bit-Kohorte hing nicht von einem positiven Ein-Bit-Ergebnis ab. "
        "Das kontrollierte Protokoll wurde nicht anhand kontrollierter Testergebnisse angepasst.",
    ]
    if not manifest.empty:
        counts = manifest.groupby("cohort").agg(runs=("run_id", "size"), seeds=("seed", "nunique")).reset_index()
        sections.append(_md_table(counts, ["cohort", "seeds", "runs"], ["Kohorte", "Unabhängige Seeds", "Läufe"]))
    pca_rows = []
    for summary_path in sorted((root / "analysis").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        for dim, fraction in summary.get("target_pca_explained_variance", {}).items():
            pca_rows.append({"cohort": summary_path.parent.name, "pca_dim": int(dim), "explained": fraction})
    sections += [
        "Beide Historien eines Seeds bleiben im selben Trainings-, Validierungs- oder Testsplit. "
        "Ausrichtungsreferenz, Skalierung und PCA werden ausschließlich aus Trainingsdaten bestimmt. "
        "Die Hidden-Unit-Permutation wird anhand kanonischer Finalgewichte ermittelt und für sämtliche "
        "früheren Checkpoints desselben Laufs unverändert verwendet. Die Hauptprobe erhält ausschließlich "
        "finale Gewichte. Frühere Checkpoints definieren nur die Zielkoordinaten.",
        "Die primäre Trajektorienanalyse verwendet acht PCA-Dimensionen und Ridge. "
        "Absolute Positionen und Verschiebungen z(t)−z(T) werden getrennt beurteilt. "
        "Der bekannte Endpunkt T ist kein Rekonstruktionsziel. Trainingsmittelwert und Persistenz "
        "z(t)≈z(T) dienen als Baselines. Zusätzlich verwendet die Verschiebungsanalyse "
        "mean_train[z(t)]−z(T): Sie kennt nur den mittleren früheren Zustand und den ohnehin "
        "bekannten Endpunkt. Dieser Kontrollprädiktor wurde nach der Exploration und vor "
        "kontrollierter Auswertung ergänzt. Seine MSE entspricht exakt der absoluten "
        "Trainingsmittelwertbaseline. Unsicherheiten beruhen auf vollständigen Seed-Blöcken.",
        "Die Regressionsinferenz der kontrollierten Kohorten verwendet 999 Permutationen vollständiger Zieltrajektorien "
        "zwischen Test-Seed-Blöcken bei eingefrorener Probe; sie prüft die Zuordnung der "
        "Vorhersagen zu den individuellen Testverläufen, bedingt auf die trainierte Probe. "
        "Sie ist kein erneutes Training auf zufälligen Zielzuordnungen. Holm korrigiert "
        "die zusätzlichen primären Ridge-Zeitpunkte innerhalb jeder kontrollierten Kohorte. "
        "Die heterogene explorative Kohorte erhält keine inferenziellen Permutations-p-Werte. "
        "Die 95%-Intervalle verwenden 1.000 Seed-Block-Bootstrap-Stichproben aus dem Testsplit.",
        "## Frühere Trajektorienpositionen",
        "![Trajektoriengüte nach zeitlichem Abstand](trajectory_R2_vs_lag.png)",
        "Startzustand t₀, primäre Gewichtskonfiguration W0+W1 mit acht PCA-Dimensionen:",
        _md_table(initial, ["cohort", "target", "r2", "r2_ci_low", "r2_ci_high", "mse", "mae", "p_value", "p_holm"],
                  ["Kohorte", "Ziel", "R²", "CI unten", "CI oben", "MSE", "MAE", "p", "p Holm"],
                  {"r2": 5, "mse": 6, "mae": 5}),
        "R² ist relativ zur Streuung der jeweiligen Testzielkoordinaten berechnet. "
        "Ein negatives R² bedeutet eine schlechtere Vorhersage als der nachträglich bekannte "
        "Testmittelwert. Die praktisch nutzbare Mittelwertbaseline wird dagegen ausschließlich "
        "aus Trainingsdaten berechnet. Ein positiver Score allein ersetzt daher nicht den Vergleich "
        "mit Trainingsmittelwert und Persistenz. Auch Verschiebungs-R² ist kein unabhängiger Beleg "
        "für genaue Startrekonstruktion: Das subtrahierte z(T) ist bekannt und kann die Varianz "
        "von z(t)−z(T) dominieren, wenn frühe Koordinaten zwischen Runs wenig streuen. "
        "Deshalb sind absolute Koordinaten, MSE und beide Baselines gemeinsam zu lesen.",
        "![Beispielpfade aus zurückgehaltenen Läufen](trajectory_examples.png)",
        "Die Beispielauswahl folgt der gespeicherten Testreihenfolge und wird nicht nach "
        "Vorhersagegüte getroffen. Die Darstellung zeigt nur zwei Koordinaten; der bekannte "
        "Endpunkt wird ausdrücklich markiert und nicht als erfolgreiche Rekonstruktion gezählt.",
    ]
    if pca_rows:
        sections += ["Anteil der Trainings-Trajektorienvarianz, den die jeweiligen PCA-Zielräume erklären:",
                     _md_table(pd.DataFrame(pca_rows), ["cohort", "pca_dim", "explained"],
                               ["Kohorte", "Dimensionen", "Erklärte Varianz"]),
                     "Diese Varianzanteile beziehen sich auf die nach Layer-RMS skalierten und ausgerichteten "
                     "Trainingsgewichte über alle Checkpoints. Eine gute Vorhersage des PCA-Raums "
                     "belegt keine entsprechend gute Rekonstruktion sämtlicher originaler Gewichte."]
    displacement_start = primary[(primary["update"] == 0) & (primary.target == "displacement") &
                                 primary.model.isin(["ridge", "mean_early_minus_final"])]
    if not displacement_start.empty:
        sections += ["Startverschiebung im Vergleich zur Kontrolle mit bekanntem Endpunkt:",
                     _md_table(displacement_start, ["cohort", "model", "r2", "mse", "mae"],
                               ["Kohorte", "Prädiktor", "R²", "MSE", "MAE"],
                               {"r2": 6, "mse": 6, "mae": 5})]
    if events is not None and not events.empty:
        main_events = events[events.features == "both"]
        if "model" in main_events:
            main_events = main_events[main_events.model.isin(["logistic", "linear", "ridge_classifier"])]
        main_events = main_events.sort_values(["cohort", "bit"])
        sections += [
            "## Zeitlich lokalisierte Aufgabenreihenfolge",
            "![Dekodierung früherer Aufgabenpaare](event_decoding_vs_age.png)",
            _md_table(main_events, ["cohort", "bit", "age_updates", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "auc", "p_value", "p_holm"],
                      ["Kohorte", "Bit (0-basiert)", "Alter in Updates", "Accuracy", "CI unten", "CI oben", "AUC", "p", "p Holm"],
                      {"accuracy": 4, "auc": 4}),
            "Jedes Bit ist innerhalb eines Seed-Paars exakt balanciert; das Zufallsniveau beträgt 50 %. "
            "Die vier Bitpositionen gehören zu denselben Läufen und werden nicht als unabhängige "
            "Stichproben behandelt. Die Blockpermutation tauscht ganze Historienvektoren innerhalb "
            "eines Seeds. Sie prüft historische Information insgesamt; sie identifiziert keinen "
            "isolierten kausalen Effekt eines einzelnen Bits bei konstant gehaltenen übrigen Bits.",
            "Die lineare Ereignisprobe ist ein L2-regularisierter Least-Squares-Klassifikator (Ridge). "
            "Die Regularisierungsstärke wird auf Validierungsseeds gewählt. Für jede der 999 "
            "Blockpermutationen wird diese Auswahl wiederholt; die p-Wert-Auflösung beträgt 1/1000 = 0,001. "
            "Die früheste Bitposition ist der vorab bestimmte primäre Ereignisendpunkt. "
            "Holm korrigiert gemeinsam alle linearen Feature×Bit-Endpunkte einer Kohorte: "
            "sechs Tests bei einem Bit, 24 bei vier Bits. Alternative Features und MLP "
            "dienen der Robustheitsprüfung.",
        ]
        multi_profile = main_events[main_events.cohort == "multi_bit"].sort_values("bit")
        if len(multi_profile) == 4:
            profile_values = ", ".join(f"{100*value:.2f} %" for value in multi_profile.accuracy)
            corrected_values = ", ".join(fmt(value) for value in multi_profile.p_holm)
            sections.append("Von der frühesten zur spätesten Bitposition betragen die Accuracies "
                            f"{profile_values}; die zugehörigen Holm-p-Werte sind {corrected_values}. "
                            "Das ist kein gewöhnliches Vergessensprofil mit schwächerer Dekodierbarkeit "
                            "älterer Ereignisse: Hier ist das früheste Paar am besten dekodierbar. "
                            "Zusammen mit dem gescheiterten C-Lernen und der fehlenden Gleichheit "
                            "der Endfunktion verbietet dies eine einfache Interpretation als "
                            "unverzerrtes zeitliches Gedächtnisprofil.")
        interval_path = root / "provenance/multi_bit_feasibility_final_pair_success_intervals.csv"
        if interval_path.exists():
            intervals = pd.read_csv(interval_path, dtype={"bit": str})
            sections += [
                "**Grenzfall fehlerfreier Testklassifikation:** Wenn alle Testvorhersagen richtig sind, "
                "erzeugt der empirische Seed-Bootstrap erneut nur richtige Vorhersagen und liefert das "
                "degenerierte Intervall [1; 1]. Es ist keine Garantie perfekter Populationsleistung. "
                "Als Ergänzung folgen exakte zweiseitige 95%-Clopper-Pearson-Intervalle für eine andere "
                "Zielgröße: die Wahrscheinlichkeit, bei einem neuen Seed **beide Historienarme** "
                "richtig zu klassifizieren, bedingt auf die bereits trainierte Probe. "
                "Diese Intervalle sind ausdrücklich keine exakten Intervalle der Accuracy pro einzelnem Lauf. "
                "Bei joint_all_bits müssen sämtliche Bits in beiden Historien richtig sein.",
                _md_table(intervals,
                          ["cohort", "bit", "independent_seed_pairs", "fully_correct_pairs", "pair_success_rate", "pair_success_ci_low", "pair_success_ci_high"],
                          ["Kohorte", "Bit/Ziel", "Seed-Paare", "Vollständig richtige Paare", "Paar-Erfolgsrate", "CI unten", "CI oben"],
                          {"pair_success_rate": 4, "pair_success_ci_low": 4, "pair_success_ci_high": 4})
            ]
        else:
            interval_path = root / "provenance/one_bit_boundary_interval.json"
            if interval_path.exists():
                interval = json.loads(interval_path.read_text())
                sections.append("**Grenzfall fehlerfreier Testklassifikation:** Das empirische Bootstrap-CI "
                                "[1; 1] ist bei vollständig richtigen Testvorhersagen degeneriert und "
                                "belegt keine perfekte Populationsleistung. Für die Ein-Bit-Probe sind "
                                f"{interval['fully_correct_blocks']} von {interval['independent_test_seed_blocks']} "
                                "unabhängigen Seed-Paaren in beiden Armen vollständig richtig. "
                                "Das ergänzende exakte 95%-Clopper-Pearson-Intervall für die Wahrscheinlichkeit, "
                                "bei einem neuen Seed beide Arme richtig zu klassifizieren, beträgt "
                                f"[{fmt(interval['lower'], 4)}; {fmt(interval['upper'], 4)}]. "
                                "Es gilt bedingt auf die trainierte Probe und ist kein exaktes Intervall "
                                "für die Accuracy pro einzelnem Lauf.")
    if feasibility is not None:
        full = final_feasibility_path.exists()
        n = feasibility.get("runs", feasibility.get("snapshot_runs"))
        acc = feasibility["C_accuracy"]
        inactivity = feasibility.get("final_sample_dead_units", feasibility.get("sample_dead_units", {}))
        pre_inactivity = feasibility.get("before_C_sample_dead_units", {})
        sections += [
            "## Nachträgliche Prüfung der C-Trainingsmachbarkeit",
            (f"Die abschließende Diagnose umfasst die **vollständige Vier-Bit-Kohorte mit {n} Läufen**."
             if full else f"Der vorläufige diagnostische Snapshot umfasst lediglich {n} bereits abgeschlossene Läufe; er ist keine vollständige Kohortenauswertung."),
            f"C-Accuracy: Mittel {100*acc['mean']:.2f} %, Minimum {100*acc['min']:.2f} %, "
            f"Maximum {100*acc['max']:.2f} %. "
            "Die schwache und stark unterschiedliche Endleistung verhindert die beabsichtigte saubere "
            "Interpretation als Reihenfolgegedächtnis bei vergleichbarer erfolgreicher Endaufgabe.",
            "Diese Diagnose wurde nach Sichtung der schwachen C-Leistung ergänzt. Sie verändert weder "
            "Curriculum, Probe-Features, Splits noch Regularisierung und entfernt keine Läufe. "
            "Die Aktivierungsprüfung verwendet fest die ersten 1.024 offiziellen MNIST-Testbilder "
            "mit der ursprünglichen Normalisierung. Eine Einheit heißt hier nur dann "
            "stichprobeninaktiv, wenn ihre ReLU-Aktivierung auf jedem dieser Bilder exakt null ist. "
            "Das ist kein Beweis global inaktiver Neuronen und kein kausaler Nachweis der Ursache "
            "der schwachen Endleistung.",
            f"Am Ende sind auf diesem festen Bildsatz im Mittel {fmt(inactivity.get('mean'), 2)} "
            "der 64 Hidden Units inaktiv."
            + (f" Unmittelbar vor C sind es {fmt(pre_inactivity.get('mean'), 2)}."
               if pre_inactivity else ""),
            f"[Diagnostikprotokoll]({feasibility_path.relative_to(root)}) · "
            f"[Werte je Lauf]({feasibility_path.with_suffix('.csv').relative_to(root)})"
        ]
        if full:
            counts = feasibility.get("posthoc_descriptive_accuracy_counts", {})
            if counts:
                sections.append("Nachträglich gewählte, rein deskriptive Schwellen: "
                                + "; ".join(f"{count}/{n} Läufe mit C-Accuracy unter {100*float(key.replace('below_', '')):g} %"
                                            for key, count in counts.items()) + ". "
                                "Diese Schwellen sind keine vorab festgelegten Ausschlussregeln.")
            gap = feasibility.get("paired_accuracy_absolute_gap", {})
            if "mean" in gap:
                sections.append("Der absolute C-Accuracy-Abstand zwischen den beiden Historien desselben "
                                f"Seeds beträgt im Mittel {100*gap['mean']:.2f} Prozentpunkte "
                                f"(Median {100*gap['median']:.2f}; Maximum {100*gap['max']:.2f}).")
    if not manifest.empty:
        perf = manifest.groupby("cohort").agg(accuracy_mean=("final_accuracy", "mean"), accuracy_sd=("final_accuracy", "std"),
                                               accuracy_min=("final_accuracy", "min"), accuracy_max=("final_accuracy", "max"),
                                               loss_mean=("final_loss", "mean")).reset_index()
        sections += ["## Finale Leistung und Kontrollen",
                     _md_table(perf, ["cohort", "accuracy_mean", "accuracy_sd", "accuracy_min", "accuracy_max", "loss_mean"],
                               ["Kohorte", "C-Accuracy Mittel", "SD", "Minimum", "Maximum", "C-Loss Mittel"]),
                     "Ein gemeinsames C-Curriculum garantiert keine exakt gleiche Endleistung. "
                     "Leistungs- und Norm-Probes prüfen daher, ob die Aufgabenhistorie bereits in "
                     "einfachen finalen Diagnostika erkennbar ist. Solche Unterschiede begrenzen "
                     "eine Interpretation als ausschließlich verborgene Gewichtsspur."]
        if events is not None and not events.empty:
            controls = events[events.features.isin(["diagnostics", "performance", "norms"]) &
                              (events.model == "ridge_classifier")].sort_values(["cohort", "features", "bit"])
            if not controls.empty:
                sections.append(_md_table(controls, ["cohort", "features", "bit", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "auc", "p_value", "p_holm"],
                                          ["Kohorte", "Kontrolle", "Bit", "Accuracy", "CI unten", "CI oben", "AUC", "p", "p Holm"]))
        perf_rows = []
        paired_path = root / "final_performance_controls.csv"
        for cohort, group in (manifest.groupby("cohort") if not paired_path.exists() else []):
            if "bits" not in group:
                continue
            width = max(len(str(value)) for value in group.bits)
            for bit in range(width):
                for label in (0, 1):
                    selected = group[group.bits.map(lambda value: int(str(value)[bit]) == label)]
                    perf_rows.append({"cohort": cohort, "bit": bit, "label": label,
                                      "n": len(selected), "accuracy": selected.final_accuracy.mean(),
                                      "loss": selected.final_loss.mean()})
        if perf_rows:
            sections += ["Finale C-Leistung nach dem jeweiligen historischen Bit (alle Läufe; deskriptiv):",
                         _md_table(pd.DataFrame(perf_rows), ["cohort", "bit", "label", "n", "accuracy", "loss"],
                                   ["Kohorte", "Bit", "Label", "Läufe", "C-Accuracy", "C-Loss"])]
        if paired_path.exists():
            paired = pd.read_csv(paired_path)
            sections.append("Die folgenden Leistungsdifferenzen sind innerhalb identischer Initialisierungsseeds "
                            "gepaart. Δ = Leistung bei Bit 1 minus Leistung bei Bit 0; die Intervalle "
                            "verwenden 1.000 Seed-Block-Bootstraps. Accuracy ist als Anteil angegeben "
                            "(Δ = 0,001 entspricht 0,1 Prozentpunkten). Es wurde kein Äquivalenztest "
                            "mit vorab definierter Äquivalenzmarge durchgeführt. Ein Intervall, das null "
                            "enthält, belegt deshalb keine gleiche Endleistung.")
            for split, title in (("all", "Gepaarte Endleistungsunterschiede: vollständige Kohorten (deskriptiv)"),
                                 ("test", "Gepaarte Endleistungsunterschiede: zurückgehaltene Testseeds")):
                selected = paired[paired["split"] == split].sort_values(["cohort", "bit_index", "metric"])
                if not selected.empty:
                    sections += [title + ":",
                                 _md_table(selected,
                                           ["cohort", "bit_index", "metric", "seed_blocks", "bit0_mean", "bit1_mean", "paired_difference_bit1_minus_bit0", "ci_low", "ci_high"],
                                           ["Kohorte", "Bit", "Maß", "Seeds", "Bit 0 Mittel", "Bit 1 Mittel", "Δ (1−0)", "CI unten", "CI oben"],
                                           {"bit0_mean": 5, "bit1_mean": 5, "paired_difference_bit1_minus_bit0": 5, "ci_low": 5, "ci_high": 5})]
    negative = regression[regression.model.isin(["random_target_ridge", "random_target_mean"]) & (regression.layer == "both") &
                          (regression.pca_dim == 8)]
    if not negative.empty:
        sections += ["## Negativkontrollen und Robustheit",
                     "Unabhängig zufällige Zielkoordinaten prüfen die Regressionspipeline. "
                     "Zufallslabels werden für die Ereigniskontrolle auf Seed-Blockebene vertauscht. "
                     "Eine einzelne Kontrolle muss wegen Stichprobenrauschen nicht exakt R² = 0 "
                     "oder Accuracy = 0,5 ergeben; entscheidend ist ein Vergleich mit Baselines und Unsicherheit.",
                     _md_table(negative.sort_values(["cohort", "update"]),
                               ["cohort", "model", "update", "r2", "r2_ci_low", "r2_ci_high", "mse"],
                               ["Kohorte", "Modell", "Update", "R²", "CI unten", "CI oben", "MSE"],
                               {"r2": 5, "mse": 6})]
    if events is not None:
        shuffled = events[events.model == "label_shuffle"].sort_values(["cohort", "bit"])
        if not shuffled.empty:
            sections.append(_md_table(shuffled, ["cohort", "features", "bit", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "auc"],
                                      ["Kohorte", "Features", "Bit", "Zufallslabel-Accuracy", "CI unten", "CI oben", "AUC"]))
        nonlinear = events[events.model == "mlp"].sort_values(["cohort", "bit"])
        if not nonlinear.empty:
            sections += ["Sekundäre MLP-Ereignisprobes auf denselben Seed-Splits:",
                         _md_table(nonlinear, ["cohort", "bit", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "auc"],
                                   ["Kohorte", "Bit", "Accuracy", "CI unten", "CI oben", "AUC"])]
    robustness = regression[(regression["update"] == 0) & (regression.target == "absolute") &
                            regression.model.isin(["ridge", "mlp"]) & (regression.cohort != "exploratory")]
    if not robustness.empty:
        sections += ["Startzustandsrekonstruktion über alternative Layer, PCA-Dimensionen und Modelle:",
                     _md_table(robustness, ["cohort", "layer", "pca_dim", "model", "r2", "mse"],
                               ["Kohorte", "Layer", "PCA-Dimension", "Modell", "R²", "MSE"])]
    sections.append("Die sekundäre MLP-Probe verwendet ein ausschließlich auf finalen Trainingsgewichten "
                    "bestimmtes PCA-Embedding mit bis zu 32 Dimensionen (Exploration: 17 bei 18 Trainingsläufen), "
                    "32 versteckte Einheiten und maximal "
                    "300 Optimierungsschritte. Die Validierungsdaten bestimmen den gespeicherten "
                    "Schritt. Die Testdaten bestimmen weder Architektur noch Stoppzeitpunkt.")
    validation_path = root / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        sections.append("## Validierung und Schutz der Originaldaten")
        if validation.get("passed"):
            sections.append(f"Der unabhängige vollständige Archive-Audit wurde bestanden: "
                            f"{validation.get('runs_audited', '–')} Trainingsläufe wurden geprüft. "
                            "Er prüft unter anderem NPZ-/PyTorch-Hashes, gespeicherte Array-Hashes, "
                            "Gewichts- und Momentumzustände, Checkpoint-Raster, die rekonstruierte "
                            "Bildreihenfolge, gepaarte Initialisierung, die Ereignisbalance und "
                            "die Trennung der Seed-Splits.")
        else:
            sections.append("Der gespeicherte abschließende Audit ist nicht bestanden. "
                            "Offene Fehler: " + "; ".join(validation.get("errors", ["siehe validation.json"])) + ".")
        inventory = validation.get("original_inventory", {})
        if inventory:
            sections.append(f"Geschützte Originaldateien geprüft: {inventory.get('checked_files', '–')}; "
                            f"veränderte Dateien: {len(inventory.get('changed_files', []))}. "
                            + ("Alle geprüften Originaldateien sind unverändert."
                               if inventory.get("passed") else "Die Originalintegritätsprüfung ist fehlgeschlagen."))
        test_log = root / "logs/tests.txt"
        if test_log.exists():
            log = test_log.read_text()
            passed = re.findall(r"(\d+) passed", log)
            failures = re.findall(r"(\d+) failed", log)
            if passed:
                sections.append(f"Automatisierte technische Tests: {passed[-1]} bestanden" +
                                (f", {failures[-1]} fehlgeschlagen." if failures else ".") +
                                " Die Tests prüfen unter anderem train-only-Normalisierung/PCA, "
                                "Unabhängigkeit der Finalfeatures von früheren Testgewichten, "
                                "Umnummerierungsinvarianz, manuell nachgerechnete Metriken und "
                                "die Übereinstimmung gebündelter mit seriellen Permutationsberechnungen.")
    recovery_path = root / "provenance/execution_recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        sections.append("Ausführungshinweis: Der erste Trainingsprozess wurde nach "
                        f"{recovery.get('completed_manifest_runs_at_exit', '–')} gespeicherten Läufen "
                        "durch ein externes SIGTERM beendet; der Absender ist nicht bekannt. "
                        "Die Ausführung wurde mit unverändertem Code, Curriculum und Seeds fortgesetzt. "
                        "Vollständige Archive wurden anhand ihrer Fingerprints geprüft und wiederverwendet. "
                        "[Wiederaufnahmeprotokoll](provenance/execution_recovery.json).")
    sections += [
        "## Reichweite der Ergebnisse",
        "Eine erfolgreiche Probe belegt Vorhersagbarkeit bestimmter niedrigdimensionaler "
        "Zustandsmerkmale oder vergangener Aufgabenreihenfolgen unter diesem Curriculum. "
        "Sie belegt keine eindeutige Inversion von SGD und keine exakte Rekonstruktion der "
        "vollständigen hochdimensionalen Trainingshistorie. Insbesondere dürfen eine ähnliche "
        "PCA-Pfadform, die triviale Nähe zum bekannten Endpunkt und ein dekodierbares Ereignisbit "
        "nicht gleichgesetzt werden.",
        "Ein Nullresultat beschränkt die gewählten Features und Probes; es beweist nicht, dass "
        "die Gewichte keinerlei historische Information enthalten. Ergebnisse beziehen sich auf "
        "MNIST, diese Architektur, die festen Labelpermutationen und fünf finale C-Epochen. "
        "Die Test-CIs beschreiben die Unsicherheit über zurückgehaltene Seed-Blöcke bei bereits "
        "trainierter Probe; sie enthalten keine Wiederholung der gesamten Kohortenerzeugung.",
        "## Dateien und Reproduzierbarkeit",
        "- [Run-Manifest](trajectory_run_manifest.csv) und [Probe-Splits](trajectory_probe_splits.json)\n"
        "- [Alle Regressionsmesswerte](trajectory_regression_results.csv)\n"
        "- [Alle Ereignisprobes](event_decoding_results.csv)\n"
        "- [Gepaarte finale Leistungskontrollen](final_performance_controls.csv) und [Validierung](validation.json)\n"
        "- [Nachträgliche vollständige C-Machbarkeitsdiagnose](provenance/multi_bit_feasibility_final.json)\n"
        "- [Analyseplan](analysis_plan.json) und [Trainingsplan](training_plan.json)\n"
        "- [Code und Ausführung](README.md)\n"
        "- [Figurengalerie](plots/index.html)",
    ]
    (root / "trajectory_decoding_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def write_gallery(root: Path):
    figures = [("trajectory_R2_vs_lag", "Frühere Zustände aus Endgewichten"),
               ("event_decoding_vs_age", "Gedächtnisprofil vergangener Aufgabenpaare"),
               ("trajectory_examples", "Zurückgehaltene Beispieltrajektorien")]
    cards = []
    for stem, caption in figures:
        if (root / f"{stem}.png").exists():
            cards.append(f'<section><h2>{html.escape(caption)}</h2><a href="../{stem}.png">'
                         f'<img src="../{stem}.png" alt="{html.escape(caption)}"></a>'
                         f'<p><a href="../{stem}.pdf">PDF</a> · <a href="../{stem}.png">PNG</a></p></section>')
    folder = root / "plots"
    folder.mkdir(exist_ok=True)
    (folder / "index.html").write_text('''<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Trajektorien- und Ereignisdekodierung</title>
<style>body{font:16px/1.5 system-ui,sans-serif;background:#f3f5f8;color:#172b40;max-width:1250px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:20px}section{background:white;border:1px solid #dce2e9;border-radius:10px;padding:20px;margin:24px 0}img{max-width:100%;height:auto}a{color:#2166ac}</style>
<h1>Trajektorien- und Ereignisdekodierung</h1><p>Experiment 4 · separate Ausführung in 08_Trajectory_Decoding</p>
<p><a href="../trajectory_decoding_report.md">Ergebnisbericht</a> · <a href="../trajectory_regression_results.csv">Regressionsdaten</a> · <a href="../event_decoding_results.csv">Ereignisprobes</a></p>
''' + "\n".join(cards) + "</html>\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output.resolve()
    setup_style()
    regression = pd.read_csv(root / "trajectory_regression_results.csv")
    event_path = root / "event_decoding_results.csv"
    events = pd.read_csv(event_path) if event_path.exists() else None
    plot_regression(regression, root)
    if events is not None:
        plot_events(events, root)
    examples = sorted((root / "analysis").glob("*/heldout_examples.npz"))
    controlled_examples = [path for path in examples if path.parent.name in ("one_bit", "multi_bit")]
    if controlled_examples:
        examples = sorted(controlled_examples, key=lambda path: {"one_bit": 0, "multi_bit": 1}.get(path.parent.name, 2))
    if not examples:
        examples = sorted(root.glob("*_examples.npz"))
    plot_examples(examples, root)
    write_report(root, regression, events)
    write_gallery(root)
    figure_count = sum((root / f"{stem}.png").exists() for stem in ("trajectory_R2_vs_lag", "event_decoding_vs_age", "trajectory_examples"))
    print(json.dumps({"report": str(root / "trajectory_decoding_report.md"), "figures": figure_count,
                      "note": "Rendered from saved results; no model fitting."}))


if __name__ == "__main__":
    main()
