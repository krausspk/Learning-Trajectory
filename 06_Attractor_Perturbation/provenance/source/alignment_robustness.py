#!/usr/bin/env python3
"""Summarize prespecified alternative alignments from immutable recovery CSVs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


GROUPS = ["stage", "dataset", "checkpoint", "layer", "epsilon", "momentum"]
METRICS = ["ratio_git", "ratio_pif", "ratio_alpha_matching", "alpha_change"]
BOOTSTRAP_SEED = 620260118


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_endpoints(path, stage):
    frame = pd.read_csv(path)
    frame = frame[frame.phase.isin(["pilot", "expanded"]) & (frame.epsilon > 0)].copy()
    frame["stage"] = stage
    assert frame["order"].eq("same").all()
    keys = GROUPS + ["job_id", "seed", "direction"]
    columns = ["distance", "pif_distance", "alpha_distance", "alpha", "state_available", "baseline_available"]
    initial = frame.loc[frame["update"] == 0, keys + columns]
    final = frame.loc[frame["update"] == 2350, keys + columns]
    assert len(initial) == len(final) == frame.job_id.nunique(), f"Incomplete scheduled endpoints: {path}"
    paired = initial.merge(final, on=keys, suffixes=("_initial", "_final"), validate="one_to_one")
    for output, source in [("ratio_git", "distance"), ("ratio_pif", "pif_distance"), ("ratio_alpha_matching", "alpha_distance")]:
        denominator = paired[source + "_initial"].to_numpy(dtype=float)
        numerator = paired[source + "_final"].to_numpy(dtype=float)
        paired[output] = np.divide(numerator, denominator, out=np.full(len(paired), np.nan),
                                   where=np.isfinite(denominator) & (denominator > 1e-12))
    paired["alpha_change"] = paired.alpha_final - paired.alpha_initial
    paired["complete_pair"] = (paired.state_available_initial & paired.state_available_final
                               & paired.baseline_available_initial & paired.baseline_available_final)
    paired.loc[~paired.complete_pair, METRICS] = np.nan
    paired["source_csv"] = str(path)
    return paired


def seed_bootstrap(frame):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for group, values in frame.groupby(GROUPS, sort=True, dropna=False):
        seed_means = values.groupby("seed")[METRICS].agg(lambda column: column.mean(skipna=False))
        assert len(seed_means) == 10
        indices = rng.integers(0, len(seed_means), size=(10000, len(seed_means)))
        row = dict(zip(GROUPS, group))
        row.update(n_seeds=len(seed_means), n_runs=len(values), incomplete_pairs=int((~values.complete_pair).sum()))
        for metric in METRICS:
            numbers = seed_means[metric].to_numpy(dtype=float)
            row[metric + "_finite_seeds"] = int(np.isfinite(numbers).sum())
            if not np.isfinite(numbers).all():
                row.update({metric + suffix: np.nan for suffix in ("_mean", "_low", "_high")})
                continue
            draws = numbers[indices].mean(axis=1)
            row.update({metric + "_mean": float(numbers.mean()),
                        metric + "_low": float(np.quantile(draws, .025)),
                        metric + "_high": float(np.quantile(draws, .975))})
        rows.append(row)
    return pd.DataFrame(rows)


def category_counts(frame, metric, threshold):
    valid = frame[metric + "_low"].notna() & frame[metric + "_high"].notna()
    below = valid & (frame[metric + "_high"] < threshold)
    above = valid & (frame[metric + "_low"] > threshold)
    return dict(cells=len(frame), below=int(below.sum()), above=int(above.sum()),
                overlaps=int((valid & ~below & ~above).sum()), undefined=int((~valid).sum()))


def run(root):
    root = Path(root).resolve()
    sources = [(root / "recovery_metrics.csv", "pilot")]
    sources.extend((path, "expanded") for path in sorted((root / "expanded").glob("*/recovery_metrics.csv")))
    assert len(sources) == 10, "Expected pilot and all nine dataset/checkpoint cells"
    hashes_before = {str(path.relative_to(root)): digest(path) for path, _ in sources}
    runs = pd.concat([read_endpoints(path, stage) for path, stage in sources], ignore_index=True)
    assert not runs.duplicated(["stage", "job_id"]).any()
    assert runs.groupby("stage").size().to_dict() == {"expanded": 2160, "pilot": 300}
    runs = runs.sort_values(GROUPS + ["seed", "direction"])
    summary = seed_bootstrap(runs)
    assert summary.groupby("stage").size().to_dict() == {"expanded": 216, "pilot": 6}
    runs.to_csv(root / "alignment_robustness.csv", index=False)
    summary.to_csv(root / "alignment_robustness_summary.csv", index=False)

    rows = []
    counts = {}
    labels = {"ratio_git": "Git Re-Basin", "ratio_pif": "PIF/Hungarian",
              "ratio_alpha_matching": "Alpha-Matching"}
    for stage, group in summary.groupby("stage", sort=False):
        counts[stage] = {}
        for metric, label in labels.items():
            values = category_counts(group, metric, 1.)
            counts[stage][metric] = values
            rows.append(f"| {'Pilot' if stage == 'pilot' else 'Erweiterung'} | {label} | {values['cells']} | {values['below']} | {values['above']} | {values['overlaps']} | {values['undefined']} |")
        counts[stage]["alpha_change"] = category_counts(group, "alpha_change", 0.)
    git_contract = summary.ratio_git_high < 1
    pif_contract = summary.ratio_pif_high < 1
    alpha_contract = summary.ratio_alpha_matching_high < 1
    disagreements = dict(
        git_contraction_cells=int(git_contract.sum()),
        all_three_contraction_cells=int((git_contract & pif_contract & alpha_contract).sum()),
        pif_contraction_without_git=int((pif_contract & ~git_contract).sum()),
        alpha_matching_contraction_without_git=int((alpha_contract & ~git_contract).sum()),
        git_contraction_not_confirmed_pif=int((git_contract & ~pif_contract).sum()),
        git_contraction_not_confirmed_alpha=int((git_contract & ~alpha_contract).sum()),
    )
    alpha_rows = []
    for stage in ("pilot", "expanded"):
        v = counts[stage]["alpha_change"]
        alpha_rows.append(f"| {'Pilot' if stage == 'pilot' else 'Erweiterung'} | {v['below']} | {v['above']} | {v['overlaps']} | {v['undefined']} |")
    overall = {metric: category_counts(summary, metric, 1.) for metric in labels}
    headline = (f"Keine der drei Methoden ergibt eine Zelle mit einem vollständig unter 1 liegenden Verhältnisintervall. "
                f"Git Re-Basin und PIF zeigen in jeweils {overall['ratio_git']['above']} von {len(summary)} Zellen größere End- als Anfangsabstände; "
                f"Alpha-Matching bestätigt dies in {overall['ratio_alpha_matching']['above']} Zellen.")
    exceptions = summary[(summary.ratio_alpha_matching_low <= 1) & (summary.ratio_alpha_matching_high >= 1)]
    exception_text = " ".join(
        f"Das Alpha-Matching-Intervall für {row.dataset}, E{row.checkpoint}, Schicht {row.layer}, ε={row.epsilon:g}, "
        f"Momentum {row.momentum} umfasst 1: Mittelwert {row.ratio_alpha_matching_mean:.3f}, "
        f"95%-KI [{row.ratio_alpha_matching_low:.3f}; {row.ratio_alpha_matching_high:.3f}]."
        for row in exceptions.itertuples())
    text = "\n".join([
        "# Sekundäre Robustheit gegenüber der Alignment-Methode", "",
        headline, "",
        "Die zusätzlichen Matching-Methoden waren vorab vorgesehen. Diese gemeinsame Zusammenfassung wurde nach Vorliegen der Ergebnisse erstellt; sie verändert weder die primäre Analyse noch deren Kriterien.", "",
        "Ausgewertet werden 300 gestörte Pilotläufe in sechs Zellen und 2.160 gestörte Erweiterungsläufe in 216 Zellen. Jede Zelle umfasst zehn unabhängige Ausgangsnetze. Im Pilot werden fünf Richtungen zuerst innerhalb eines Ausgangsnetzes gemittelt, in der Erweiterung liegt eine Richtung pro Ausgangsnetz vor. Baselines und Kontrollen mit anderer Batchreihenfolge sind hier ausgeschlossen.", "",
        "Für jede Matching-Methode gilt r = Endabstand / Abstand unmittelbar nach der Störung. Die Abstände sind kanonische Frobeniusnormen über beide Gewichtsmatrizen. Die Tabellen zählen Zellen, deren punktweises 95%-Seed-Bootstrapintervall vollständig unter beziehungsweise über 1 liegt (10.000 Resamples). Fehlende Werte werden innerhalb eines Seeds nicht ausgelassen.", "",
        "| Stufe | Matching | Zellen | KI vollständig < 1 | KI vollständig > 1 | KI umfasst 1 | Undefiniert |",
        "|---|---|---:|---:|---:|---:|---:|", *rows, "",
        exception_text, "",
        "Git Re-Basin minimiert bei dieser Architektur den euklidischen Abstand der kanonischen Gewichte exakt über die gemeinsame Neuronenpermutation. PIF/Hungarian optimiert die Zuordnung anhand komprimierter Neuronenmerkmale; es minimiert den vollständigen Gewichtsabstand nicht exakt. Auch das Alpha-Matching optimiert ein anderes Zuordnungskriterium. Unterschiedliche Anfangs- und Endzuordnungen können die relativen Abstände deutlich verändern. Alternative Kontraktionsbefunde sind deshalb keine unabhängigen Replikationen derselben Quotientenmetrik.", "",
        "Zusätzlich wird die Änderung des Alpha-Vorzeichenmaßes ausgewertet: ΔAlpha = Alpha am Ende − Alpha unmittelbar nach der Störung. Ein Alpha-Verhältnis wird nicht verwendet. Alpha gewichtet Vorzeichenabweichungen mit den Gewichtsbeträgen; es ist keine euklidische Distanz. Negative Änderungen bedeuten weniger gewichtete Vorzeichenabweichung, nicht automatisch eine strukturelle Rückkehr.", "",
        "| Stufe | ΔAlpha-KI vollständig < 0 | ΔAlpha-KI vollständig > 0 | KI umfasst 0 | Undefiniert |",
        "|---|---:|---:|---:|---:|", *alpha_rows, "",
        "Die Intervalle sind explorativ und nicht für multiple Vergleiche korrigiert. Ihre Grenzen sind keine simultanen Aussagen über alle Zellen. Eine Übereinstimmung mehrerer Matchings beweist weder eine Attraktorstruktur noch eine Basin-Grenze; ein richtungsabhängiger Befund bleibt insbesondere in der Erweiterung möglich.", "",
        "[Einzelwerte](alignment_robustness.csv) · [Seed-Mittelwerte und Intervalle](alignment_robustness_summary.csv) · [Reproduktionsskript](code/alignment_robustness.py)", "",
    ])
    (root / "alignment_robustness.md").write_text(text)
    hashes_after = {str(path.relative_to(root)): digest(path) for path, _ in sources}
    assert hashes_before == hashes_after
    receipt = dict(passed=True, created_at=datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
                   seed=BOOTSTRAP_SEED, bootstrap_resamples=10000, run_count=len(runs), cell_count=len(summary),
                   source_sha256=hashes_before, source_inputs_unchanged=True, script_sha256=digest(__file__),
                   method_counts=counts, contraction_disagreements=disagreements)
    (root / "validation" / "alignment_robustness.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "run_count": len(runs), "cell_count": len(summary),
                      "method_counts": counts, "contraction_disagreements": disagreements}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
