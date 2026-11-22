"""Scientific RSA figures from the saved distance matrices and inference tables.

This module never trains networks, changes checkpoint files, or computes p-values.
Run ``make_plots(Path('01_RSA'))`` only after distance and inference jobs finish.
Every figure is exported as PNG and vector PDF into that experiment directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd


DATASETS = ("mnist", "fashion_mnist", "kmnist")
NAMES = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST"}
COLORS = {"mnist": "#4477AA", "fashion_mnist": "#CC6677", "kmnist": "#C78C19"}
PRIMARY = ("frobenius", "total", "optimal", "canonical")
ALIGNMENTS = ("optimal", "fixed_E0", "fixed_E20")
ALIGNMENT_NAMES = {"optimal": "Reoptimized at each checkpoint", "fixed_E0": "Permutation fixed at E=0", "fixed_E20": "Permutation fixed at E=20"}
ALIGNMENT_COLORS = {"optimal": "#4477AA", "fixed_E0": "#AA4499", "fixed_E20": "#228833"}
METRIC_NAMES = {"frobenius": "Frobenius", "normalized_frobenius": "Normalized Frobenius", "cosine": "Cosine dissimilarity", "alpha2": "Alpha (α=2)", "snd": "SND"}
LAYER_NAMES = {"total": "total", "W0": "W₀", "W1": "W₁", "balanced": "layer-balanced", "size_weighted": "size-weighted"}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 250, "pdf.fonttype": 42,
    "axes.formatter.useoffset": False,
}


def _table_path(out: Path, name: str) -> Path:
    for directory in (out / "results", out):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing RSA plotting table {name} in {out}/results or {out}")


def _read_table(out: Path, name: str, columns: set[str]) -> pd.DataFrame:
    path = _table_path(out, name)
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"Invalid RSA table {path}: empty={frame.empty}, missing columns={missing}")
    return frame


def _where(frame: pd.DataFrame, **criteria) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in criteria.items():
        mask &= frame[key].eq(value).to_numpy()
    result = frame.loc[mask]
    if result.empty:
        raise ValueError(f"Missing RSA plotting rows: {criteria}")
    return result


def _select(frame: pd.DataFrame, dataset: str, key=PRIMARY, checkpoint=None) -> pd.DataFrame:
    metric, layer, alignment, gauge = key
    criteria = dict(dataset=dataset, metric=metric, layer=layer, alignment=alignment, gauge=gauge)
    if checkpoint is not None:
        criteria["checkpoint"] = checkpoint
    rows = _where(frame, **criteria).sort_values("checkpoint")
    if rows.checkpoint.duplicated().any():
        raise ValueError(f"Duplicate RSA checkpoint rows: {criteria}")
    return rows


def _endpoint(frame: pd.DataFrame, dataset: str, key=PRIMARY) -> pd.Series:
    rows = _select(frame, dataset, key, 20)
    if len(rows) != 1:
        raise ValueError(f"Expected one E=20 row for {dataset}/{key}")
    return rows.iloc[0]


def _distance_data(out: Path) -> dict[str, dict]:
    result = {}
    for dataset in DATASETS:
        path = out / "data" / f"{dataset}_distances.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k] for k in ("distances", "metric_keys", "checkpoints", "run_ids", "seeds", "pairs")}
        keys = [str(value) for value in arrays["metric_keys"]]
        if len(set(keys)) != len(keys) or "|".join(PRIMARY) not in keys:
            raise ValueError(f"Missing or ambiguous primary metric key in {path}")
        cp = arrays["checkpoints"]
        if not np.array_equal(cp, np.arange(21)):
            raise ValueError(f"Unexpected checkpoint order in {path}: {cp}")
        if len(arrays["run_ids"]) != 10 or len(set(arrays["run_ids"].tolist())) != 10:
            raise ValueError(f"Expected ten distinct run identities in {path}")
        if len(arrays["seeds"]) != 10 or arrays["pairs"].shape != (45, 2):
            raise ValueError(f"Invalid seeds/pairs in {path}")
        expected_pairs = np.stack(np.triu_indices(10, 1), axis=1)
        if not np.array_equal(arrays["pairs"], expected_pairs):
            raise ValueError(f"Pair ordering differs from upper triangle in {path}")
        distances = arrays["distances"]
        if distances.shape != (len(keys), 21, 10, 10) or not np.isfinite(distances).all():
            raise ValueError(f"Invalid distance matrices in {path}")
        if not np.allclose(distances, distances.swapaxes(-1, -2), atol=1e-10, rtol=1e-10):
            raise ValueError(f"Asymmetric distances in {path}")
        if not np.allclose(np.diagonal(distances, axis1=-2, axis2=-1), 0, atol=1e-10):
            raise ValueError(f"Nonzero distance diagonal in {path}")
        matrices = distances[keys.index("|".join(PRIMARY))]
        arrays["primary_pairs"] = matrices[:, expected_pairs[:, 0], expected_pairs[:, 1]]
        arrays["source_path"] = path
        result[dataset] = arrays
    return result


def _style_axis(ax, *, panel=None, correlation=False):
    ax.grid(axis="y", color="#DEE4EA", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    if panel is not None:
        ax.text(-.08, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=13, va="bottom")
    if correlation:
        ax.axhline(0, color="#7D8793", lw=.9, ls="--", zorder=1)
        ax.set_ylim(-1.04, 1.04)
        ax.set_yticks([-1, -.5, 0, .5, 1])


def _save(fig, out: Path, name: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _footer(fig, text: str):
    fig.text(.5, .015, text, ha="center", va="bottom", fontsize=8.5, color="#475363")


def _p_text(value: float) -> str:
    if not np.isfinite(value):
        return "not estimated"
    return f"{value:.2g}" if value < .001 else f"{value:.4f}".rstrip("0").rstrip(".")


def _finite_ci(row) -> bool:
    return bool(np.isfinite([row["rho"], row["ci_low"], row["ci_high"]]).all())


def _rho_line(ax, rows: pd.DataFrame, color: str, label: str, *, linestyle="-", marker="o"):
    x = rows.checkpoint.to_numpy(float)
    y = rows.rho.to_numpy(float)
    ax.plot(x, y, color=color, label=label, lw=1.8, marker=marker, markersize=3.3, linestyle=linestyle)
    low = np.maximum(-1, rows.ci_low.to_numpy(float))
    high = np.minimum(1, rows.ci_high.to_numpy(float))
    ax.fill_between(x, low, high, color=color, alpha=.13, linewidth=0)


def _plot_scatter(out, table, data):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6))
    fig.subplots_adjust(left=.065, right=.99, top=.77, bottom=.20, wspace=.32)
    fig.suptitle("Does initial weight-space geometry persist after training?", fontsize=14, y=.98)
    for i, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        pairs = data[dataset]["primary_pairs"]
        stat = _endpoint(table, dataset)
        ax.scatter(pairs[0], pairs[20], s=32, alpha=.76, color=COLORS[dataset], edgecolors="white", linewidth=.45, zorder=3)
        ax.set(xlabel="Initial distance, D(0)", ylabel="Final distance, D(20)")
        ax.set_title(NAMES[dataset], pad=45)
        p_annotation = r"$p_{\mathrm{perm},>}$" + f" = {_p_text(stat.p_perm_greater)}"
        if "p_holm_greater" in stat and np.isfinite(stat.p_holm_greater):
            p_annotation += f"; Holm p = {_p_text(stat.p_holm_greater)}"
        ax.text(.5, 1.025, f"Spearman ρ = {stat.rho:.3f}\n" + p_annotation,
                transform=ax.transAxes, va="bottom", ha="center", fontsize=9.5)
        for axis in (ax.xaxis, ax.yaxis):
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
            axis.set_major_locator(MaxNLocator(5))
        _style_axis(ax)
        ax.text(-.08, 1.16, chr(65+i), transform=ax.transAxes, fontweight="bold", fontsize=13, va="bottom")
    _footer(fig, "Canonical total Frobenius distance (parameter units), optimal shared W₀/W₁ permutation at each checkpoint.\n"
            "10 networks / 45 dependent pair values per dataset; one-sided identity-permutation p and Holm correction across three datasets. Axes use separate scales.")
    return _save(fig, out, "rsa_initial_vs_final")


def _plot_training(out, table):
    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    fig.subplots_adjust(left=.1, right=.98, top=.86, bottom=.22)
    ax.set_title("Initial-to-current weight-space RSA", pad=12)
    for dataset in DATASETS:
        _rho_line(ax, _select(table, dataset), COLORS[dataset], NAMES[dataset])
    _style_axis(ax, correlation=True)
    ax.set(xlabel="Training checkpoint E (one step = half an epoch)", ylabel="Spearman ρ[D(0), D(E)]", xlim=(-.4, 20.4))
    ax.set_xticks([0, 1, 2, 5, 10, 15, 20])
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DEE4EA")
    _footer(fig, "Primary metric: canonical total Frobenius distance, reoptimized pair permutation. E=0 is the self-correlation sanity check.\n"
            "Shading: 95% run-level jackknife intervals (10 networks per dataset), displayed within [−1, 1].")
    return _save(fig, out, "rsa_over_training")


def _plot_distributions(out, data):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    fig.subplots_adjust(left=.07, right=.99, top=.9, bottom=.15, hspace=.58, wspace=.32)
    fig.suptitle("Initial and final within-dataset distances", fontsize=14)
    for col, dataset in enumerate(DATASETS):
        for row, checkpoint in enumerate((0, 20)):
            ax = axes[row, col]
            values = data[dataset]["primary_pairs"][checkpoint]
            counts, _, _ = ax.hist(values, bins=10, color=COLORS[dataset], alpha=.7, edgecolor="white", linewidth=.8)
            ax.set_ylim(0, float(np.max(counts)) * 1.42)
            ax.plot(values, np.zeros_like(values), "|", color="#273241", markersize=7, clip_on=False)
            mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
            cv = sd / mean if mean else np.nan
            ax.text(.97, .96, f"Mean = {mean:.4g}\nSD = {sd:.3g}\nCV = {100 * cv:.3g}%",
                    transform=ax.transAxes, va="top", ha="right", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
            ax.set(title=f"{NAMES[dataset]} · {'Initial' if checkpoint == 0 else 'Final'} (E={checkpoint})",
                   xlabel="Canonical total Frobenius distance", ylabel="Number of pairs")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            _style_axis(ax, panel=chr(65+row*3+col))
    _footer(fig, "Each panel contains 45 dependent pair values from 10 networks; distributions and spread statistics are descriptive.\n"
            "All distances are in canonical parameter units. Independent x-axis scales expose initial distance concentration; CV = SD / mean.")
    return _save(fig, out, "distance_distributions")


def _metric_order(key):
    metric, layer, alignment, gauge = key
    return (("frobenius", "normalized_frobenius", "cosine", "alpha2", "snd").index(metric)
            if metric in METRIC_NAMES else 99,
            ("total", "W0", "W1", "balanced", "size_weighted").index(layer)
            if layer in LAYER_NAMES else 99, gauge != "canonical", alignment)


def _metric_label(key):
    metric, layer, _, gauge = key
    return f"{METRIC_NAMES.get(metric, metric)} · {LAYER_NAMES.get(layer, layer)} · {gauge}"


def _forest_point(ax, y, row, color, primary=False):
    if _finite_ci(row):
        ax.plot([max(-1, row.ci_low), min(1, row.ci_high)], [y, y], color=color, lw=1.4, zorder=2)
        ax.plot(row.rho, y, "D" if primary else "o", color=color, markersize=5 if primary else 4, zorder=3)
    elif np.isfinite(row.rho):
        ax.plot(row.rho, y, "x", color=color, markersize=5, zorder=3)


def _plot_robustness(out, table):
    endpoint = table[table.checkpoint.eq(20) & ~table.alignment.isin(("fixed_E0", "fixed_E20"))]
    columns = ["metric", "layer", "alignment", "gauge"]
    keys = sorted(set(endpoint[columns].itertuples(index=False, name=None)), key=_metric_order)
    if not keys:
        raise ValueError("No endpoint robustness metrics available")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, max(6.4, len(keys)*.31+1.8)), sharey=True)
    fig.subplots_adjust(left=.325, right=.99, top=.93, bottom=.11, wspace=.12)
    fig.suptitle("Endpoint RSA across distance definitions, layers and gauges", fontsize=14, y=.985)
    for col, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        for y, key in enumerate(keys):
            stat = _endpoint(table, dataset, key)
            if key == PRIMARY:
                ax.axhspan(y-.4, y+.4, color="#EEF2F6", zorder=0)
            _forest_point(ax, y, stat, COLORS[dataset], primary=key == PRIMARY)
        ax.set_title(NAMES[dataset])
        ax.set(xlim=(-1.05, 1.05), ylim=(len(keys)-.3, -.7), xlabel="Spearman ρ[D(0), D(20)]")
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_yticks(np.arange(len(keys)), [_metric_label(k) for k in keys], fontsize=9)
        ax.axvline(0, color="#7D8793", lw=.9, ls="--")
        ax.grid(axis="x", color="#DEE4EA", lw=.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    _footer(fig, "Points and horizontal bars: dataset-specific correlations and 95% run-level jackknife intervals (displayed within [−1, 1]); diamond: primary endpoint.\n"
            "Frobenius, normalized Frobenius and cosine use shared optimal weight matching. Alpha and SND use their respective profile assignments.\n"
            "Alternative metrics define distinct geometries; these controls are exploratory. 10 networks / 45 dependent pair values per dataset.")
    return _save(fig, out, "rsa_robustness")


def _plot_alignments(out, table):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), sharey=True)
    fig.subplots_adjust(left=.065, right=.99, top=.83, bottom=.28, wspace=.18)
    fig.suptitle("Sensitivity to changing versus fixed neuron assignments", fontsize=14, y=.98)
    for i, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        for alignment, linestyle, marker in zip(ALIGNMENTS, ("-", "--", ":"), ("o", "s", "^")):
            key = ("frobenius", "total", alignment, "canonical")
            _rho_line(ax, _select(table, dataset, key), ALIGNMENT_COLORS[alignment], ALIGNMENT_NAMES[alignment], linestyle=linestyle, marker=marker)
        _style_axis(ax, panel=chr(65+i), correlation=True)
        ax.set(title=NAMES[dataset], xlabel="Training checkpoint E", xlim=(-.4, 20.4))
        ax.set_xticks([0, 2, 5, 10, 15, 20])
    axes[0].set_ylabel("Spearman ρ[D(0), D(E)]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .125), ncol=3, frameon=False)
    _footer(fig, "Canonical total Frobenius distance throughout. Each curve uses its own alignment protocol for both D(0) and D(E).\n"
            "Shading: 95% run-level jackknife intervals, displayed within [−1, 1]. Fixed E=20 assignments are a retrospective sensitivity control.")
    return _save(fig, out, "alignment_sensitivity")


def _norm_label(key):
    gauge, analysis, checkpoint, predictor = key
    if analysis == "partial_rsa":
        return f"Partial RSA, E=0 → {checkpoint} · {gauge}"
    short = {"normsum": "Initial norm sum", "normabsdiff": "Initial norm difference",
             "norm_sum": "Initial norm sum", "norm_absdiff": "Initial norm difference",
             "norm_abs_difference": "Initial norm difference"}.get(predictor, predictor)
    return f"{short} → D({checkpoint}) · {gauge}"


def _plot_norms(out):
    path = None
    for name in ("norm_controls.csv", "normcontrols.csv"):
        try:
            path = _table_path(out, name)
            break
        except FileNotFoundError:
            continue
    if path is None:
        return []
    table = pd.read_csv(path)
    required = {"dataset", "gauge", "analysis", "checkpoint", "predictor", "rho", "ci_low", "ci_high", "layer"}
    if table.empty or not required.issubset(table.columns):
        raise ValueError(f"Invalid norm-control table: {path}")
    table = _where(table, layer="total")
    columns = ["gauge", "analysis", "checkpoint", "predictor"]
    keys = sorted(set(table[columns].itertuples(index=False, name=None)), key=lambda k: (k[1] != "partial_rsa", k[0] != "canonical", k[2], k[3]))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, max(5.7, len(keys)*.38+1.8)), sharey=True)
    fig.subplots_adjust(left=.32, right=.99, top=.86, bottom=.18, wspace=.12)
    fig.suptitle("Initial norm controls for weight-space RSA", fontsize=14, y=.98)
    for dataset, ax in zip(DATASETS, axes):
        for y, key in enumerate(keys):
            rows = _where(table, dataset=dataset, **dict(zip(columns, key)))
            if len(rows) != 1:
                raise ValueError(f"Ambiguous norm control {dataset}/{key}")
            _forest_point(ax, y, rows.iloc[0], COLORS[dataset], primary=key[1] == "partial_rsa")
        ax.set(title=NAMES[dataset], xlabel="Spearman / partial rank correlation", xlim=(-1.05, 1.05), ylim=(len(keys)-.3, -.7))
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_yticks(np.arange(len(keys)), [_norm_label(k) for k in keys], fontsize=9)
        ax.axvline(0, color="#7D8793", lw=.9, ls="--")
        ax.grid(axis="x", color="#DEE4EA", lw=.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    _footer(fig, "Diamond: initial-to-final partial rank RSA after controlling initial pair norm sum and absolute norm difference.\n"
            "Bars: 95% run-level jackknife intervals, displayed within [−1, 1]. Exploratory controls with 10 networks per dataset.")
    return _save(fig, out, "norm_controls")


def make_plots(out: Path) -> list[Path]:
    """Render all RSA figures from completed analysis outputs; return saved paths."""
    out = Path(out).resolve()
    table = _read_table(out, "rsa_by_dataset.csv", {
        "dataset", "checkpoint", "metric", "layer", "alignment", "gauge", "rho",
        "p_perm_greater", "p_perm_two_sided", "jackknife_se", "ci_low", "ci_high",
        "n_runs", "n_pairs", "n_permutations",
    })
    for dataset in DATASETS:
        for alignment in ALIGNMENTS:
            rows = _select(table, dataset, ("frobenius", "total", alignment, "canonical"))
            if not np.array_equal(rows.checkpoint.to_numpy(), np.arange(21)):
                raise ValueError(f"Incomplete checkpoint series for {dataset}/{alignment}")
            if not (rows.n_runs.eq(10) & rows.n_pairs.eq(45)).all():
                raise ValueError(f"Unexpected replication counts for {dataset}/{alignment}")
    data = _distance_data(out)
    paths = []
    with plt.rc_context(STYLE):
        paths += _plot_scatter(out, table, data)
        paths += _plot_training(out, table)
        paths += _plot_distributions(out, data)
        paths += _plot_robustness(out, table)
        paths += _plot_alignments(out, table)
        paths += _plot_norms(out)
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for generated in make_plots(args.out):
        print(generated)
