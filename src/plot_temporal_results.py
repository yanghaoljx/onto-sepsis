"""Create manuscript-ready figures from ``run_temporal_experiments.py`` output.

The plots intentionally separate the primary validation results from the
independent test confirmation.  The script writes PNG, PDF and SVG files so
that the same figure can be inspected locally and placed into a manuscript.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


TASK_ORDER = ["pre_6h", "pre_12h", "pre_24h"]
VARIANT_ORDER = ["structured", "ontology"]
MODEL_ORDER = ["logreg", "random_forest", "xgboost", "lstm", "gru", "gru_d", "transformer"]
MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lstm": "BiLSTM",
    "gru": "BiGRU",
    "gru_d": "GRU-D",
    "transformer": "Time-aware Transformer",
}
VARIANT_LABELS = {"structured": "Structured", "ontology": "Ontology-enhanced"}
TASK_LABELS = {"pre_6h": "6 h", "pre_12h": "12 h", "pre_24h": "24 h"}
VARIANT_COLORS = {"structured": "#277da1", "ontology": "#d1495b"}
HORIZON_COLORS = {"pre_6h": "#277da1", "pre_12h": "#f4a261", "pre_24h": "#d1495b"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_pipeline(output_dir: Path) -> None:
    data_root = output_dir.parents[2]
    cohort = pd.read_csv(data_root / "sepsis_cohort.csv", dtype={"visit_no": str, "patient_id": str})
    summaries = {
        h: pd.read_json(data_root / "temporal_results" / "main" / f"pre_{h}h_ontology_dataset_summary.json", typ="series")
        for h in (6, 12, 24)
    }
    raw_n = int(cohort.visit_no.nunique())
    positive_n = int((cohort.cohort == "positive").sum())
    negative_n = int((cohort.cohort == "negative").sum())
    excluded = cohort[cohort.cohort == "excluded"].exclusion_reason.value_counts()
    admission_excluded = int(excluded.get("sepsis_but_excluded_event_definition", 0))
    other_visit_excluded = int(excluded.get("patient_has_sepsis_in_another_visit", 0))

    fig = plt.figure(figsize=(7.25, 5.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.35], wspace=0.18)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    palette = {"neutral": "#D5E8F1", "entity": "#337BAC", "relation": "#4FB1B2",
               "mint": "#CAEBE7", "ink": "#263238", "muted": "#6F7E86"}

    def box(ax, x, y, w, h, title, detail, fill, edge=None):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               facecolor=fill, edgecolor=edge or fill, linewidth=0.8)
        ax.add_patch(patch)
        ax.text(x + 0.035, y + h * 0.63, title, ha="left", va="center", fontsize=7.1,
                fontweight="bold", color=palette["ink"])
        ax.text(x + 0.035, y + h * 0.30, detail, ha="left", va="center", fontsize=6.25,
                color=palette["muted"], linespacing=1.25)

    def down(ax, x, y1, y2):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.75, color=palette["muted"]))

    axes[0].text(-0.02, 1.01, "a", fontweight="bold", fontsize=9, va="bottom")
    axes[0].text(0.04, 0.965, "Local cohort construction", fontsize=8.2, fontweight="bold", color=palette["ink"])
    box(axes[0], 0.06, 0.79, 0.88, 0.12, "Source hospital records",
        f"50,161 source rows × 8 columns\n{raw_n:,} unique admissions; 45,394 patients", palette["neutral"])
    down(axes[0], 0.50, 0.79, 0.72)
    box(axes[0], 0.06, 0.57, 0.88, 0.15, "Outcome-definition screen",
        f"{positive_n:,} eligible sepsis events  |  {negative_n:,} candidate controls\nExcluded: {admission_excluded:,} event-definition; {other_visit_excluded:,} other-visit sepsis", "#ABD7DF")
    down(axes[0], 0.50, 0.57, 0.50)
    box(axes[0], 0.06, 0.34, 0.88, 0.16, "Observed pre-index ward rounds",
        "4,079 positive admissions with usable history\n40,086 eligible negative admissions", palette["mint"])
    down(axes[0], 0.50, 0.34, 0.27)
    box(axes[0], 0.06, 0.08, 0.88, 0.19, "Matched development cohort",
        "4,079 positive + 4,079 time-matched negative admissions\nPatient-level split: 70% train / 15% validation / 15% test",
        "#A9D9BB")

    axes[1].text(-0.02, 1.01, "b", fontweight="bold", fontsize=9, va="bottom")
    axes[1].text(0.035, 0.965, "Horizon-specific endpoints and model comparison", fontsize=8.2,
                 fontweight="bold", color=palette["ink"])
    for i, h in enumerate((6, 12, 24)):
        d = summaries[h]
        y = 0.78 - i * 0.21
        box(axes[1], 0.04, y, 0.92, 0.145, f"Prediction at least {h} h before sepsis",
            f"Train {int(d['n_train']):,}  |  validation {int(d['n_validation']):,}  |  test {int(d['n_test']):,}\nTest: {int(d['positive_test']):,} positive, {int(d['n_test']-d['positive_test']):,} negative",
            ["#D5E8F1", "#ABD7DF", "#CAEBE7"][i])
    box(axes[1], 0.04, 0.08, 0.43, 0.16, "Structured input", "Measurements + missingness\n+ irregular time", "#90B4CF")
    box(axes[1], 0.53, 0.08, 0.43, 0.16, "Ontology-enhanced", "Structured + entities\n+ semantic relations", "#4FB1B2")
    axes[1].text(0.50, 0.025, "7 model families; validation selects, locked test confirms",
                 ha="center", va="bottom", fontsize=6.4, color=palette["muted"])
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.96)
    save_figure(fig, output_dir, "Figure_1_experimental_pipeline")


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    frame = frame[frame["split"].isin(["validation", "test"])].copy()
    # A resumed run should not create duplicate rows.  Never let pivot_table
    # silently average structured and ontology rows together.
    frame = frame.drop_duplicates(
        subset=["task", "variant", "model", "split"], keep="last"
    )
    frame["task"] = pd.Categorical(frame["task"], TASK_ORDER, ordered=True)
    frame["variant"] = pd.Categorical(frame["variant"], VARIANT_ORDER, ordered=True)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    return frame


def heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    vmin: float,
    vmax: float,
    panel_label: str,
) -> mpl.image.AxesImage:
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [TASK_LABELS[x] for x in table.columns])
    ax.set_yticks(range(len(table.index)), [MODEL_LABELS.get(x, x) for x in table.index])
    ax.set_title(title, pad=7)
    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                midpoint = (vmin + vmax) / 2.0
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < midpoint else "black",
                    fontsize=7.2,
                )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    return image


def figure_performance_heatmaps(metrics: pd.DataFrame, output_dir: Path) -> None:
    for metric, stem in (("auprc", "Figure_2A_AUPRC"), ("auroc", "Figure_2B_AUROC")):
        values = metrics[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        vmin = math.floor((float(values.min()) - 0.01) * 100) / 100
        vmax = math.ceil((float(values.max()) + 0.01) * 100) / 100
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1), constrained_layout=True)
        images = []
        panel = 0
        for row, split in enumerate(["validation", "test"]):
            for col, variant in enumerate(VARIANT_ORDER):
                subset = metrics[(metrics["split"] == split) & (metrics["variant"] == variant)]
                table = subset.pivot(index="model", columns="task", values=metric)
                table = table.reindex(index=MODEL_ORDER, columns=TASK_ORDER)
                image = heatmap(
                    axes[row, col],
                    table,
                    f"{split.capitalize()} · {VARIANT_LABELS[variant]}",
                    vmin,
                    vmax,
                    chr(ord("a") + panel),
                )
                images.append(image)
                axes[row, col].set_xlabel("Prediction horizon")
                axes[row, col].set_ylabel("Model")
                panel += 1
        fig.colorbar(images[0], ax=axes, shrink=0.78, label=metric.upper())
        fig.suptitle(
            f"{metric.upper()} across irregular-time models and feature representations",
            y=1.02,
        )
        save_figure(fig, output_dir, stem)


def _best_model(metrics: pd.DataFrame, task: str, variant: str, split: str = "validation") -> str:
    subset = metrics[
        (metrics["task"] == task)
        & (metrics["variant"] == variant)
        & (metrics["split"] == split)
    ].sort_values("auprc", ascending=False)
    if subset.empty:
        raise ValueError(f"No metrics found for {task}/{variant}/{split}")
    return str(subset.iloc[0]["model"])


def _prediction_path(results_dir: Path, task: str, variant: str, model: str, split: str) -> Path:
    return results_dir / "predictions" / f"{task}_{variant}_{model}_{split}.csv"


def figure_curves(metrics: pd.DataFrame, results_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for col, task in enumerate(TASK_ORDER):
        for variant in VARIANT_ORDER:
            model = _best_model(metrics, task, variant)
            path = _prediction_path(results_dir, task, variant, model, "validation")
            if not path.exists():
                continue
            pred = pd.read_csv(path)
            fpr, tpr, _ = roc_curve(pred["label"], pred["probability"])
            precision, recall, _ = precision_recall_curve(pred["label"], pred["probability"])
            color = colors[VARIANT_ORDER.index(variant)]
            label = f"{VARIANT_LABELS[variant]} ({MODEL_LABELS.get(model, model)})"
            axes[0, col].plot(fpr, tpr, lw=1.7, color=color, label=label)
            axes[1, col].plot(recall, precision, lw=1.7, color=color, label=label)
        axes[0, col].plot([0, 1], [0, 1], color="#999999", ls="--", lw=0.8)
        axes[0, col].set_title(f"Validation: {TASK_LABELS[task]}")
        axes[1, col].set_title(f"Validation: {TASK_LABELS[task]}")
        axes[0, col].set_xlabel("False-positive rate")
        axes[0, col].set_ylabel("Sensitivity")
        axes[1, col].set_xlabel("Sensitivity")
        axes[1, col].set_ylabel("Precision")
        axes[0, col].set_xlim(0, 1)
        axes[0, col].set_ylim(0, 1)
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_ylim(0, 1)
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.suptitle("Receiver-operating and precision–recall curves", y=1.02)
    save_figure(fig, output_dir, "Figure_3_validation_curves")


def figure_calibration(metrics: pd.DataFrame, results_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), constrained_layout=True)
    colors = {"structured": "#277da1", "ontology": "#d1495b"}
    for ax, task in zip(axes, TASK_ORDER):
        for variant in VARIANT_ORDER:
            model = _best_model(metrics, task, variant)
            path = _prediction_path(results_dir, task, variant, model, "validation")
            if not path.exists():
                continue
            pred = pd.read_csv(path)
            fraction, mean_predicted = calibration_curve(
                pred["label"], pred["probability"], n_bins=10, strategy="quantile"
            )
            ax.plot(mean_predicted, fraction, marker="o", ms=3, lw=1.4, color=colors[variant], label=VARIANT_LABELS[variant])
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
        ax.set_title(f"{TASK_LABELS[task]}")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.suptitle("Validation calibration of the best model in each feature arm", y=1.04)
    save_figure(fig, output_dir, "Figure_4_validation_calibration")


def _delta_table(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            structured = metrics[
                (metrics.task == task)
                & (metrics.model == model)
                & (metrics.variant == "structured")
                & (metrics.split == split)
            ]
            ontology = metrics[
                (metrics.task == task)
                & (metrics.model == model)
                & (metrics.variant == "ontology")
                & (metrics.split == split)
            ]
            if structured.empty or ontology.empty:
                continue
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "delta_auprc": float(ontology.iloc[0].auprc - structured.iloc[0].auprc),
                }
            )
    return pd.DataFrame(rows)


def figure_ablation(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True, constrained_layout=True)
    offsets = np.linspace(-0.24, 0.24, len(TASK_ORDER))
    width = 0.22
    x = np.arange(len(MODEL_ORDER))
    effect_path = output_dir.parent / "ontology_paired_effects.jsonl"
    effect = pd.read_json(effect_path, lines=True) if effect_path.exists() else pd.DataFrame()
    for ax, split, panel in zip(axes, ["validation", "test"], ["a", "b"]):
        table = _delta_table(metrics, split)
        if table.empty:
            ax.axis("off")
            continue
        for offset, task in zip(offsets, TASK_ORDER):
            values = []
            lower_errors = []
            upper_errors = []
            for model in MODEL_ORDER:
                row = table[(table.model == model) & (table.task == task)]
                value = float(row.iloc[0].delta_auprc) if not row.empty else np.nan
                values.append(value)
                if not effect.empty:
                    ci = effect[
                        (effect.task == task)
                        & (effect.model == model)
                        & (effect.split == split)
                    ]
                else:
                    ci = pd.DataFrame()
                if not ci.empty and np.isfinite(value):
                    lower_errors.append(max(value - float(ci.iloc[0].delta_auprc_lower), 0.0))
                    upper_errors.append(max(float(ci.iloc[0].delta_auprc_upper) - value, 0.0))
                else:
                    lower_errors.append(0.0)
                    upper_errors.append(0.0)
            ax.bar(
                x + offset,
                values,
                width=width,
                color=HORIZON_COLORS[task],
                edgecolor="white",
                linewidth=0.35,
                label=TASK_LABELS[task],
                yerr=np.vstack([lower_errors, upper_errors]),
                error_kw={"ecolor": "#333333", "elinewidth": 0.7, "capsize": 2},
            )
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xticks(x, [MODEL_LABELS.get(model, model) for model in MODEL_ORDER], rotation=35, ha="right")
        ax.set_title(split.capitalize())
        ax.set_ylabel("Δ AUPRC\n(ontology − structured)")
        ax.text(-0.12, 1.08, panel, transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Incremental value of ontology-enhanced representations", y=1.03)
    save_figure(fig, output_dir, "Figure_5_ontology_ablation")


def figure_stratified(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "sequence_stratified_metrics.csv"
    if not path.exists():
        return
    strata = pd.read_csv(path)
    models = ["xgboost", "gru_d", "transformer"]
    model_colors = {"xgboost": "#264653", "gru_d": "#e76f51", "transformer": "#2a9d8f"}
    stratum_orders = {
        "length_bin": ["1–3", "4–8", "9–16", "17–24"],
        "time_bin": ["0–12 h", "12–24 h", "24–48 h", ">48 h"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.4), constrained_layout=True)
    for row, stratifier in enumerate(["length_bin", "time_bin"]):
        for col, task in enumerate(TASK_ORDER):
            ax = axes[row, col]
            subset = strata[
                (strata.task == task)
                & (strata.variant == "ontology")
                & (strata.model.isin(models))
                & (strata.split == "test")
                & (strata.stratifier == stratifier)
            ]
            for model in models:
                model_data = subset[subset.model == model]
                if model_data.empty:
                    continue
                model_data = model_data.copy()
                model_data["stratum"] = pd.Categorical(
                    model_data["stratum"], stratum_orders[stratifier], ordered=True
                )
                model_data = model_data.sort_values("stratum")
                ax.plot(
                    model_data["stratum"],
                    model_data["auprc"],
                    marker="o",
                    ms=3.5,
                    lw=1.5,
                    color=model_colors[model],
                    label=MODEL_LABELS[model],
                )
            ax.set_title(f"{TASK_LABELS[task]}")
            ax.set_xlabel("Sequence length bin" if row == 0 else "Endpoint time bin")
            ax.set_ylabel("Test AUPRC")
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", rotation=25)
    axes[0, 0].text(-0.18, 1.12, "a", transform=axes[0, 0].transAxes, fontsize=11, fontweight="bold")
    axes[1, 0].text(-0.18, 1.12, "b", transform=axes[1, 0].transAxes, fontsize=11, fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("Test performance across irregular-sequence strata", y=1.03)
    save_figure(fig, output_dir, "Figure_6_sequence_strata")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ProcessedData/temporal_results/main"))
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()
    configure_style()
    figure_dir = args.figure_dir or args.results_dir / "figures"
    metrics = load_metrics(args.results_dir / "metrics.jsonl")
    figure_pipeline(figure_dir)
    figure_performance_heatmaps(metrics, figure_dir)
    figure_curves(metrics, args.results_dir, figure_dir)
    figure_calibration(metrics, args.results_dir, figure_dir)
    figure_ablation(metrics, figure_dir)
    figure_stratified(args.results_dir, figure_dir)
    print(f"Figures written to {figure_dir}")


if __name__ == "__main__":
    main()
