"""Nature-style population figure for ontology-Transformer patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

HORIZONS = ["6 h", "12 h", "24 h"]
TASK_TO_HORIZON = {"pre_6h": "6 h", "pre_12h": "12 h", "pre_24h": "24 h"}
COLORS = {
    "pale_blue": "#D5E8F1", "pale_cyan": "#ABD7DF", "mint": "#CAEBE7",
    "green": "#A9D9BB", "steel": "#90B4CF", "blue": "#337BAC",
    "teal": "#4FB1B2", "ink": "#263238", "muted": "#6F7E86", "grid": "#E5ECEF",
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nature_blue", ["#F7FBFC", COLORS["pale_blue"], COLORS["pale_cyan"], COLORS["teal"], COLORS["blue"]]
)


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def label(ax, value: str) -> None:
    ax.annotate(
        value, xy=(0, 1), xycoords="axes fraction", xytext=(-24, 14),
        textcoords="offset points", ha="left", va="bottom", clip_on=False,
        fontweight="bold", fontsize=9,
    )


def clean(name: str) -> str:
    text = name.replace("state__", "").replace("relation_count__", "")
    output = ""
    for char in text:
        if char.isupper() and output and not output.endswith(" "):
            output += " "
        output += char
    output = output.replace("_", " ")
    abbreviations = {
        "Renal Replacement Therapy": "Renal replacement", "Vasopressor Support": "Vasopressor support",
        "Cardiovascular Dysfunction": "CV dysfunction", "Microbiology Observation": "Microbiology evidence",
        "Mechanical Ventilation": "Mechanical ventilation", "Pathogen Detection": "Pathogen detected",
        "Abdominal Infection": "Abdominal infection", "Elevated Lactate": "Elevated lactate",
        "Worsening Trend": "Worsening trend",
    }
    return abbreviations.get(output, output)


def finish_axis(ax, axis="y") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], lw=0.55)
    ax.set_axisbelow(True)


def panel_transitions(ax, transitions: pd.DataFrame) -> list[Patch]:
    label(ax, "a")
    categories = ["Stable TP", "FN corrected by ontology", "Persistent FN", "New FN"]
    colors = [COLORS["steel"], COLORS["teal"], COLORS["pale_cyan"], COLORS["green"]]
    pivot = transitions[transitions.group.isin(categories)].pivot(
        index="horizon", columns="group", values="n").reindex(HORIZONS)
    bottom = np.zeros(3)
    for category, color in zip(categories, colors):
        values = pivot[category].to_numpy()
        ax.bar(np.arange(3), values, bottom=bottom, color=color, width=0.67,
               edgecolor="white", linewidth=0.55)
        if category == "FN corrected by ontology":
            for x, value, base in zip(np.arange(3), values, bottom):
                ax.text(x, base + value / 2, str(int(value)), ha="center", va="center",
                        fontsize=6.2, color="white", fontweight="bold")
        bottom += values
    ax.set_xticks(np.arange(3), HORIZONS); ax.set_ylabel("Sepsis admissions, n")
    ax.set_title("Error transitions across horizons", loc="left", pad=6); finish_axis(ax)
    return [Patch(facecolor=c, label=k) for k, c in zip(categories, colors)]


def panel_modules(ax, modules: pd.DataFrame) -> list[Patch]:
    label(ax, "b")
    selected = ["Clinical states", "Ontology relations"]
    colors = [COLORS["blue"], COLORS["teal"]]
    x = np.arange(3); width = 0.30
    for offset, (module, color) in enumerate(zip(selected, colors)):
        values = modules[modules.module == module].set_index("horizon").reindex(HORIZONS).auprc_drop.to_numpy()
        bars = ax.bar(x + (offset - 0.5) * width, values, width=width, color=color,
                      edgecolor="white", linewidth=0.45)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.0011, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=5.1, color=COLORS["muted"])
    ax.axhline(0, color=COLORS["muted"], lw=0.7)
    ax.set_xticks(x, HORIZONS); ax.set_ylabel("Test AUPRC decrease")
    ax.set_ylim(0, modules[modules.module.isin(selected)].auprc_drop.max() * 1.24)
    ax.set_title("Counterfactual module importance", loc="left", pad=6); finish_axis(ax)
    display = ["Ontology entities", "Semantic relations"]
    return [Patch(facecolor=c, label=k) for k, c in zip(display, colors)]


def panel_ig_heatmap(ax, ig: pd.DataFrame, top_n=9):
    label(ax, "c")
    data = ig[ig.stratum == "Sepsis"].copy()
    ranking = data.groupby("feature").mean_absolute_attribution.mean().nlargest(top_n).index
    table = data[data.feature.isin(ranking)].pivot(
        index="feature", columns="horizon", values="mean_absolute_attribution").reindex(index=ranking, columns=HORIZONS)
    values = table.to_numpy()
    image = ax.imshow(values, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=np.quantile(values, 0.97))
    ax.set_yticks(np.arange(len(table)), [clean(value) for value in table.index]); ax.set_xticks(np.arange(3), HORIZONS)
    ax.set_title("Recurrent ontology attributions", loc="left", pad=6)
    return image


def panel_forest(ax, data: pd.DataFrame, top_n=7) -> None:
    label(ax, "d")
    if data.empty:
        ax.text(0.5, 0.5, "Not estimable\n(no persistent FN contrast)", ha="center", va="center", fontsize=7, color=COLORS["muted"])
        ax.set_axis_off()
        return
    selected = data[(data.task == "pre_6h") & (data.lower > 0)].nlargest(top_n, "log2_odds_ratio").sort_values("log2_odds_ratio")
    y = np.arange(len(selected))
    ax.errorbar(selected.log2_odds_ratio, y,
                xerr=[selected.log2_odds_ratio - selected.lower, selected.upper - selected.log2_odds_ratio],
                fmt="o", ms=3.8, color=COLORS["blue"], ecolor=COLORS["steel"], elinewidth=0.9, capsize=2)
    ax.axvline(0, color=COLORS["muted"], lw=0.7, ls="--")
    ax.set_yticks(y, [clean(value) for value in selected.feature]); ax.set_xlabel("log$_2$ odds ratio (95% CI)")
    ax.set_title("Corrected false-negative enrichment (6 h)", loc="left", pad=6); finish_axis(ax, "x")


def panel_gain(ax, patients: pd.DataFrame) -> None:
    label(ax, "e")
    groups = ["Stable TP", "FN corrected by ontology", "Persistent FN", "New FN"]
    palette = [COLORS["steel"], COLORS["teal"], COLORS["pale_cyan"], COLORS["green"]]
    rng = np.random.default_rng(2026); subset = patients[patients.classification_group.isin(groups)]; values = []
    for position, (group, color) in enumerate(zip(groups, palette)):
        vector = subset[subset.classification_group == group].probability_gain.to_numpy(); values.append(vector)
        ax.scatter(np.full(len(vector), position) + rng.normal(0, 0.045, len(vector)), vector,
                   s=3.2, alpha=0.17, color=color, edgecolor="none", rasterized=True)
    violin = ax.violinplot(values, positions=np.arange(4), widths=0.72, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], palette):
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(0.72)
    violin["cmedians"].set_color(COLORS["ink"]); violin["cmedians"].set_linewidth(0.8)
    ax.axhline(0, color=COLORS["muted"], lw=0.7, ls="--")
    ax.set_xticks(np.arange(4), ["Stable\nTP", "Corrected\nFN", "Persistent\nFN", "New\nFN"])
    ax.set_ylabel("Ontology − structured probability"); ax.set_title("Probability gain by error transition", loc="left", pad=6)
    finish_axis(ax)


def panel_stability(ax, corrected: pd.DataFrame, top_n=8):
    label(ax, "f")
    if corrected.empty:
        ax.text(0.5, 0.5, "Not estimable\n(no persistent FN contrast)", ha="center", va="center", fontsize=7, color=COLORS["muted"])
        ax.set_axis_off()
        return None
    data = corrected.copy(); data["horizon"] = data.task.map(TASK_TO_HORIZON)
    ranking = data.groupby("feature").log2_odds_ratio.mean().nlargest(top_n).index
    table = data[data.feature.isin(ranking)].pivot(index="feature", columns="horizon", values="log2_odds_ratio").reindex(index=ranking, columns=HORIZONS)
    vmax = max(1.0, float(np.nanquantile(table.to_numpy(), 0.97)))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(table)), [clean(v) for v in table.index]); ax.set_xticks(np.arange(3), HORIZONS)
    ax.set_title("Corrected-FN enrichment is horizon-stable", loc="left", pad=6)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("ProcessedData/temporal_results/cohort_ontology_explanations"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(); output = args.output_dir or args.source_dir / "figures"; output.mkdir(parents=True, exist_ok=True)
    configure()
    modules = pd.read_csv(args.source_dir / "cohort_module_occlusion.csv")
    ig = pd.read_csv(args.source_dir / "cohort_integrated_gradients.csv")
    transitions = pd.read_csv(args.source_dir / "classification_transition_counts.csv")
    patients = pd.read_csv(args.source_dir / "patient_probability_transitions.csv")
    corrected = pd.read_csv(args.source_dir / "corrected_fn_odds_ratios.csv")

    fig, axes = plt.subplots(3, 2, figsize=(7.35, 7.75))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.065, top=0.895, hspace=0.70, wspace=0.62)
    transition_handles = panel_transitions(axes[0, 0], transitions); module_handles = panel_modules(axes[0, 1], modules)
    heat_a = panel_ig_heatmap(axes[1, 0], ig); panel_forest(axes[1, 1], corrected)
    panel_gain(axes[2, 0], patients); heat_b = panel_stability(axes[2, 1], corrected)
    fig.legend(handles=transition_handles, loc="upper left", bbox_to_anchor=(0.16, 0.985), ncol=2,
               fontsize=5.8, columnspacing=1.1, handlelength=1.4)
    fig.legend(handles=module_handles, loc="upper left", bbox_to_anchor=(0.59, 0.985), ncol=2,
               fontsize=5.6, columnspacing=1.0, handlelength=1.25)
    for ax, image in [(axes[1, 0], heat_a), (axes[2, 1], heat_b)]:
        if image is not None:
            cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.16, aspect=25)
            cbar.ax.tick_params(labelsize=5.5, length=2); cbar.outline.set_linewidth(0.5)
    stem = output / "Figure_population_ontology_patterns"
    fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600); fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig); print(f"Population figure written to {output}")


if __name__ == "__main__":
    main()
