"""Nature-style ontology-Transformer evidence-chain figure from real source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "raw": "#D5E8F1",
    "state": "#CAEBE7",
    "relation": "#ABD7DF",
    "risk": "#337BAC",
    "structured": "#90B4CF",
    "ontology": "#337BAC",
    "positive": "#4FB1B2",
    "negative": "#90B4CF",
    "green": "#A9D9BB",
    "teal": "#4FB1B2",
    "ink": "#253238",
    "muted": "#69777D",
}
DIVERGING = LinearSegmentedColormap.from_list(
    "nature_diverging", [COLORS["risk"], "#FFFFFF", COLORS["teal"]]
)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction", xytext=(-24, 14),
        textcoords="offset points", ha="left", va="bottom", clip_on=False,
        fontsize=9, fontweight="bold",
    )


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
                text: str, facecolor: str, dashed: bool = False, fontsize: float = 6.5) -> None:
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor, edgecolor="#526168", linewidth=0.7,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#77858B") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.7, color=color, shrinkA=2, shrinkB=2))


def curved_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
                 color: str, radius: float) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color=color,
        connectionstyle=f"arc3,rad={radius}", shrinkA=2, shrinkB=3,
    ))


def clean_name(name: str) -> str:
    value = name.replace("state__", "").replace("relation_count__", "")
    pieces = []
    current = ""
    for char in value:
        if char.isupper() and current:
            pieces.append(current)
            current = char
        else:
            current += char
    if current:
        pieces.append(current)
    return " ".join(pieces).replace("_", " ")


def admission_time_labels(hours: np.ndarray | pd.Series) -> list[str]:
    """Format irregular event positions using rounded hours since admission."""
    return [f"+{int(round(float(value)))} h" for value in hours]


def draw_workflow(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "a")
    boxes = [
        (0.01, 0.20, 0.17, 0.58, "Heterogeneous EHR\nlabs · vitals · notes", COLORS["raw"]),
        (0.23, 0.20, 0.18, 0.58, "Ontology entities\n(patient-instantiated)", COLORS["state"]),
        (0.46, 0.20, 0.18, 0.58, "Ontology relations\nand trajectories", COLORS["relation"]),
        (0.69, 0.20, 0.15, 0.58, "Time-aware\nTransformer", COLORS["green"]),
        (0.89, 0.20, 0.10, 0.58, "6/12/24 h\nrisk", COLORS["structured"]),
    ]
    for x, y, w, h, text, color in boxes:
        rounded_box(ax, (x, y), w, h, text, color, fontsize=7)
    for first, second in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (first[0] + first[2], 0.49), (second[0], 0.49))
    ax.text(0.50, 0.02, "traceable provenance", ha="center", color=COLORS["muted"], fontsize=6.5)


def draw_evidence_chain(ax: plt.Axes, metadata: dict, attribution: pd.DataFrame | None = None) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "a")
    ax.set_title("Patient-level ontology-supported evidence pathway", loc="left", pad=5)
    headers = [(0.01, "Pre-cutoff evidence"), (0.265, "Ontology entity"),
               (0.505, "Semantic relation"), (0.725, "Evidence\nfusion"), (0.892, "Risk\noutput")]
    for x, text in headers:
        ax.text(x, 0.92, text, fontsize=5.8, fontweight="bold", color=COLORS["muted"], va="top")
    # Keep the pathway tied to the same case-level explanation data used by
    # the downstream attribution and occlusion panels.  The previous version
    # used a fixed illustrative pathway, which could contradict panel d.
    if attribution is not None:
        ontology = attribution[attribution["feature"].str.startswith(("state__", "relation_count__"))].copy()
        state_rank = (ontology[ontology.feature.str.startswith("state__")]
                      .groupby("feature")["integrated_gradient"]
                      .apply(lambda x: np.abs(x).sum()).sort_values(ascending=False))
        relation_rank = (ontology[ontology.feature.str.startswith("relation_count__")]
                         .groupby("feature")["integrated_gradient"]
                         .apply(lambda x: np.abs(x).sum()).sort_values(ascending=False))
        states = [clean_name(x) for x in state_rank.head(4).index]
        relations = [clean_name(x) for x in relation_rank.head(4).index]
        raw = [(f"Observed evidence\nfor {s}", False) for s in states]
        states = [s.replace(" ", "\n", 1) for s in states]
        relations = [r.replace(" ", "", 1) for r in relations]
    else:
        raw = [("Observed clinical\nevidence", False)] * 4
        states = ["Ontology\nstate"] * 4
        relations = ["ontologyRelation"] * 4
    ys = [0.76, 0.58, 0.40, 0.22]
    path_colors = [COLORS["risk"], COLORS["risk"], COLORS["teal"], COLORS["teal"]]
    for y, (raw_text, pending), state, relation, path_color in zip(ys, raw, states, relations, path_colors):
        rounded_box(ax, (0.01, y - 0.058), 0.205, 0.112, raw_text, COLORS["raw"], dashed=pending, fontsize=5.6)
        rounded_box(ax, (0.265, y - 0.058), 0.175, 0.112, state, COLORS["state"], dashed=pending, fontsize=5.6)
        rounded_box(ax, (0.505, y - 0.058), 0.175, 0.112, relation, COLORS["relation"], fontsize=4.7)
        arrow(ax, (0.215, y), (0.265, y), path_color)
        arrow(ax, (0.440, y), (0.505, y), path_color)
    rounded_box(ax, (0.735, 0.365), 0.105, 0.25,
                "Ontology\nevidence\nfusion", COLORS["green"], fontsize=5.8)
    rounded_box(ax, (0.885, 0.33), 0.105, 0.32, "", COLORS["risk"], fontsize=5.8)
    for y, path_color, radius in zip(ys, path_colors, [-0.22, -0.09, 0.09, 0.22]):
        curved_arrow(ax, (0.680, y), (0.735, 0.49), path_color, radius)
    arrow(ax, (0.840, 0.49), (0.885, 0.49), COLORS["risk"])
    ax.text(0.9375, 0.585, "6-h risk", ha="center", va="center", fontsize=5.4, color="white")
    ax.text(0.9375, 0.505, f"{metadata['ontology_probability']:.3f}", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="white")
    ax.text(0.9375, 0.395, f"structured\n{metadata['structured_probability']:.3f}",
            ha="center", va="center", fontsize=4.9, color="white")
    ax.text(0.01, 0.015, "Solid border, accepted mapping; dashed border, pending clinical review. All evidence precedes the prediction cutoff.",
            fontsize=5.5, color=COLORS["muted"])


def draw_risk_trajectory(ax: plt.Axes, prefix: pd.DataFrame, attention: pd.DataFrame) -> None:
    panel_label(ax, "b")
    x = np.arange(len(prefix))
    ax.plot(x, prefix["probability"], color=COLORS["ontology"], lw=1.8, marker="o", ms=4)
    ax.axhline(0.5, color="#9AA4A8", lw=0.7, ls="--")
    for i, value in enumerate(prefix["probability"]):
        if i % 2 == 0 or i == len(prefix) - 1:
            ax.text(i, value + 0.045, f"{value:.2f}", ha="center", fontsize=5.5)
    sizes = 25 + 180 * attention["endpoint_attention"].to_numpy()
    ax.scatter(x, np.full(len(x), 0.06), s=sizes, color="#6F76A8", alpha=0.7, edgecolor="white", lw=0.4)
    ax.text(0.03, 0.08, "endpoint attention", transform=ax.transAxes, fontsize=5.5, color="#555C83")
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.35, len(x) - 0.65)
    shown = np.unique(np.r_[np.arange(0, len(x), 3), len(x)-1])
    labels = admission_time_labels(prefix["hours_from_admission"])
    ax.set_xticks(shown, [labels[i] for i in shown], rotation=35, ha="right")
    ax.set_ylabel("Predicted risk")
    ax.set_xlabel("Time since admission")
    ax.set_title("Temporal risk accumulation", loc="left", pad=5)
    ax.grid(axis="y", color="#E4E8EA", lw=0.5)


def draw_attribution_heatmap(ax: plt.Axes, attribution: pd.DataFrame, top_n: int = 11) -> None:
    panel_label(ax, "c")
    ontology = attribution[attribution["feature"].str.startswith(("state__", "relation_count__"))].copy()
    ranking = ontology.groupby("feature")["integrated_gradient"].apply(lambda x: np.abs(x).sum()).nlargest(top_n).index
    table = ontology[ontology.feature.isin(ranking)].pivot(index="feature", columns="event_index", values="integrated_gradient")
    table = table.reindex(ranking)
    values = table.to_numpy()
    limit = float(np.quantile(np.abs(values), 0.95)) or 1.0
    image = ax.imshow(values, aspect="auto", cmap=DIVERGING, vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(table)), [clean_name(name) for name in table.index])
    event_hours = attribution.groupby("event_index")["hours_from_admission"].first().reindex(table.columns)
    ticks = np.unique(np.r_[np.arange(0, len(table.columns), 2), len(table.columns)-1])
    labels = admission_time_labels(event_hours)
    ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=35, ha="right")
    ax.set_xlabel("Time since admission")
    ax.set_title("Integrated-gradient attribution across ontology concepts", loc="left", pad=5)
    return image


def draw_occlusion(ax: plt.Axes, occlusion: pd.DataFrame) -> None:
    panel_label(ax, "d")
    order = ["Clinical states", "Ontology relations", "Structured measurements", "Time representation"]
    labels = ["Entities", "Relations", "Structured", "Time"]
    subset = occlusion.set_index("group").reindex(order).reset_index()
    values = subset["probability_drop"].to_numpy()
    colors = [COLORS["teal"], COLORS["green"], COLORS["structured"], COLORS["pale_blue"] if "pale_blue" in COLORS else COLORS["raw"]]
    ax.barh(np.arange(len(order)), values, color=colors, edgecolor="#59666B", linewidth=0.5)
    ax.axvline(0, color="#59666B", lw=0.7)
    ax.set_yticks(np.arange(len(order)), labels)
    ax.invert_yaxis()
    for i, value in enumerate(values):
        if value > 0.10:
            x_text, align = value + 0.018, "left"
        else:
            x_text, align = 0.025, "left"
        ax.text(x_text, i, f"{value:+.3f}", va="center", ha=align, fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85})
    ax.set_xlabel("Risk decrease after group occlusion")
    ax.set_title("Counterfactual explanation fidelity", loc="left", pad=5)
    ax.set_xlim(min(-0.10, values.min() - 0.04), max(1.0, values.max() + 0.08))


def draw_cohort(ax: plt.Axes, cohort: pd.DataFrame, metadata: dict) -> None:
    panel_label(ax, "e")
    subset = cohort[cohort.task == "pre_6h"]
    for label, color, text in [(0, COLORS["negative"], "Non-sepsis"), (1, COLORS["positive"], "Sepsis")]:
        group = subset[subset.label == label]
        ax.scatter(group.structured_probability, group.ontology_probability, s=8, alpha=0.35,
                   color=color, edgecolor="none", label=text)
    ax.plot([0, 1], [0, 1], color="#899499", lw=0.8, ls="--")
    selected = subset[subset.visit_no.astype(str) == str(metadata["visit_no"])]
    ax.scatter(selected.structured_probability, selected.ontology_probability, marker="*", s=90,
               color=COLORS["ink"], edgecolor="white", lw=0.7, zorder=5, label="Selected case")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Structured Transformer probability")
    ax.set_ylabel("Ontology Transformer probability")
    ax.set_title("Admission-level probability shifts (6 h)", loc="left", pad=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=5.6,
              handletextpad=0.4, columnspacing=0.9)


def load_test_auprc(results_dir: Path, variant: str) -> dict[str, float]:
    metrics = pd.read_json(results_dir / "metrics.jsonl", lines=True)
    if "split" in metrics.columns:
        test = metrics[(metrics.split == "test") & (metrics.variant == variant)]
    else:
        test = metrics[metrics.variant == variant]
    return dict(zip(test.task, test.auprc))


def draw_performance(ax: plt.Axes, structured_dir: Path, ontology_dir: Path) -> None:
    panel_label(ax, "f")
    structured = load_test_auprc(structured_dir, "structured")
    ontology = load_test_auprc(ontology_dir, "ontology")
    tasks = ["pre_6h", "pre_12h", "pre_24h"]
    x = np.arange(3)
    s = np.asarray([structured[t] for t in tasks]); o = np.asarray([ontology[t] for t in tasks])
    ax.plot(x, s, marker="o", ms=4, lw=1.5, color=COLORS["structured"], label="Structured")
    ax.plot(x, o, marker="o", ms=4, lw=1.7, color=COLORS["ontology"], label="Ontology-enhanced")
    for i, (left, right) in enumerate(zip(s, o)):
        ax.text(i, right + 0.004, f"+{right-left:.3f}", ha="center", fontsize=5.8, color=COLORS["ontology"])
    ax.set_xticks(x, ["6 h", "12 h", "24 h"])
    ax.set_ylim(min(s.min(), o.min()) - 0.015, max(s.max(), o.max()) + 0.015)
    ax.set_ylabel("Test AUPRC")
    ax.set_xlabel("Prediction horizon")
    ax.set_title("Consistent incremental predictive value", loc="left", pad=5)
    ax.text(2.02, s[-1], "Structured", color=COLORS["structured"], fontsize=5.8,
            va="center", ha="left")
    ax.text(2.02, o[-1], "Ontology-enhanced", color=COLORS["ontology"], fontsize=5.8,
            va="center", ha="left")
    ax.set_xlim(-0.1, 2.45)
    ax.grid(axis="y", color="#E4E8EA", lw=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("ProcessedData/temporal_results/transformer_explanations"))
    parser.add_argument("--structured-results", type=Path, default=Path("ProcessedData/temporal_results/explainable_structured"))
    parser.add_argument("--ontology-results", type=Path, default=Path("ProcessedData/temporal_results/explainable_transformer"))
    parser.add_argument("--output-dir", type=Path, default=Path("ProcessedData/temporal_results/transformer_explanations/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    metadata = json.loads((args.source_dir / "selected_case_metadata.json").read_text(encoding="utf-8"))
    attribution = pd.read_csv(args.source_dir / "selected_case_attributions.csv")
    attention = pd.read_csv(args.source_dir / "selected_case_attention.csv")
    prefix = pd.read_csv(args.source_dir / "selected_case_prefix_risk.csv")
    occlusion = pd.read_csv(args.source_dir / "selected_case_occlusion.csv")
    cohort = pd.read_csv(args.source_dir / "cohort_probability_comparison.csv", dtype={"visit_no": str})

    fig = plt.figure(figsize=(7.2, 7.35), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.985, hspace=0.62, wspace=1.35)
    grid = fig.add_gridspec(3, 6, height_ratios=[2.15, 2.05, 1.65])
    ax_b = fig.add_subplot(grid[0, :4]); ax_c = fig.add_subplot(grid[0, 4:])
    middle = grid[1, :].subgridspec(1, 3, width_ratios=[4.0, 1.15, 1.85], wspace=0)
    ax_d = fig.add_subplot(middle[0, 0]); ax_e = fig.add_subplot(middle[0, 2])
    ax_f = fig.add_subplot(grid[2, :3]); ax_g = fig.add_subplot(grid[2, 3:])
    draw_evidence_chain(ax_b, metadata); draw_risk_trajectory(ax_c, prefix, attention)
    image = draw_attribution_heatmap(ax_d, attribution); draw_occlusion(ax_e, occlusion)
    draw_cohort(ax_f, cohort, metadata); draw_performance(ax_g, args.structured_results, args.ontology_results)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.030, pad=0.025)
    colorbar.ax.tick_params(labelsize=6)
    stem = args.output_dir / "Figure_ontology_transformer_evidence_chain"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {args.output_dir}")


if __name__ == "__main__":
    main()
