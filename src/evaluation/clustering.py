#!/usr/bin/env python3
#
# Example usage on BSC:
"""    ml miniforge/24.3.0-0
cd /gpfs/scratch/ehpc399/vincent/code/mlpf

PYTHONPATH=/gpfs/scratch/ehpc399/vincent/pydeps \
/gpfs/scratch/ehpc399/jp/envs/mlpf/bin/python -m src.evaluation.clustering \
  --mlpf \
    "/gpfs/scratch/ehpc399/vincent/models/arc_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_arc_*.pkl0_0_None.pt" \
    "/gpfs/scratch/ehpc399/vincent/models/05_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_05_*.pkl0_0_None.pt" \
  --labels "ARC HitPF" "05 HitPF" \
  --pandora \
    "/gpfs/scratch/ehpc399/vincent/models/arc_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_arc_*_pandora.pt" \
    "/gpfs/scratch/ehpc399/vincent/models/05_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_05_*_pandora.pt" \
  --pandora-labels "ARC Pandora" "05 Pandora" \
  --output-dir /gpfs/scratch/ehpc399/vincent/models/clustering_compare_1M_arc_05
 """
import argparse
import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.inference.efficiency_calc_and_plots import calculate_eff, calculate_fakes
from src.utils.inference.pandas_helpers import concat_with_batch_fix, open_mlpf_dataframe
from src.utils.pid_conversion import our_to_pandora_mapping


matplotlib.rc("font", size=15)
plt.rc("text", usetex=False)
plt.rc("font", family="serif")


PARTICLES = [
    {
        "key": "charged_hadrons",
        "label": "Charged hadrons",
        "ids": our_to_pandora_mapping[1],
        "pid": 211,
        "xlim": (0.3, 40.0),
        "legend_loc_eff": "lower right",
        "legend_loc_fake": "upper right",
        "legend_loc_fake_energy": "upper right",
    },
    {
        "key": "photons",
        "label": "Photons",
        "ids": our_to_pandora_mapping[3],
        "pid": 22,
        "xlim": (0.1, 40.0),
        "legend_loc_eff": "lower right",
        "legend_loc_fake": "upper right",
        "legend_loc_fake_energy": "upper right",
    },
    {
        "key": "neutral_hadrons",
        "label": "Neutral hadrons",
        "ids": our_to_pandora_mapping[2],
        "pid": 130,
        "xlim": (1.5, 40.0),
        "legend_loc_eff": "lower right",
        "legend_loc_fake": "lower left",
        "legend_loc_fake_energy": "lower left",
    },
]

GEOMETRY_COLORS = {
    "ARC": "#E36414",
    "05": "#0F4C5C",
}

PLOT_COLORS = {
    "ARC HitPF": "#E36414",
    "ARC Pandora": "#C7522A",
    "05 HitPF": "#0F4C5C",
    "05 Pandora": "#4F84C4",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare clustering-only performance for one or more MLPF evaluation outputs."
    )
    parser.add_argument(
        "--mlpf",
        nargs="+",
        required=True,
        help="MLPF evaluation pickle(s) or quoted glob(s), one entry per run to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Human-readable labels matching the --mlpf inputs, e.g. ARC 05.",
    )
    parser.add_argument(
        "--pandora",
        nargs="*",
        default=[],
        help="Optional Pandora evaluation pickle(s) or quoted glob(s) to overlay.",
    )
    parser.add_argument(
        "--pandora-labels",
        nargs="*",
        default=[],
        help="Labels for --pandora inputs. Defaults to Pandora, Pandora 2, ...",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary plots and CSV files will be written.",
    )
    return parser.parse_args()


def resolve_inputs(pattern):
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches
    if os.path.exists(pattern):
        return [pattern]
    raise FileNotFoundError(f"No files matched input '{pattern}'")


def load_eval_dataframe(pattern):
    frames = []
    for path in resolve_inputs(pattern):
        frame, _ = open_mlpf_dataframe(path, False, False)
        if "pandora_calibrated_pfo" in frame.columns and "pandora_calibrated_E" not in frame.columns:
            frame = frame.copy()
            frame["pandora_calibrated_E"] = frame["pandora_calibrated_pfo"]
        frames.append(frame)
    if len(frames) == 1:
        return frames[0]
    return concat_with_batch_fix(frames)


def normalize_comparison_label(label, default_method="HitPF"):
    cleaned = str(label).strip()
    lowered = cleaned.lower()
    if "arc" in lowered:
        geometry = "ARC"
    elif "05" in lowered or "o5" in lowered:
        geometry = "05"
    else:
        geometry = ""

    method = "Pandora" if "pandora" in lowered else default_method
    if geometry:
        return f"{geometry} {method}"
    return cleaned


def comparison_sort_key(label):
    normalized = normalize_comparison_label(label)
    lowered = normalized.lower()
    geometry_rank = 0 if "arc" in lowered else 1 if "05" in lowered else 2
    method_rank = 0 if "hitpf" in lowered else 1 if "pandora" in lowered else 2
    return (geometry_rank, method_rank, normalized)


def ordered_curve_items(curves_by_label):
    return sorted(curves_by_label.items(), key=lambda item: comparison_sort_key(item[0]))


def build_style(label):
    normalized = normalize_comparison_label(label)
    lowered = normalized.lower()
    color = PLOT_COLORS.get(normalized)
    if color is None:
        if "arc" in lowered:
            color = GEOMETRY_COLORS["ARC"]
        elif "05" in lowered:
            color = GEOMETRY_COLORS["05"]
        else:
            color = "black"

    is_pandora = "pandora" in lowered
    return {
        "color": color,
        "marker": "^" if is_pandora else "o",
        "linestyle": "--" if is_pandora else "-",
        "markersize": 6 if is_pandora else 7,
        "facecolor": color,
        "alpha": 0.10 if is_pandora else 0.14,
    }


def _centers_to_bins(centers):
    full_bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.2))
    full_centers = 0.5 * (full_bins[:-1] + full_bins[1:])
    edges = []
    for center in centers:
        idx = int(np.argmin(np.abs(full_centers - center)))
        edges.append(full_bins[idx])
    last_idx = int(np.argmin(np.abs(full_centers - centers[-1])))
    edges.append(full_bins[last_idx + 1])
    return np.asarray(edges)


def _draw_error_band(ax, x_centers, y_values, y_errors, style):
    if len(x_centers) == 0:
        return
    bins = _centers_to_bins(np.asarray(x_centers))
    y_values = np.asarray(y_values)
    y_errors = np.asarray(y_errors)
    for idx, value in enumerate(y_values):
        lower = max(0.0, value - y_errors[idx])
        height = 2.0 * y_errors[idx]
        ax.fill_between(
            [bins[idx], bins[idx + 1]],
            [lower, lower],
            [lower + height, lower + height],
            color=style["facecolor"],
            alpha=style["alpha"],
            linewidth=0,
        )


def is_pandora_frame(frame):
    return "pandora_pid" in frame.columns and "pandora_calibrated_pfo" in frame.columns


def compute_particle_curves(frame):
    pandora = is_pandora_frame(frame)
    has_pid = "pandora_pid" in frame.columns if pandora else "pred_pid_matched" in frame.columns
    curves = {}
    for particle in PARTICLES:
        mask = frame["pid"].isin(particle["ids"])
        eff, energy, errors = calculate_eff(frame[mask], log_scale=True, pandora=pandora)
        if has_pid:
            (
                fake_rate,
                fake_energy,
                fake_energy_fraction,
                _fake_percent_reco,
                fake_rate_err,
                fake_energy_fraction_err,
            ) = calculate_fakes(frame, None, log_scale=True, pandora=pandora, id=particle["pid"])
        else:
            fake_rate = []
            fake_energy = []
            fake_energy_fraction = []
            fake_rate_err = []
            fake_energy_fraction_err = []
        curves[particle["key"]] = {
            "energy": np.asarray(energy),
            "eff": np.asarray(eff),
            "errors": np.asarray(errors),
            "fake_energy": np.asarray(fake_energy),
            "fake_rate": np.asarray(fake_rate),
            "fake_rate_err": np.asarray(fake_rate_err),
            "fake_energy_fraction": np.asarray(fake_energy_fraction),
            "fake_energy_fraction_err": np.asarray(fake_energy_fraction_err),
            "label": particle["label"],
            "xlim": particle["xlim"],
            "legend_loc_eff": particle["legend_loc_eff"],
            "legend_loc_fake": particle["legend_loc_fake"],
            "legend_loc_fake_energy": particle["legend_loc_fake_energy"],
        }
    return curves


def summarize_run(label, frame):
    pandora = is_pandora_frame(frame)
    has_pid = "pandora_pid" in frame.columns if pandora else "pred_pid_matched" in frame.columns
    rows = []
    for particle in PARTICLES:
        subset = frame[frame["pid"].isin(particle["ids"])]
        reco_energy_col = "pandora_calibrated_pfo" if pandora else "pred_showers_E"
        fake_energy_col = "pandora_calibrated_pfo" if pandora else "calibrated_E"
        clustered = int(np.sum(~subset[reco_energy_col].isna()))
        if has_pid:
            if pandora:
                predicted_as_particle = frame["pandora_pid"].isin(particle["ids"])
            else:
                predicted_as_particle = frame["pred_pid_matched"] == particle["pid"]
            fake_predicted = predicted_as_particle & frame["pid"].isna()
            total_predicted_as_particle = int(np.sum(predicted_as_particle))
            fake_energy_total = float(frame.loc[predicted_as_particle, fake_energy_col].sum())
            fake_energy = float(frame.loc[fake_predicted, fake_energy_col].sum())
            n_fake_clusters = int(np.sum(fake_predicted))
            fake_rate = (
                np.sum(fake_predicted) / total_predicted_as_particle
                if total_predicted_as_particle
                else np.nan
            )
            fake_energy_fraction = (
                fake_energy / fake_energy_total if fake_energy_total else np.nan
            )
        else:
            total_predicted_as_particle = np.nan
            n_fake_clusters = np.nan
            fake_rate = np.nan
            fake_energy_fraction = np.nan
        rows.append(
            {
                "run": label,
                "category": particle["label"],
                "n_truth": len(subset),
                "n_clustered": clustered,
                "clustering_efficiency": clustered / len(subset) if len(subset) else np.nan,
                "pid_metrics_available": has_pid,
                "n_predicted_as_category": total_predicted_as_particle,
                "n_fake_clusters": n_fake_clusters,
                "fake_rate": fake_rate,
                "fake_energy_fraction": fake_energy_fraction,
            }
        )
    return rows


def plot_efficiency_comparison(curves_by_label, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    for ax, particle in zip(axes, PARTICLES):
        key = particle["key"]

        curve = None
        for label, curves in ordered_curve_items(curves_by_label):
            curve = curves[key]
            style = build_style(label)
            ax.plot(
                curve["energy"],
                curve["eff"],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(ax, curve["energy"], curve["eff"], curve["errors"] / 2.0, style)

        ax.set_title(curve["label"])
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Clustering efficiency")
        ax.set_ylim(0.0, 1.01)
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.tick_params(axis="both", which="minor", labelsize=17)
        ax.legend(
            fontsize=17,
            title_fontsize=17,
            title="Geometry / reco",
            loc=curve["legend_loc_eff"],
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fake_rate_comparison(curves_by_label, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    for ax, particle in zip(axes, PARTICLES):
        key = particle["key"]

        curve = None
        for label, curves in ordered_curve_items(curves_by_label):
            curve = curves[key]
            style = build_style(label)
            ax.plot(
                curve["fake_energy"],
                curve["fake_rate"],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(
                ax,
                curve["fake_energy"],
                curve["fake_rate"],
                curve["fake_rate_err"] / 2.0,
                style,
            )

        ax.set_title(curve["label"])
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Fake rate")
        ax.set_ylim(1e-4, 1.0)
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.tick_params(axis="both", which="minor", labelsize=17)
        ax.legend(
            fontsize=17,
            title_fontsize=17,
            title="Geometry / reco",
            loc=curve["legend_loc_fake"],
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fake_energy_comparison(curves_by_label, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    for ax, particle in zip(axes, PARTICLES):
        key = particle["key"]

        curve = None
        for label, curves in ordered_curve_items(curves_by_label):
            curve = curves[key]
            style = build_style(label)
            ax.plot(
                curve["fake_energy"],
                curve["fake_energy_fraction"],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(
                ax,
                curve["fake_energy"],
                curve["fake_energy_fraction"],
                curve["fake_energy_fraction_err"],
                style,
            )

        ax.set_title(curve["label"])
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Fake Energy %")
        ax.set_ylim(1e-4, 5.0)
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.tick_params(axis="both", which="minor", labelsize=17)
        ax.legend(
            fontsize=17,
            title_fontsize=17,
            title="Geometry / reco",
            loc=curve["legend_loc_fake_energy"],
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if len(args.mlpf) != len(args.labels):
        raise ValueError("--mlpf and --labels must have the same length")
    if args.pandora_labels and len(args.pandora) != len(args.pandora_labels):
        raise ValueError("--pandora and --pandora-labels must have the same length")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_dir = os.path.join(args.output_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    comparison_frames = {
        normalize_comparison_label(label, default_method="HitPF"): load_eval_dataframe(pattern)
        for label, pattern in zip(args.labels, args.mlpf)
    }
    pandora_labels = (
        args.pandora_labels
        if args.pandora_labels
        else ["Pandora" if idx == 0 else f"Pandora {idx + 1}" for idx in range(len(args.pandora))]
    )
    comparison_frames.update(
        {
            normalize_comparison_label(label, default_method="Pandora"): load_eval_dataframe(pattern)
            for label, pattern in zip(pandora_labels, args.pandora)
        }
    )
    pid_metrics_available = all(
        is_pandora_frame(frame) or "pred_pid_matched" in frame.columns
        for frame in comparison_frames.values()
    )

    comparison_curves = {
        label: compute_particle_curves(frame)
        for label, frame in comparison_frames.items()
    }

    summary_rows = []
    for label, frame in ordered_curve_items(comparison_frames):
        summary_rows.extend(summarize_run(label, frame))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "clustering_summary.csv"), index=False)

    plot_efficiency_comparison(
        comparison_curves,
        os.path.join(summary_dir, "overview_Efficiency_clustering_comparison.pdf"),
    )
    if pid_metrics_available:
        plot_fake_rate_comparison(
            comparison_curves,
            os.path.join(summary_dir, "overview_FakeRate_clustering_comparison.pdf"),
        )
        plot_fake_energy_comparison(
            comparison_curves,
            os.path.join(summary_dir, "overview_FakeEnergy_clustering_comparison.pdf"),
        )
    else:
        print("Skipping fake-rate plots because PID columns are not available in all inputs.")

    print("Wrote clustering-only comparison plots to", summary_dir)
    print("Summary table:", os.path.join(args.output_dir, "clustering_summary.csv"))


if __name__ == "__main__":
    main()
