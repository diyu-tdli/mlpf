#!/usr/bin/env python3
#
# Example usage on BSC:
"""ml miniforge/24.3.0-0
CONDA_BASE="$(dirname "$(dirname "$(which conda)")")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

cd /gpfs/scratch/ehpc399/vincent/code/mlpf

python -m src.evaluation.full_evaluation \
  --arc-mlpf "/path/to/arc_eval.pt" \
  --o5-mlpf "/path/to/o5_eval.pt" \
  --datatype hitpf \
  --output-dir /path/to/output
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

from src.evaluation.plots.mass import get_response_for_event_energy
from src.evaluation.plots.resolution import calculate_response
from src.utils.inference.efficiency_calc_and_plots import calculate_eff, calculate_fakes
from src.utils.inference.event_metrics import calculate_event_mass_resolution
from src.utils.inference.pandas_helpers import concat_with_batch_fix, open_mlpf_dataframe
from src.utils.pid_conversion import our_to_pandora_mapping


matplotlib.rc("font", size=15)
plt.rc("text", usetex=False)
plt.rc("font", family="serif")


PARTICLES = [
    {
        "key": "charged_hadrons",
        "label": "Charged hadrons",
        "truth_ids": our_to_pandora_mapping[1],
        "class_id": 1,
        "fake_pid": 211,
        "resolution_truth_ids": [211, -211, 2212, -2212],
        "resolution_pid": 211,
        "xlim": (0.3, 40.0),
    },
    {
        "key": "photons",
        "label": "Photons",
        "truth_ids": our_to_pandora_mapping[3],
        "class_id": 3,
        "fake_pid": 22,
        "resolution_truth_ids": [22],
        "resolution_pid": 22,
        "xlim": (0.1, 40.0),
    },
    {
        "key": "neutral_hadrons",
        "label": "Neutral hadrons",
        "truth_ids": our_to_pandora_mapping[2],
        "class_id": 2,
        "fake_pid": 130,
        "resolution_truth_ids": [2112, 130],
        "resolution_pid": 2112,
        "xlim": (1.5, 40.0),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare full evaluation outputs for ARC and 05, with optional Pandora overlays."
    )
    parser.add_argument("--arc-mlpf", required=True, help="ARC HitPF evaluation pickle or quoted glob.")
    parser.add_argument("--o5-mlpf", required=True, help="05 HitPF evaluation pickle or quoted glob.")
    parser.add_argument("--arc-pandora", default="", help="Optional ARC Pandora pickle or quoted glob.")
    parser.add_argument("--o5-pandora", default="", help="Optional 05 Pandora pickle or quoted glob.")
    parser.add_argument(
        "--datatype",
        default="hitpf",
        help="Free-form dataset tag. If it contains 'pandora', Pandora curves are included when paths are set.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where plots and CSV summaries are written.")
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
        frames.append(frame)
    if len(frames) == 1:
        return frames[0]
    return concat_with_batch_fix(frames)


def should_include_pandora(args):
    if args.arc_pandora and args.o5_pandora:
        return True
    return "pandora" in args.datatype.lower()


def build_datasets(args):
    datasets = [
        {"label": "ARC HitPF", "frame": prepare_mlpf_frame(load_eval_dataframe(args.arc_mlpf)), "is_pandora": False},
        {"label": "05 HitPF", "frame": prepare_mlpf_frame(load_eval_dataframe(args.o5_mlpf)), "is_pandora": False},
    ]
    if should_include_pandora(args):
        if not (args.arc_pandora and args.o5_pandora):
            raise ValueError(
                "Pandora comparison requested, but --arc-pandora and --o5-pandora were not both provided."
            )
        datasets.extend(
            [
                {"label": "ARC Pandora", "frame": load_eval_dataframe(args.arc_pandora), "is_pandora": True},
                {"label": "05 Pandora", "frame": load_eval_dataframe(args.o5_pandora), "is_pandora": True},
            ]
        )
    return datasets


def prepare_mlpf_frame(frame):
    frame = frame.copy()
    if "pred_pid_matched" in frame.columns and "calibrated_E" in frame.columns:
        # Keep the same low-energy muon-to-charged-hadron correction used in the existing eval script.
        mask = (frame["pred_pid_matched"] == 4) & (frame["calibrated_E"] < 1.5)
        frame.loc[mask, "pred_pid_matched"] = 1
    return frame


def build_style(label):
    if label == "ARC HitPF":
        return {"color": "#E36414", "marker": ".", "linestyle": "None", "markersize": 7, "alpha": 0.35}
    if label == "05 HitPF":
        return {"color": "#0F4C5C", "marker": "s", "linestyle": "None", "markersize": 6, "alpha": 0.35}
    if label == "ARC Pandora":
        return {"color": "#E36414", "marker": "^", "linestyle": "None", "markersize": 7, "alpha": 0.20}
    if label == "05 Pandora":
        return {"color": "#0F4C5C", "marker": "D", "linestyle": "None", "markersize": 6, "alpha": 0.20}
    return {"color": "black", "marker": "o", "linestyle": "None", "markersize": 6, "alpha": 0.30}


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
            color=style["color"],
            alpha=style["alpha"],
            linewidth=0,
        )


def compute_particle_curves(frame, is_pandora):
    curves = {}
    for particle in PARTICLES:
        subset = frame[frame["pid"].isin(particle["truth_ids"])].copy()

        eff, energy, errors = calculate_eff(subset, log_scale=True, pandora=is_pandora)

        subset_pid = subset.copy()
        if is_pandora:
            wrong_pid = ~subset_pid["pandora_pid"].isin(particle["truth_ids"])
            subset_pid.loc[wrong_pid, "pandora_calibrated_pfo"] = np.nan
            if "pandora_calibrated_E" in subset_pid.columns:
                subset_pid.loc[wrong_pid, "pandora_calibrated_E"] = np.nan
        else:
            subset_pid.loc[subset_pid["pred_pid_matched"] != particle["class_id"], "pred_showers_E"] = np.nan
        eff_pid, energy_pid, errors_pid = calculate_eff(subset_pid, log_scale=True, pandora=is_pandora)

        fake_rate, fake_energy, fake_energy_fraction, _fake_percent_reco, fake_rate_err, fake_energy_fraction_err = (
            calculate_fakes(frame, None, log_scale=True, pandora=is_pandora, id=particle["fake_pid"])
        )

        curves[particle["key"]] = {
            "energy": np.asarray(energy),
            "eff": np.asarray(eff),
            "errors": np.asarray(errors),
            "energy_pid": np.asarray(energy_pid),
            "eff_pid": np.asarray(eff_pid),
            "errors_pid": np.asarray(errors_pid),
            "fake_energy": np.asarray(fake_energy),
            "fake_rate": np.asarray(fake_rate),
            "fake_rate_err": np.asarray(fake_rate_err),
            "fake_energy_fraction": np.asarray(fake_energy_fraction),
            "fake_energy_fraction_err": np.asarray(fake_energy_fraction_err),
            "label": particle["label"],
            "xlim": particle["xlim"],
        }
    return curves


def summarize_run(label, frame, is_pandora):
    rows = []
    for particle in PARTICLES:
        subset = frame[frame["pid"].isin(particle["truth_ids"])]
        if is_pandora:
            clustered = int(np.sum(~subset["pandora_calibrated_pfo"].isna()))
            predicted_mask = frame["pandora_pid"].isin(particle["truth_ids"])
            fake_predicted = predicted_mask & frame["pid"].isna()
            reco_energy_column = "pandora_calibrated_pfo"
        else:
            clustered = int(np.sum(~subset["pred_showers_E"].isna()))
            predicted_mask = frame["pred_pid_matched"] == particle["class_id"]
            fake_predicted = predicted_mask & frame["pid"].isna()
            reco_energy_column = "calibrated_E"
        total_predicted = int(np.sum(predicted_mask))
        fake_energy_total = float(frame.loc[predicted_mask, reco_energy_column].sum())
        fake_energy = float(frame.loc[fake_predicted, reco_energy_column].sum())
        rows.append(
            {
                "run": label,
                "category": particle["label"],
                "n_truth": len(subset),
                "n_clustered": clustered,
                "clustering_efficiency": clustered / len(subset) if len(subset) else np.nan,
                "n_predicted_as_category": total_predicted,
                "n_fake_clusters": int(np.sum(fake_predicted)),
                "fake_rate": np.sum(fake_predicted) / total_predicted if total_predicted else np.nan,
                "fake_energy_fraction": fake_energy / fake_energy_total if fake_energy_total else np.nan,
            }
        )
    return rows


def plot_metric_grid(model_curves, metric_key, error_key, ylabel, output_path, logy=False):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    for ax, particle in zip(axes, PARTICLES):
        key = particle["key"]
        all_y = []
        for label, curves in model_curves.items():
            curve = curves[key]
            x_key = "fake_energy" if "fake" in metric_key else ("energy_pid" if metric_key == "eff_pid" else "energy")
            style = build_style(label)
            x_values = np.asarray(curve[x_key], dtype=float)
            y_values = np.asarray(curve[metric_key], dtype=float)
            y_errors = np.asarray(curve[error_key], dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if not np.any(valid):
                continue
            ax.plot(
                x_values[valid],
                y_values[valid],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(ax, x_values[valid], y_values[valid], y_errors[valid] / 2.0, style)
            all_y.extend(y_values[valid].tolist())

        ax.set_title(particle["label"])
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        if metric_key.startswith("eff"):
            ax.set_ylim(0.0, 1.01)
        else:
            positive_y = np.asarray([y for y in all_y if np.isfinite(y) and y > 0.0], dtype=float)
            if len(positive_y):
                ymin = 10 ** np.floor(np.log10(np.min(positive_y)))
                ymax = 10 ** np.ceil(np.log10(np.max(positive_y) * 1.5))
            else:
                ymin, ymax = (1e-4, 1.0)
            if "fraction" in metric_key:
                ymax = max(ymax, 1e-3)
            ax.set_ylim(ymin, ymax)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_event_distributions(frame, is_pandora):
    if is_pandora:
        dic = calculate_event_mass_resolution(frame, True, ML_pid=True, fake=False)
        return np.asarray(dic["E_over_true"]), np.asarray(dic["mass_over_true_p"])
    dic = get_response_for_event_energy(None, frame, perfect_pid=False, mass_zero=False, ML_pid=True, pandora=False)
    return np.asarray(dic["energy_over_true"]), np.asarray(dic["mass_over_true_model"])


def plot_event_comparison(datasets, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    bins = np.linspace(0.5, 1.25, 160)

    for dataset in datasets:
        style = build_style(dataset["label"])
        energy_over_true, mass_over_true = compute_event_distributions(dataset["frame"], dataset["is_pandora"])

        if len(energy_over_true):
            weights = np.ones_like(energy_over_true) / len(energy_over_true)
            axes[0].hist(
                energy_over_true,
                bins=bins,
                histtype="step",
                weights=weights,
                color=style["color"],
                linewidth=2,
                label=dataset["label"],
            )
        if len(mass_over_true):
            weights = np.ones_like(mass_over_true) / len(mass_over_true)
            axes[1].hist(
                mass_over_true,
                bins=bins,
                histtype="step",
                weights=weights,
                color=style["color"],
                linewidth=2,
                label=dataset["label"],
            )

    axes[0].set_xlabel(r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$")
    axes[1].set_xlabel(r"$M_{\mathrm{reco}} / M_{\mathrm{true}}$")
    axes[0].set_ylabel("Normalized entries")
    axes[1].set_ylabel("Normalized entries")
    axes[0].set_xlim(0.5, 1.25)
    axes[1].set_xlim(0.5, 1.25)
    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_resolution_curves(frame, is_pandora):
    curves = {}
    for particle in PARTICLES:
        subset = frame[frame["pid"].isin(particle["resolution_truth_ids"])].copy()
        if is_pandora:
            subset = subset[subset["pandora_pid"] == particle["resolution_pid"]]
        else:
            subset = subset[subset["pred_pid_matched"] == particle["class_id"]]
        (
            mean,
            variance_om,
            _mean_true_rec,
            _variance_om_true_rec,
            energy_resolutions,
            _energy_resolutions_reco,
            _mean_baseline,
            _variance_om_baseline,
            _e_over_e_distr_model,
            _mean_errors,
            variance_errors,
            _mean_pxyz,
            _variance_om_pxyz,
            _masses,
            _pxyz_true,
            _phi_error,
            _sigma_phi,
            _sigma_theta,
            _variance_theta_errors,
            _distr_phi,
            _distr_E_reco,
            _m_cld,
            _var_cld,
        ) = calculate_response(
            subset,
            is_pandora,
            False,
            tracks=True,
            perfect_pid=False,
            mass_zero=False,
            ML_pid=True,
            pid=particle["resolution_pid"],
            ch=particle["resolution_pid"] == 211,
        )

        mean = np.asarray(mean)
        variance_om = np.asarray(variance_om)
        variance_errors = np.asarray(variance_errors)
        with np.errstate(divide="ignore", invalid="ignore"):
            resolution = np.where(mean != 0.0, variance_om / (2.0 * mean), np.nan)

        curves[particle["key"]] = {
            "energy": np.asarray(energy_resolutions),
            "resolution": resolution,
            "errors": variance_errors,
            "label": particle["label"],
            "xlim": particle["xlim"],
        }
    return curves


def plot_resolution_comparison(model_curves, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    for ax, particle in zip(axes, PARTICLES):
        key = particle["key"]
        all_y = []
        for label, curves in model_curves.items():
            curve = curves[key]
            style = build_style(label)
            x_values = np.asarray(curve["energy"], dtype=float)
            y_values = np.asarray(curve["resolution"], dtype=float)
            y_errors = np.asarray(curve["errors"], dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0)
            if not np.any(valid):
                continue
            ax.plot(
                x_values[valid],
                y_values[valid],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(ax, x_values[valid], y_values[valid], y_errors[valid] / 2.0, style)
            all_y.extend((y_values[valid] + y_errors[valid] / 2.0).tolist())

        ax.set_title(particle["label"])
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel(r"Energy resolution $\sigma/\mu$")
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.set_xticks([x for x in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] if particle["xlim"][0] <= x <= particle["xlim"][1]])
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        positive_y = np.asarray([y for y in all_y if np.isfinite(y) and y > 0.0], dtype=float)
        if len(positive_y):
            ymax = np.max(positive_y) * 1.2
            if ymax < 0.03:
                ymax = 0.03
            ax.set_ylim(0.0, ymax)
        else:
            ax.set_ylim(0.0, 0.1)
        ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_dir = os.path.join(args.output_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    datasets = build_datasets(args)

    model_eff_curves = {}
    resolution_curves = {}
    summary_rows = []

    for dataset in datasets:
        label = dataset["label"]
        frame = dataset["frame"]
        is_pandora = dataset["is_pandora"]
        model_eff_curves[label] = compute_particle_curves(frame, is_pandora)
        resolution_curves[label] = compute_resolution_curves(frame, is_pandora)
        summary_rows.extend(summarize_run(label, frame, is_pandora))
        for particle in PARTICLES:
            curve = model_eff_curves[label][particle["key"]]
            resolution_curve = resolution_curves[label][particle["key"]]
            print(
                f"[{label}] {particle['label']}: "
                f"eff_bins={len(curve['energy'])}, pid_eff_bins={len(curve['energy_pid'])}, "
                f"fake_bins={len(curve['fake_energy'])}, res_bins={len(resolution_curve['energy'])}"
            )

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "full_evaluation_summary.csv"), index=False)

    plot_metric_grid(
        model_eff_curves,
        "eff",
        "errors",
        "Clustering efficiency",
        os.path.join(summary_dir, "overview_Efficiency_clustering_comparison.pdf"),
        logy=False,
    )
    plot_metric_grid(
        model_eff_curves,
        "eff_pid",
        "errors_pid",
        "Efficiency with PID",
        os.path.join(summary_dir, "overview_Efficiency_pid_comparison.pdf"),
        logy=False,
    )
    plot_metric_grid(
        model_eff_curves,
        "fake_rate",
        "fake_rate_err",
        "Fake rate",
        os.path.join(summary_dir, "overview_FakeRate_comparison.pdf"),
        logy=True,
    )
    plot_metric_grid(
        model_eff_curves,
        "fake_energy_fraction",
        "fake_energy_fraction_err",
        "Fake energy %",
        os.path.join(summary_dir, "overview_FakeEnergy_comparison.pdf"),
        logy=True,
    )
    plot_event_comparison(datasets, os.path.join(summary_dir, "event_energy_mass_comparison.pdf"))
    plot_resolution_comparison(
        resolution_curves, os.path.join(summary_dir, "particle_energy_resolution_comparison.pdf")
    )

    print("Wrote full-evaluation comparison plots to", summary_dir)
    print("Summary table:", os.path.join(args.output_dir, "full_evaluation_summary.csv"))


if __name__ == "__main__":
    main()
