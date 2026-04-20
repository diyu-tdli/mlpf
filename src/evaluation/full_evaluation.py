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
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.evaluation.plots.mass import get_response_for_event_energy
from src.evaluation.plots.resolution import calculate_response
from src.utils.inference.efficiency_calc_and_plots import calculate_eff, calculate_fakes
from src.utils.inference.event_metrics import calculate_event_mass_resolution
from src.utils.inference.pandas_helpers import concat_with_batch_fix, open_mlpf_dataframe
from src.utils.pid_conversion import our_to_pandora_mapping, pandora_to_our_mapping, pid_conversion_dict


matplotlib.rc("font", size=15)
plt.rc("text", usetex=False)
plt.rc("font", family="serif")


GEOMETRY_COLORS = {
    "ARC": "#E36414",
    "05": "#0F4C5C",
}
GEOMETRY_DISPLAY_NAMES = {
    "ARC": "o3_v01",
    "05": "o2_v05",
}

PLOT_COLORS = {
    "o3_v01 HitPF": "#E36414",
    "o3_v01 Pandora": "#C7522A",
    "o2_v05 HitPF": "#0F4C5C",
    "o2_v05 Pandora": "#4F84C4",
}
RATIO_METHOD_STYLES = {
    "HitPF": {"color": "#1D3557", "marker": "o", "linestyle": "-", "markersize": 5, "alpha": 0.15},
    "Pandora": {"color": "#9C6644", "marker": "^", "linestyle": "--", "markersize": 5, "alpha": 0.15},
}

CONFUSION_CLASS_ORDER = [1, 3, 2, 0, 4]
CONFUSION_CLASS_NAMES = {
    1: "CH",
    3: r"$\gamma$",
    2: "NH",
    0: "e",
    4: r"$\mu$",
}
CONFUSION_ENERGY_BINS = [
    (1.0, 10.0),
    (10.0, 100.0),
]
CONFUSION_DATASET_LAYOUT = [
    ("o3_v01 HitPF", (0.25, 0.25)),
    ("o2_v05 HitPF", (0.75, 0.25)),
    ("o3_v01 Pandora", (0.25, 0.75)),
    ("o2_v05 Pandora", (0.75, 0.75)),
]


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

EVENT_COMPONENTS = [
    {
        "key": "charged_hadrons",
        "label": "Charged hadrons",
        "class_id": 1,
        "output_name": "event_energy_mass_comparison_charged_hadrons.pdf",
        "xlim": (0.9, 1.05),
        "bins": np.linspace(0.9, 1.05, 260),
        "logy": True,
    },
    {
        "key": "photons",
        "label": "Photons",
        "class_id": 3,
        "output_name": "event_energy_mass_comparison_photons.pdf",
        "xlim": (0.5, 1.25),
        "bins": np.linspace(0.5, 1.25, 160),
        "logy": False,
    },
    {
        "key": "neutral_hadrons",
        "label": "Neutral hadrons",
        "class_id": 2,
        "output_name": "event_energy_mass_comparison_neutral_hadrons.pdf",
        "xlim": (0.5, 1.25),
        "bins": np.linspace(0.5, 1.25, 160),
        "logy": False,
    },
    {
        "key": "klong",
        "label": r"$K_L$",
        "truth_ids": [130],
        "output_name": "event_energy_mass_comparison_klong.pdf",
        "xlim": (0.5, 1.25),
        "bins": np.linspace(0.5, 1.25, 160),
        "logy": False,
    },
    {
        "key": "neutrons",
        "label": "Neutrons",
        "truth_ids": [2112],
        "output_name": "event_energy_mass_comparison_neutrons.pdf",
        "xlim": (0.5, 1.25),
        "bins": np.linspace(0.5, 1.25, 160),
        "logy": False,
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
        if "pandora_calibrated_pfo" in frame.columns and "pandora_calibrated_E" not in frame.columns:
            frame = frame.copy()
            frame["pandora_calibrated_E"] = frame["pandora_calibrated_pfo"]
        frames.append(frame)
    if len(frames) == 1:
        return frames[0]
    return concat_with_batch_fix(frames)


def should_include_pandora(args):
    if args.arc_pandora and args.o5_pandora:
        return True
    return "pandora" in args.datatype.lower()


def normalize_comparison_label(label, default_method="HitPF"):
    cleaned = str(label).strip()
    lowered = cleaned.lower()
    if "arc" in lowered or "o3_v01" in lowered:
        geometry = "ARC"
    elif "05" in lowered or "o5" in lowered or "o2_v05" in lowered or "02_v05" in lowered:
        geometry = "05"
    else:
        geometry = ""

    method = "Pandora" if "pandora" in lowered else default_method
    if geometry:
        return f"{GEOMETRY_DISPLAY_NAMES[geometry]} {method}"
    return cleaned


def comparison_sort_key(label):
    normalized = normalize_comparison_label(label)
    lowered = normalized.lower()
    geometry_rank = 0 if "o3_v01" in lowered or "arc" in lowered else 1 if "o2_v05" in lowered or "05" in lowered else 2
    method_rank = 0 if "hitpf" in lowered else 1 if "pandora" in lowered else 2
    return (geometry_rank, method_rank, normalized)


def ordered_curve_items(curves_by_label):
    return sorted(curves_by_label.items(), key=lambda item: comparison_sort_key(item[0]))


def geometry_key_from_label(label):
    lowered = normalize_comparison_label(label).lower()
    if "o3_v01" in lowered or "arc" in lowered:
        return "ARC"
    if "o2_v05" in lowered or "05" in lowered:
        return "05"
    return None


def method_key_from_label(label):
    lowered = normalize_comparison_label(label).lower()
    if "pandora" in lowered:
        return "Pandora"
    if "hitpf" in lowered:
        return "HitPF"
    return None


def validate_hitpf_frame(frame, label):
    required_columns = [
        "pid",
        "true_showers_E",
        "pred_showers_E",
        "pred_pid_matched",
        "calibrated_E",
        "pred_pos_matched",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required HitPF columns: {missing}")

    frame = frame.copy()
    frame.loc[frame["pred_pid_matched"] < 0, "pred_pid_matched"] = np.nan
    print(
        f"[{label}] using HitPF columns: pred_pid_matched, calibrated_E, pred_pos_matched"
    )
    return frame


def validate_pandora_frame(frame, label):
    required_columns = [
        "pid",
        "true_showers_E",
        "pred_showers_E",
        "pandora_pid",
        "pandora_calibrated_pfo",
        "pandora_calibrated_pos",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required Pandora columns: {missing}")

    frame = frame.copy()
    if "pandora_calibrated_E" not in frame.columns:
        frame["pandora_calibrated_E"] = frame["pandora_calibrated_pfo"]
    print(
        f"[{label}] using Pandora columns: pandora_pid, pandora_calibrated_pfo, pandora_calibrated_pos"
    )
    return frame


def build_datasets(args):
    arc_hitpf_label = normalize_comparison_label("ARC HitPF")
    o5_hitpf_label = normalize_comparison_label("05 HitPF")
    datasets = [
        {
            "label": arc_hitpf_label,
            "frame": validate_hitpf_frame(prepare_mlpf_frame(load_eval_dataframe(args.arc_mlpf)), arc_hitpf_label),
            "is_pandora": False,
        },
        {
            "label": o5_hitpf_label,
            "frame": validate_hitpf_frame(prepare_mlpf_frame(load_eval_dataframe(args.o5_mlpf)), o5_hitpf_label),
            "is_pandora": False,
        },
    ]
    if should_include_pandora(args):
        if not (args.arc_pandora and args.o5_pandora):
            raise ValueError(
                "Pandora comparison requested, but --arc-pandora and --o5-pandora were not both provided."
            )
        datasets.extend(
            [
                {
                    "label": normalize_comparison_label("ARC Pandora", default_method="Pandora"),
                    "frame": validate_pandora_frame(
                        load_eval_dataframe(args.arc_pandora),
                        normalize_comparison_label("ARC Pandora", default_method="Pandora"),
                    ),
                    "is_pandora": True,
                },
                {
                    "label": normalize_comparison_label("05 Pandora", default_method="Pandora"),
                    "frame": validate_pandora_frame(
                        load_eval_dataframe(args.o5_pandora),
                        normalize_comparison_label("05 Pandora", default_method="Pandora"),
                    ),
                    "is_pandora": True,
                },
            ]
        )
    return sorted(datasets, key=lambda dataset: comparison_sort_key(dataset["label"]))


def prepare_mlpf_frame(frame):
    frame = frame.copy()
    if "pred_pid_matched" in frame.columns and "calibrated_E" in frame.columns:
        # Keep the same low-energy muon-to-charged-hadron correction used in the existing eval script.
        mask = (frame["pred_pid_matched"] == 4) & (frame["calibrated_E"] < 1.5)
        frame.loc[mask, "pred_pid_matched"] = 1
    return frame


def build_style(label, confusion=False):
    normalized = normalize_comparison_label(label)
    geometry_key = geometry_key_from_label(normalized)
    method_key = method_key_from_label(normalized)
    if confusion:
        if geometry_key == "ARC":
            color = GEOMETRY_COLORS["ARC"]
        elif geometry_key == "05":
            color = GEOMETRY_COLORS["05"]
        else:
            color = "black"
    else:
        legacy_label = normalized.replace("o3_v01", "ARC").replace("o2_v05", "05")
        color = PLOT_COLORS.get(normalized) or PLOT_COLORS.get(legacy_label)
        if color is None:
            if geometry_key == "ARC":
                color = GEOMETRY_COLORS["ARC"]
            elif geometry_key == "05":
                color = GEOMETRY_COLORS["05"]
            else:
                color = "black"

    is_pandora = method_key == "Pandora"
    return {
        "color": color,
        "marker": "^" if is_pandora else "o",
        "linestyle": "--" if is_pandora else "-",
        "markersize": 6 if is_pandora else 7,
        "alpha": 0.10 if is_pandora else 0.14,
    }


def build_ratio_style(method_key):
    return RATIO_METHOD_STYLES[method_key]


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


def build_ratio_pairs(curves_by_label):
    labels_by_geometry_method = {}
    for label in curves_by_label:
        geometry_key = geometry_key_from_label(label)
        method_key = method_key_from_label(label)
        if geometry_key is None or method_key is None:
            continue
        labels_by_geometry_method[(geometry_key, method_key)] = label

    pairs = []
    for method_key in ["HitPF", "Pandora"]:
        numerator_label = labels_by_geometry_method.get(("ARC", method_key))
        denominator_label = labels_by_geometry_method.get(("05", method_key))
        if numerator_label and denominator_label:
            pairs.append((method_key, numerator_label, denominator_label))
    return pairs


def compute_ratio_curve(x_num, y_num, err_num, x_den, y_den, err_den):
    x_num = np.asarray(x_num, dtype=float)
    y_num = np.asarray(y_num, dtype=float)
    err_num = np.asarray(err_num, dtype=float)
    x_den = np.asarray(x_den, dtype=float)
    y_den = np.asarray(y_den, dtype=float)
    err_den = np.asarray(err_den, dtype=float)

    ratio_x = []
    ratio_y = []
    ratio_err = []
    for idx, x_value in enumerate(x_num):
        matches = np.where(np.isclose(x_den, x_value, rtol=1e-6, atol=1e-12))[0]
        if len(matches) == 0:
            continue
        den_idx = int(matches[0])
        num_value = y_num[idx]
        den_value = y_den[den_idx]
        num_err = err_num[idx]
        den_err = err_den[den_idx]
        if not (
            np.isfinite(num_value)
            and np.isfinite(den_value)
            and np.isfinite(num_err)
            and np.isfinite(den_err)
            and den_value != 0.0
        ):
            continue
        ratio_value = num_value / den_value
        ratio_error = np.sqrt((num_err / den_value) ** 2 + ((num_value * den_err) / (den_value ** 2)) ** 2)
        ratio_x.append(x_value)
        ratio_y.append(ratio_value)
        ratio_err.append(ratio_error)

    return np.asarray(ratio_x), np.asarray(ratio_y), np.asarray(ratio_err)


def metric_x_key(metric_key):
    if "fake" in metric_key:
        return "fake_energy"
    if metric_key == "eff_pid":
        return "energy_pid"
    return "energy"


def metric_scale(metric_key):
    if metric_key == "fake_energy_fraction":
        return 100.0
    return 1.0


def set_metric_ylim(ax, metric_key, all_y):
    if metric_key.startswith("eff"):
        ax.set_ylim(0.0, 1.01)
        return

    positive_y = np.asarray([y for y in all_y if np.isfinite(y) and y > 0.0], dtype=float)
    if len(positive_y):
        ymin = 10 ** np.floor(np.log10(np.min(positive_y)))
        ymax = 10 ** np.ceil(np.log10(np.max(positive_y) * 1.5))
    else:
        ymin, ymax = (1e-4, 1.0)
    if "fraction" in metric_key:
        ymax = max(ymax, 1e-3)
    ax.set_ylim(ymin, ymax)


def style_metric_axis(ax, particle, ylabel, metric_key, all_y, logy):
    ax.set_title(particle["label"])
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*particle["xlim"])
    ax.set_xscale("log")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    set_metric_ylim(ax, metric_key, all_y)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=15, title="Geometry / method", title_fontsize=13)


def style_ratio_axis(ax, particle, ratio_ylabel, ratio_values, metric_key=None):
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(ratio_ylabel)
    ax.set_xlim(*particle["xlim"])
    ax.set_xscale("log")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ratio_values = np.asarray([y for y in ratio_values if np.isfinite(y)], dtype=float)
    if len(ratio_values):
        if metric_key in {"fake_rate", "fake_energy_fraction"}:
            core_values = ratio_values
        else:
            core_values = robust_ratio_core(ratio_values)
        lower = min(np.min(core_values), 1.0)
        upper = max(np.max(core_values), 1.0)
        span = max(upper - lower, 0.02)
        pad = max(0.025, 0.08 * span)
        ymin = max(0.0, lower - pad)
        ymax = upper + pad
        if ymax - ymin < 0.08:
            center = 0.5 * (ymin + ymax)
            ymin = max(0.0, center - 0.04)
            ymax = center + 0.04
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0.85, 1.15)
    ax.legend(fontsize=12, title="Method", title_fontsize=11, loc="best")


def style_resolution_ratio_axis(ax, particle, ratio_ylabel, ratio_values):
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(ratio_ylabel)
    ax.set_xlim(*particle["xlim"])
    ax.set_xscale("log")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ratio_values = np.asarray([y for y in ratio_values if np.isfinite(y)], dtype=float)
    if len(ratio_values):
        core_values = robust_resolution_ratio_core(ratio_values)
        lower = min(np.min(core_values), 1.0)
        upper = max(np.max(core_values), 1.0)
        span = max(upper - lower, 0.01)
        pad = max(0.012, 0.05 * span)
        ymin = max(0.0, lower - pad)
        ymax = upper + pad
        if ymax - ymin < 0.05:
            center = 0.5 * (ymin + ymax)
            ymin = max(0.0, center - 0.025)
            ymax = center + 0.025
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0.95, 1.05)
    ax.legend(fontsize=12, title="Method", title_fontsize=11, loc="best")


def get_resolution_ylim(particle_key, positive_y):
    positive_y = np.asarray([y for y in positive_y if np.isfinite(y) and y > 0.0], dtype=float)
    if len(positive_y) == 0:
        return (0.0, 0.1)
    if particle_key == "charged_hadrons":
        robust_high = np.nanpercentile(positive_y, 85)
        ymax = max(0.05, min(0.25, robust_high * 1.25))
        return (0.0, ymax)
    ymax = np.max(positive_y) * 1.2
    if ymax < 0.03:
        ymax = 0.03
    return (0.0, ymax)


def robust_ratio_core(ratio_values):
    ratio_values = np.asarray([y for y in ratio_values if np.isfinite(y)], dtype=float)
    if len(ratio_values) < 4:
        return ratio_values
    q1 = np.nanpercentile(ratio_values, 25)
    q3 = np.nanpercentile(ratio_values, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0.0:
        return ratio_values
    lower = q1 - 2.5 * iqr
    upper = q3 + 2.5 * iqr
    core_values = ratio_values[(ratio_values >= lower) & (ratio_values <= upper)]
    if len(core_values) == 0:
        return ratio_values
    return core_values


def robust_resolution_ratio_core(ratio_values):
    ratio_values = np.asarray([y for y in ratio_values if np.isfinite(y)], dtype=float)
    if len(ratio_values) < 4:
        return ratio_values

    median = np.nanmedian(ratio_values)
    abs_dev = np.abs(ratio_values - median)
    mad = np.nanmedian(abs_dev)
    if np.isfinite(mad) and mad > 0.0:
        robust_sigma = 1.4826 * mad
        keep_width = max(3.5 * robust_sigma, 0.03)
        core_values = ratio_values[abs_dev <= keep_width]
        if len(core_values) >= max(3, len(ratio_values) // 2):
            return core_values

    lower = np.nanpercentile(ratio_values, 15)
    upper = np.nanpercentile(ratio_values, 85)
    core_values = ratio_values[(ratio_values >= lower) & (ratio_values <= upper)]
    if len(core_values) == 0:
        return ratio_values
    return core_values


def filter_resolution_plot_points(particle_key, x_values, y_values, y_errors=None):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0)
    if y_errors is not None:
        y_errors = np.asarray(y_errors, dtype=float)
        valid &= np.isfinite(y_errors)

    if y_errors is None:
        return valid
    return valid, y_errors


def mixed_percentages(cm, fake_row, fake_norm="column"):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    if fake_norm == "row":
        return cm_row

    col_sums = cm.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_col = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums != 0) * 100.0

    disp = cm_row.copy()
    if fake_row < cm_row.shape[0]:
        disp[fake_row, :] = cm_col[fake_row, :]
    return disp


def truth_pid_to_confusion_class(pid_value):
    if pd.isna(pid_value):
        return np.nan
    return pid_conversion_dict.get(pid_value, np.nan)


def predicted_pid_to_confusion_class(pid_value, is_pandora):
    if pd.isna(pid_value):
        return np.nan
    if is_pandora:
        return pandora_to_our_mapping.get(pid_value, np.nan)
    if pid_value < 0:
        return np.nan
    return pid_value


def compute_confusion_matrix(frame, is_pandora, energy_low, energy_high):
    reco_energy_column = "pandora_calibrated_pfo" if is_pandora else "calibrated_E"
    predicted_pid_column = "pandora_pid" if is_pandora else "pred_pid_matched"

    truth_in_bin = (frame["true_showers_E"] > energy_low) & (frame["true_showers_E"] < energy_high)
    fake_in_bin = frame["pid"].isna() & (frame[reco_energy_column] > energy_low) & (frame[reco_energy_column] < energy_high)
    selected = truth_in_bin | fake_in_bin
    subset = frame.loc[selected].copy()

    n_regular_classes = len(CONFUSION_CLASS_ORDER)
    fake_row = n_regular_classes
    missed_col = n_regular_classes
    class_to_index = {cls: idx for idx, cls in enumerate(CONFUSION_CLASS_ORDER)}
    matrix = np.zeros((n_regular_classes + 1, n_regular_classes + 1), dtype=int)

    for true_pid, predicted_pid in zip(subset["pid"].values, subset[predicted_pid_column].values):
        true_class = truth_pid_to_confusion_class(true_pid)
        pred_class = predicted_pid_to_confusion_class(predicted_pid, is_pandora=is_pandora)
        if not pd.isna(true_class) and true_class not in class_to_index:
            continue
        row = class_to_index.get(true_class, fake_row)
        col = class_to_index.get(pred_class, missed_col)
        matrix[row, col] += 1

    return matrix


def plot_confusion_matrix_grid(datasets, output_path):
    datasets_by_label = {dataset["label"]: dataset for dataset in datasets}
    missing_labels = [label for label, _ in CONFUSION_DATASET_LAYOUT if label not in datasets_by_label]
    if missing_labels:
        raise ValueError(f"Missing datasets for combined confusion matrix: {missing_labels}")

    fig, axes = plt.subplots(1, len(CONFUSION_ENERGY_BINS), figsize=(12.5 * len(CONFUSION_ENERGY_BINS), 11.0))
    if len(CONFUSION_ENERGY_BINS) == 1:
        axes = np.asarray([axes])

    datasets_in_cell_order = [datasets_by_label[label] for label, _ in CONFUSION_DATASET_LAYOUT]
    x_labels = [CONFUSION_CLASS_NAMES[cls] for cls in CONFUSION_CLASS_ORDER] + ["missed"]
    y_labels = [CONFUSION_CLASS_NAMES[cls] for cls in CONFUSION_CLASS_ORDER] + ["fake"]
    fake_row = len(CONFUSION_CLASS_ORDER)
    mini_box_size = 0.40
    mini_box_cmaps = {
        label: sns.blend_palette(["#ffffff", build_style(label, confusion=True)["color"]], as_cmap=True)
        for label, _ in CONFUSION_DATASET_LAYOUT
    }

    for ax, (energy_low, energy_high) in zip(axes, CONFUSION_ENERGY_BINS):
        matrices = []
        percentages = []
        for dataset in datasets_in_cell_order:
            matrix = compute_confusion_matrix(dataset["frame"], dataset["is_pandora"], energy_low, energy_high)
            matrices.append(matrix)
            percentages.append(mixed_percentages(matrix, fake_row, fake_norm="column"))

        n_rows = len(y_labels)
        n_cols = len(x_labels)
        ax.set_xlim(-1.35, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_aspect("equal")

        ax.text(-0.68, -0.18, r"$N_{\mathrm{true}}$", ha="center", va="center", fontsize=13)

        for i in range(n_rows):
            if i < fake_row:
                ax.add_patch(
                    Rectangle((-1.30, i), 1.15, 1, facecolor="#fafafa", edgecolor="0.45", linewidth=1.2)
                )
                ax.plot([-0.725, -0.725], [i, i + 1], color="0.45", linewidth=0.8)
                ax.plot([-1.30, -0.15], [i + 0.5, i + 0.5], color="0.45", linewidth=0.8)
                for (label, (x_frac, y_frac)), matrix in zip(CONFUSION_DATASET_LAYOUT, matrices):
                    style = build_style(label, confusion=True)
                    count_value = int(np.sum(matrix[i, :]))
                    x_center = -1.30 + 1.15 * x_frac
                    y_center = i + y_frac
                    x0 = x_center - mini_box_size * 0.9 / 2.0
                    y0 = y_center - mini_box_size / 2.0
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            mini_box_size * 0.9,
                            mini_box_size,
                            facecolor="white",
                            edgecolor=style["color"],
                            linewidth=1.6,
                            linestyle=style["linestyle"],
                        )
                    )
                    ax.text(
                        x_center,
                        y_center,
                        f"{count_value}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=style["color"],
                    )
            for j in range(n_cols):
                ax.add_patch(
                    Rectangle((j, i), 1, 1, facecolor="#f8f8f8", edgecolor="0.45", linewidth=1.5)
                )
                ax.plot([j + 0.5, j + 0.5], [i, i + 1], color="0.45", linewidth=1.0)
                ax.plot([j, j + 1], [i + 0.5, i + 0.5], color="0.45", linewidth=1.0)
                for (label, (x_frac, y_frac)), matrix_percent in zip(CONFUSION_DATASET_LAYOUT, percentages):
                    style = build_style(label, confusion=True)
                    x0 = j + x_frac - mini_box_size / 2.0
                    y0 = i + y_frac - mini_box_size / 2.0
                    percent_value = float(np.clip(matrix_percent[i, j], 0.0, 100.0))
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            mini_box_size,
                            mini_box_size,
                            facecolor=mini_box_cmaps[label](0.03 + 0.97 * percent_value / 100.0),
                            edgecolor=style["color"],
                            linewidth=1.8,
                            linestyle=style["linestyle"],
                        )
                    )
                    ax.text(
                        j + x_frac,
                        i + y_frac,
                        f"{int(np.rint(percent_value))}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black" if percent_value < 65.0 else "white",
                    )
                ax.add_patch(
                    Rectangle((j, i), 1, 1, facecolor="none", edgecolor="0.20", linewidth=2.4)
                )

        ax.hlines(fake_row, xmin=-1.35, xmax=n_cols, linewidth=2.4, color="black")
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_yticklabels(y_labels, rotation=0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{energy_low:.0f} GeV < E < {energy_high:.0f} GeV", fontsize=18)
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = []
    for label, _ in CONFUSION_DATASET_LAYOUT:
        style = build_style(label, confusion=True)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                marker="s",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                label=label,
            )
        )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
    scale = metric_scale(metric_key)

    for ax, particle in zip(axes, PARTICLES):
        key = particle["key"]
        all_y = []
        for label, curves in ordered_curve_items(model_curves):
            curve = curves[key]
            x_key = metric_x_key(metric_key)
            style = build_style(label)
            x_values = np.asarray(curve[x_key], dtype=float)
            y_values = np.asarray(curve[metric_key], dtype=float) * scale
            y_errors = np.asarray(curve[error_key], dtype=float) * scale
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

        style_metric_axis(ax, particle, ylabel, metric_key, all_y, logy)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_grid_with_ratio(model_curves, metric_key, error_key, ylabel, output_path, logy=False):
    fig = plt.figure(figsize=(32, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.2], hspace=0.08, wspace=0.25)
    scale = metric_scale(metric_key)

    for column_idx, particle in enumerate(PARTICLES):
        ax = fig.add_subplot(grid[0, column_idx])
        ratio_ax = fig.add_subplot(grid[1, column_idx], sharex=ax)
        key = particle["key"]
        all_y = []
        ratio_values_all = []

        for label, curves in ordered_curve_items(model_curves):
            curve = curves[key]
            x_key = metric_x_key(metric_key)
            style = build_style(label)
            x_values = np.asarray(curve[x_key], dtype=float)
            y_values = np.asarray(curve[metric_key], dtype=float) * scale
            y_errors = np.asarray(curve[error_key], dtype=float) * scale
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

        style_metric_axis(ax, particle, ylabel, metric_key, all_y, logy)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")

        ratio_pairs = build_ratio_pairs(model_curves)
        for method_key, numerator_label, denominator_label in ratio_pairs:
            numerator_curve = model_curves[numerator_label][key]
            denominator_curve = model_curves[denominator_label][key]
            x_key = metric_x_key(metric_key)
            ratio_x, ratio_y, ratio_err = compute_ratio_curve(
                numerator_curve[x_key],
                numerator_curve[metric_key],
                np.asarray(numerator_curve[error_key], dtype=float) / 2.0,
                denominator_curve[x_key],
                denominator_curve[metric_key],
                np.asarray(denominator_curve[error_key], dtype=float) / 2.0,
            )
            if len(ratio_x) == 0:
                continue
            style = build_ratio_style(method_key)
            ratio_ax.plot(
                ratio_x,
                ratio_y,
                label=method_key,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(ratio_ax, ratio_x, ratio_y, ratio_err, style)
            ratio_values_all.extend(ratio_y.tolist())

        style_ratio_axis(ratio_ax, particle, "o3_v01 / o2_v05", ratio_values_all, metric_key=metric_key)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_event_distributions(frame, is_pandora):
    if is_pandora:
        dic = calculate_event_mass_resolution(frame, True, ML_pid=True, fake=False)
        return np.asarray(dic["E_over_true"]), np.asarray(dic["mass_over_true_p"])
    dic = get_response_for_event_energy(None, frame, perfect_pid=False, mass_zero=False, ML_pid=True, pandora=False)
    return np.asarray(dic["energy_over_true"]), np.asarray(dic["mass_over_true_model"])


def select_event_component(frame, class_id=None, truth_ids=None):
    if truth_ids is not None:
        truth_ids_abs = {abs(int(pid)) for pid in truth_ids}
        mask = frame["pid"].apply(lambda pid: (not pd.isna(pid)) and abs(int(pid)) in truth_ids_abs)
    elif "pid_4_class_true" in frame.columns:
        mask = frame["pid_4_class_true"] == class_id
    else:
        mask = frame["pid"].map(pid_conversion_dict) == class_id
    return frame[mask].copy()


def _plot_histogram(ax, values, bins, style, label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    weights = np.ones_like(values) / len(values)
    ax.hist(
        values,
        bins=bins,
        histtype="step",
        weights=weights,
        color=style["color"],
        linewidth=2,
        linestyle=style["linestyle"],
        label=label,
    )


def summarize_distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    median = float(np.median(values))
    p16 = float(np.percentile(values, 16))
    p84 = float(np.percentile(values, 84))
    q68 = 0.5 * (p84 - p16)
    return {
        "median": median,
        "q68": q68,
    }


def plot_event_comparison(
    datasets,
    output_path,
    component_label=None,
    class_id=None,
    truth_ids=None,
    xlim=(0.5, 1.25),
    bins=None,
    logy=False,
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    if bins is None:
        bins = np.linspace(0.5, 1.25, 160)

    for dataset in datasets:
        style = build_style(dataset["label"])
        frame = dataset["frame"]
        if class_id is not None or truth_ids is not None:
            frame = select_event_component(frame, class_id=class_id, truth_ids=truth_ids)
        if len(frame) == 0:
            continue
        energy_over_true, mass_over_true = compute_event_distributions(frame, dataset["is_pandora"])
        energy_stats = summarize_distribution(energy_over_true)
        mass_stats = summarize_distribution(mass_over_true)
        energy_label = dataset["label"]
        mass_label = dataset["label"]
        if energy_stats:
            energy_label += (
                f"\nmed={energy_stats['median']:.3f}, "
                f"q68={energy_stats['q68']:.3f}"
            )
        if mass_stats:
            mass_label += (
                f"\nmed={mass_stats['median']:.3f}, "
                f"q68={mass_stats['q68']:.3f}"
            )
        _plot_histogram(axes[0], energy_over_true, bins, style, energy_label)
        _plot_histogram(axes[1], mass_over_true, bins, style, mass_label)

    axes[0].set_xlabel(r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$")
    axes[1].set_xlabel(r"$M_{\mathrm{reco}} / M_{\mathrm{true}}$")
    axes[0].set_ylabel("Normalized entries")
    axes[1].set_ylabel("Normalized entries")
    axes[0].set_xlim(*xlim)
    axes[1].set_xlim(*xlim)
    if component_label:
        fig.suptitle(component_label, fontsize=18, y=0.98)
    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--")
        if logy:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=16, title="Geometry / method", title_fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96) if component_label else None)
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
        for label, curves in ordered_curve_items(model_curves):
            curve = curves[key]
            style = build_style(label)
            x_values = np.asarray(curve["energy"], dtype=float)
            y_values = np.asarray(curve["resolution"], dtype=float)
            y_errors = np.asarray(curve["errors"], dtype=float)
            valid = filter_resolution_plot_points(particle["key"], x_values, y_values, y_errors)
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
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"Energy resolution $\sigma/\mu$")
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.set_xticks([x for x in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] if particle["xlim"][0] <= x <= particle["xlim"][1]])
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(*get_resolution_ylim(particle["key"], all_y))
        ax.legend(fontsize=16, title="Geometry / method", title_fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_comparison_with_ratio(model_curves, output_path):
    fig = plt.figure(figsize=(32, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.2], hspace=0.08, wspace=0.25)

    for column_idx, particle in enumerate(PARTICLES):
        ax = fig.add_subplot(grid[0, column_idx])
        ratio_ax = fig.add_subplot(grid[1, column_idx], sharex=ax)
        key = particle["key"]
        all_y = []
        ratio_values_all = []
        for label, curves in ordered_curve_items(model_curves):
            curve = curves[key]
            style = build_style(label)
            x_values = np.asarray(curve["energy"], dtype=float)
            y_values = np.asarray(curve["resolution"], dtype=float)
            y_errors = np.asarray(curve["errors"], dtype=float)
            valid = filter_resolution_plot_points(particle["key"], x_values, y_values, y_errors)
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
        ax.set_ylabel(r"Energy resolution $\sigma/\mu$")
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        ax.set_xticks([x for x in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] if particle["xlim"][0] <= x <= particle["xlim"][1]])
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(*get_resolution_ylim(particle["key"], all_y))
        ax.legend(fontsize=15, title="Geometry / method", title_fontsize=13)
        plt.setp(ax.get_xticklabels(), visible=False)

        ratio_pairs = build_ratio_pairs(model_curves)
        for method_key, numerator_label, denominator_label in ratio_pairs:
            numerator_curve = model_curves[numerator_label][key]
            denominator_curve = model_curves[denominator_label][key]
            numerator_x = np.asarray(numerator_curve["energy"], dtype=float)
            numerator_y = np.asarray(numerator_curve["resolution"], dtype=float)
            numerator_err = np.asarray(numerator_curve["errors"], dtype=float) / 2.0
            denominator_x = np.asarray(denominator_curve["energy"], dtype=float)
            denominator_y = np.asarray(denominator_curve["resolution"], dtype=float)
            denominator_err = np.asarray(denominator_curve["errors"], dtype=float) / 2.0

            numerator_valid = filter_resolution_plot_points(particle["key"], numerator_x, numerator_y, numerator_err)
            denominator_valid = filter_resolution_plot_points(particle["key"], denominator_x, denominator_y, denominator_err)
            ratio_x, ratio_y, ratio_err = compute_ratio_curve(
                numerator_x[numerator_valid],
                numerator_y[numerator_valid],
                numerator_err[numerator_valid],
                denominator_x[denominator_valid],
                denominator_y[denominator_valid],
                denominator_err[denominator_valid],
            )
            if len(ratio_x) == 0:
                continue
            style = build_ratio_style(method_key)
            ratio_ax.plot(
                ratio_x,
                ratio_y,
                label=method_key,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=style["markersize"],
            )
            _draw_error_band(ratio_ax, ratio_x, ratio_y, ratio_err, style)
            ratio_values_all.extend(ratio_y.tolist())

        style_resolution_ratio_axis(ratio_ax, particle, "o3_v01 / o2_v05", ratio_values_all)

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
    plot_metric_grid_with_ratio(
        model_eff_curves,
        "eff",
        "errors",
        "Clustering efficiency",
        os.path.join(summary_dir, "overview_Efficiency_clustering_comparison_with_ratio.pdf"),
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
    plot_metric_grid_with_ratio(
        model_eff_curves,
        "eff_pid",
        "errors_pid",
        "Efficiency with PID",
        os.path.join(summary_dir, "overview_Efficiency_pid_comparison_with_ratio.pdf"),
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
    plot_metric_grid_with_ratio(
        model_eff_curves,
        "fake_rate",
        "fake_rate_err",
        "Fake rate",
        os.path.join(summary_dir, "overview_FakeRate_comparison_with_ratio.pdf"),
        logy=True,
    )
    plot_metric_grid(
        model_eff_curves,
        "fake_energy_fraction",
        "fake_energy_fraction_err",
        "Fake energy [%]",
        os.path.join(summary_dir, "overview_FakeEnergy_comparison.pdf"),
        logy=True,
    )
    plot_metric_grid_with_ratio(
        model_eff_curves,
        "fake_energy_fraction",
        "fake_energy_fraction_err",
        "Fake energy [%]",
        os.path.join(summary_dir, "overview_FakeEnergy_comparison_with_ratio.pdf"),
        logy=True,
    )
    plot_confusion_matrix_grid(
        datasets,
        os.path.join(summary_dir, "pid_confusion_matrix_per_energy_comparison.pdf"),
    )
    plot_event_comparison(datasets, os.path.join(summary_dir, "event_energy_mass_comparison.pdf"))
    for component in EVENT_COMPONENTS:
        plot_event_comparison(
            datasets,
            os.path.join(summary_dir, component["output_name"]),
            component_label=component["label"],
            class_id=component.get("class_id"),
            truth_ids=component.get("truth_ids"),
            xlim=component["xlim"],
            bins=component["bins"],
            logy=component.get("logy", False),
        )
    plot_resolution_comparison(
        resolution_curves, os.path.join(summary_dir, "particle_energy_resolution_comparison.pdf")
    )
    plot_resolution_comparison_with_ratio(
        resolution_curves, os.path.join(summary_dir, "particle_energy_resolution_comparison_with_ratio.pdf")
    )

    print("Wrote full-evaluation comparison plots to", summary_dir)
    print("Summary table:", os.path.join(args.output_dir, "full_evaluation_summary.csv"))


if __name__ == "__main__":
    main()
