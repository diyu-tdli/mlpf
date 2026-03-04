#!/usr/bin/env python3
"""
Compare CLD vs ARC vs ARC-modified truth-tracking inputs from parquet files.

This script:
1) matches common pf_tree indices across three geometry folders,
2) aggregates per-event and per-track metrics,
3) writes comparison PNGs and a summary JSON.

Expected folder layout (default):
  <dataset_root>/05/*.parquet
  <dataset_root>/arc/*.parquet
  <dataset_root>/arc_modified/*.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import awkward as ak
except ModuleNotFoundError:
    ak = None


GEOMETRIES = {
    "CLD_o2_v05": "05",
    "ARC": "arc",
    "ARC_modified": "arc_modified",
}

FILE_RE = re.compile(r"pf_tree_(\d+)")
EPS = 1e-12


@dataclass
class Metrics:
    n_files: int = 0
    n_events: int = 0
    n_tracks_total: int = 0
    n_linked_tracks_total: int = 0
    n_tracks_with_cluster_total: int = 0
    n_hits_total: int = 0
    n_gen_total: int = 0

    tracks_per_event: List[float] = field(default_factory=list)
    linked_track_fraction_per_event: List[float] = field(default_factory=list)
    hits_per_event: List[float] = field(default_factory=list)
    gen_per_event: List[float] = field(default_factory=list)

    track_momentum_residual: List[float] = field(default_factory=list)
    track_chi2_over_ndf: List[float] = field(default_factory=list)
    track_cluster_delta_r: List[float] = field(default_factory=list)
    track_cluster_nhits: List[float] = field(default_factory=list)
    track_cluster_energy: List[float] = field(default_factory=list)
    cluster_spread_rms_mm: List[float] = field(default_factory=list)
    cluster_spread_true_p: List[float] = field(default_factory=list)

    pion_track_found: List[float] = field(default_factory=list)
    pion_vertex_r: List[float] = field(default_factory=list)
    pion_true_p: List[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    default_root = "/eos/user/v/vriecher/mlpf_events_new/CLD_o2_v05_ARC_ARCmod_test_5k"
    default_out = "/eos/user/v/vriecher/mlpf_events_new/CLD_o2_v05_ARC_ARCmod_test_5k/comparison_plots"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(default_root))
    parser.add_argument("--output-dir", type=Path, default=Path(default_out))
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of common pf_tree files.",
    )
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=None,
        help="Optional cap on events processed per file.",
    )
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def _extract_file_id(path: Path) -> int | None:
    match = FILE_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_common_files(dataset_root: Path) -> Tuple[List[int], Dict[str, Dict[int, Path]]]:
    per_geometry: Dict[str, Dict[int, Path]] = {}

    for geom_name, subdir in GEOMETRIES.items():
        geom_dir = dataset_root / subdir
        paths = sorted(geom_dir.glob("pf_tree_*.parquet"))
        mapping: Dict[int, Path] = {}
        for p in paths:
            fid = _extract_file_id(p)
            if fid is not None:
                mapping[fid] = p
        if not mapping:
            raise RuntimeError(f"No parquet files found for {geom_name} in {geom_dir}")
        per_geometry[geom_name] = mapping

    common_ids = sorted(set.intersection(*(set(m.keys()) for m in per_geometry.values())))
    if not common_ids:
        raise RuntimeError("No common pf_tree indices across CLD/ARC/ARC_modified")

    return common_ids, per_geometry


def to_numpy_2d(array_like) -> np.ndarray:
    arr = np.asarray(array_like)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0, 0), dtype=float)
        return arr.reshape(1, -1)
    return arr


def eta_phi_from_xyz(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    p = math.sqrt(x * x + y * y + z * z)
    if p < EPS:
        return float("nan"), float("nan")
    phi = math.atan2(y, x)
    cos_theta = np.clip(z / p, -1.0, 1.0)
    theta = math.acos(cos_theta)
    eta = -math.log(math.tan(theta / 2.0 + EPS))
    return eta, phi


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi < -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2
    return math.sqrt(deta * deta + dphi * dphi)


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def process_geometry(
    geom_name: str,
    file_paths: List[Path],
    max_events_per_file: int | None,
) -> Metrics:
    metrics = Metrics()
    required_fields = {"X_gen", "X_track", "X_hit", "ygen_track", "ygen_hit"}

    for ifile, parquet_path in enumerate(file_paths, start=1):
        outputs = ak.from_parquet(str(parquet_path))
        fields = set(outputs.fields)
        missing = required_fields - fields
        if missing:
            raise RuntimeError(
                f"{geom_name} file {parquet_path} is missing fields: {sorted(missing)}"
            )

        n_events = len(outputs["X_gen"])
        if max_events_per_file is not None:
            n_events = min(n_events, max_events_per_file)

        metrics.n_files += 1
        metrics.n_events += n_events

        for ievt in range(n_events):
            x_gen = to_numpy_2d(outputs["X_gen"][ievt])
            x_track = to_numpy_2d(outputs["X_track"][ievt])
            x_hit = to_numpy_2d(outputs["X_hit"][ievt])
            y_track = np.asarray(outputs["ygen_track"][ievt], dtype=np.int64)
            y_hit = np.asarray(outputs["ygen_hit"][ievt], dtype=np.int64)

            n_tracks = int(len(y_track))
            n_hits = int(len(y_hit))
            n_gen = int(x_gen.shape[0])

            metrics.n_tracks_total += n_tracks
            metrics.n_hits_total += n_hits
            metrics.n_gen_total += n_gen
            metrics.tracks_per_event.append(n_tracks)
            metrics.hits_per_event.append(n_hits)
            metrics.gen_per_event.append(n_gen)

            valid_track_mask = y_track >= 0
            n_linked_tracks = int(np.sum(valid_track_mask))
            metrics.n_linked_tracks_total += n_linked_tracks
            linked_frac_evt = n_linked_tracks / max(n_tracks, 1)
            metrics.linked_track_fraction_per_event.append(linked_frac_evt)

            if x_hit.shape[1] > 5:
                hit_energy = x_hit[:, 5].astype(float)
            else:
                hit_energy = np.ones(n_hits, dtype=float)
            hit_energy = np.where(np.isfinite(hit_energy), hit_energy, 0.0)

            hit_xyz = x_hit[:, 6:9] if x_hit.shape[1] > 8 else None
            track_p = x_track[:, 5] if x_track.shape[1] > 5 else np.full(n_tracks, np.nan)
            track_xyz = x_track[:, 6:9] if x_track.shape[1] > 8 else None
            true_p = x_gen[:, 11] if x_gen.shape[1] > 11 else np.full(n_gen, np.nan)

            if x_track.shape[1] > 16 and n_tracks > 0:
                ndof = np.clip(x_track[:, 16], EPS, None)
                chi2ndf = x_track[:, 15] / ndof
                metrics.track_chi2_over_ndf.extend(chi2ndf[np.isfinite(chi2ndf)].tolist())

            tracks_with_cluster_evt = 0

            for itrk, gen_idx in enumerate(y_track):
                if gen_idx < 0:
                    continue
                if gen_idx >= n_gen:
                    continue

                mask_hits = y_hit == gen_idx
                nhits_cluster = int(np.sum(mask_hits))
                metrics.track_cluster_nhits.append(nhits_cluster)

                if nhits_cluster > 0:
                    tracks_with_cluster_evt += 1
                    metrics.n_tracks_with_cluster_total += 1

                p_trk = float(track_p[itrk]) if itrk < len(track_p) else float("nan")
                p_gen = float(true_p[gen_idx]) if gen_idx < len(true_p) else float("nan")
                if np.isfinite(p_trk) and np.isfinite(p_gen) and abs(p_gen) > EPS:
                    metrics.track_momentum_residual.append((p_trk - p_gen) / p_gen)

                if nhits_cluster <= 0:
                    continue

                e_cluster = float(np.sum(hit_energy[mask_hits]))
                metrics.track_cluster_energy.append(e_cluster)

                if hit_xyz is None or track_xyz is None:
                    continue

                xyz_cluster = hit_xyz[mask_hits]
                if xyz_cluster.shape[0] == 0:
                    continue

                weights = hit_energy[mask_hits]
                if np.sum(weights) <= EPS:
                    weights = np.ones_like(weights)

                centroid = np.average(xyz_cluster, axis=0, weights=weights)
                dxyz = xyz_cluster - centroid
                dr = np.linalg.norm(dxyz, axis=1)
                rms = math.sqrt(np.average(dr * dr, weights=weights))
                if np.isfinite(rms):
                    metrics.cluster_spread_rms_mm.append(rms)
                    metrics.cluster_spread_true_p.append(p_gen)

                trk_dir = track_xyz[itrk]
                eta_trk, phi_trk = eta_phi_from_xyz(trk_dir)
                eta_cl, phi_cl = eta_phi_from_xyz(centroid)
                if np.isfinite(eta_trk) and np.isfinite(eta_cl):
                    metrics.track_cluster_delta_r.append(delta_r(eta_trk, phi_trk, eta_cl, phi_cl))

            # Charged pion truth-tracking efficiency proxies.
            if x_gen.shape[1] > 17 and x_gen.shape[1] > 0:
                pdg = x_gen[:, 0].astype(np.int64) if x_gen.shape[1] > 0 else np.array([], dtype=np.int64)
                pion_mask = np.abs(pdg) == 211
                pion_ids = np.where(pion_mask)[0]
                if pion_ids.size > 0:
                    found = np.isin(pion_ids, y_track[valid_track_mask])
                    vxyz = x_gen[pion_ids, 15:18]
                    vtx_r = np.linalg.norm(vxyz, axis=1)
                    p_true_pi = x_gen[pion_ids, 11] if x_gen.shape[1] > 11 else np.full_like(vtx_r, np.nan)
                    metrics.pion_track_found.extend(found.astype(float).tolist())
                    metrics.pion_vertex_r.extend(vtx_r.astype(float).tolist())
                    metrics.pion_true_p.extend(p_true_pi.astype(float).tolist())

    return metrics


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def overlay_hist(
    data: Dict[str, np.ndarray],
    out_path: Path,
    xlabel: str,
    ylabel: str = "Entries",
    bins: int = 70,
    xrange: Tuple[float, float] | None = None,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, values in data.items():
        arr = finite(values)
        if arr.size == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            range=xrange,
            histtype="step",
            linewidth=1.8,
            label=f"{name} (N={arr.size})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    _style_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summary_barplot(metrics_by_geom: Dict[str, Metrics], out_path: Path) -> None:
    labels = list(metrics_by_geom.keys())
    tracks_evt = [np.mean(m.tracks_per_event) if m.tracks_per_event else np.nan for m in metrics_by_geom.values()]
    linked_evt = [
        np.mean(m.linked_track_fraction_per_event) if m.linked_track_fraction_per_event else np.nan
        for m in metrics_by_geom.values()
    ]
    no_cluster = []
    median_nhits = []
    for m in metrics_by_geom.values():
        if m.n_linked_tracks_total > 0:
            no_cluster.append(1.0 - m.n_tracks_with_cluster_total / m.n_linked_tracks_total)
        else:
            no_cluster.append(np.nan)
        nh = finite(m.track_cluster_nhits)
        median_nhits.append(np.median(nh) if nh.size else np.nan)

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, tracks_evt, width=width, label="tracks/event")
    ax.bar(x - 0.5 * width, linked_evt, width=width, label="linked-track frac/event")
    ax.bar(x + 0.5 * width, no_cluster, width=width, label="linked tracks w/o cluster")
    ax.bar(x + 1.5 * width, median_nhits, width=width, label="median cluster hits/track")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Value")
    ax.set_title("Geometry comparison summary")
    _style_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def efficiency_vs_x(
    x: np.ndarray,
    found: np.ndarray,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    eff = np.full_like(centers, np.nan, dtype=float)
    err = np.full_like(centers, np.nan, dtype=float)
    for ibin in range(len(centers)):
        m = (x >= bins[ibin]) & (x < bins[ibin + 1])
        n = int(np.sum(m))
        if n == 0:
            continue
        p = float(np.mean(found[m]))
        eff[ibin] = p
        err[ibin] = math.sqrt(max(p * (1.0 - p) / n, 0.0))
    return centers, eff, err


def plot_pion_efficiencies(metrics_by_geom: Dict[str, Metrics], out_dir: Path) -> None:
    # vs production radius
    bins_r = np.linspace(0.0, 120.0, 16)
    fig_r, ax_r = plt.subplots(figsize=(9, 6))
    for name, m in metrics_by_geom.items():
        x = finite(m.pion_vertex_r)
        y = finite(m.pion_track_found)
        n = min(len(x), len(y))
        if n == 0:
            continue
        x = x[:n]
        y = y[:n]
        centers, eff, err = efficiency_vs_x(x, y, bins_r)
        ax_r.errorbar(centers, eff, yerr=err, fmt="-o", ms=4, capsize=2, label=name)
    ax_r.set_xlabel("pion production radius R [mm]")
    ax_r.set_ylabel("truth-track found efficiency")
    ax_r.set_ylim(0.0, 1.05)
    _style_axis(ax_r)
    ax_r.legend()
    fig_r.tight_layout()
    fig_r.savefig(out_dir / "pion_track_eff_vs_vertexR.png")
    plt.close(fig_r)

    # vs true momentum
    bins_p = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200], dtype=float)
    fig_p, ax_p = plt.subplots(figsize=(9, 6))
    for name, m in metrics_by_geom.items():
        x = finite(m.pion_true_p)
        y = finite(m.pion_track_found)
        n = min(len(x), len(y))
        if n == 0:
            continue
        x = x[:n]
        y = y[:n]
        centers, eff, err = efficiency_vs_x(x, y, bins_p)
        ax_p.errorbar(centers, eff, yerr=err, fmt="-o", ms=4, capsize=2, label=name)
    ax_p.set_xlabel("pion true momentum p [GeV]")
    ax_p.set_ylabel("truth-track found efficiency")
    ax_p.set_ylim(0.0, 1.05)
    ax_p.set_xscale("log")
    _style_axis(ax_p)
    ax_p.legend()
    fig_p.tight_layout()
    fig_p.savefig(out_dir / "pion_track_eff_vs_trueP.png")
    plt.close(fig_p)


def plot_cluster_spread_vs_p(metrics_by_geom: Dict[str, Metrics], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200], dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for name, m in metrics_by_geom.items():
        p = finite(m.cluster_spread_true_p)
        s = finite(m.cluster_spread_rms_mm)
        n = min(len(p), len(s))
        if n == 0:
            continue
        p = p[:n]
        s = s[:n]
        med = np.full(len(centers), np.nan, dtype=float)
        for ibin in range(len(centers)):
            mask = (p >= bins[ibin]) & (p < bins[ibin + 1])
            if np.sum(mask) > 0:
                med[ibin] = np.median(s[mask])
        ax.plot(centers, med, "-o", label=name)

    ax.set_xscale("log")
    ax.set_xlabel("true momentum p [GeV]")
    ax.set_ylabel("median cluster spread RMS [mm]")
    ax.set_title("Cluster spatial spread vs momentum")
    _style_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(metrics_by_geom: Dict[str, Metrics], out_path: Path) -> None:
    payload = {}
    for name, m in metrics_by_geom.items():
        payload[name] = {
            "n_files": m.n_files,
            "n_events": m.n_events,
            "n_tracks_total": m.n_tracks_total,
            "n_linked_tracks_total": m.n_linked_tracks_total,
            "n_tracks_with_cluster_total": m.n_tracks_with_cluster_total,
            "n_hits_total": m.n_hits_total,
            "n_gen_total": m.n_gen_total,
            "tracks_per_event_mean": float(np.mean(m.tracks_per_event)) if m.tracks_per_event else None,
            "linked_track_fraction_event_mean": (
                float(np.mean(m.linked_track_fraction_per_event))
                if m.linked_track_fraction_per_event
                else None
            ),
            "linked_tracks_without_cluster_fraction_total": (
                float(1.0 - m.n_tracks_with_cluster_total / max(m.n_linked_tracks_total, 1))
                if m.n_linked_tracks_total > 0
                else None
            ),
            "cluster_hits_per_track_median": (
                float(np.median(finite(m.track_cluster_nhits))) if len(m.track_cluster_nhits) else None
            ),
            "track_delta_r_median": (
                float(np.median(finite(m.track_cluster_delta_r))) if len(m.track_cluster_delta_r) else None
            ),
            "track_p_residual_median": (
                float(np.median(finite(m.track_momentum_residual))) if len(m.track_momentum_residual) else None
            ),
            "cluster_spread_rms_median_mm": (
                float(np.median(finite(m.cluster_spread_rms_mm))) if len(m.cluster_spread_rms_mm) else None
            ),
        }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.dpi"] = args.dpi

    if ak is None:
        raise RuntimeError(
            "Missing dependency `awkward`. Activate your mlpf environment first "
            "(the one used for notebook parquet studies)."
        )

    common_ids, per_geometry = collect_common_files(args.dataset_root)
    if args.max_files is not None:
        common_ids = common_ids[: args.max_files]
    if not common_ids:
        raise RuntimeError("No files selected after --max-files filtering.")

    print(f"Using {len(common_ids)} common pf_tree files")
    print(f"First 10 file IDs: {common_ids[:10]}")

    metrics_by_geom: Dict[str, Metrics] = {}
    for geom_name in GEOMETRIES:
        selected_files = [per_geometry[geom_name][fid] for fid in common_ids]
        print(f"[{geom_name}] processing {len(selected_files)} files")
        metrics = process_geometry(geom_name, selected_files, args.max_events_per_file)
        metrics_by_geom[geom_name] = metrics
        print(
            f"[{geom_name}] events={metrics.n_events}, tracks={metrics.n_tracks_total}, "
            f"linked_tracks={metrics.n_linked_tracks_total}"
        )

    # Core overlays
    overlay_hist(
        {k: np.asarray(v.tracks_per_event) for k, v in metrics_by_geom.items()},
        args.output_dir / "tracks_per_event.png",
        xlabel="tracks per event",
        bins=60,
    )
    overlay_hist(
        {k: np.asarray(v.linked_track_fraction_per_event) for k, v in metrics_by_geom.items()},
        args.output_dir / "linked_track_fraction_per_event.png",
        xlabel="linked track fraction per event",
        bins=60,
        xrange=(0, 1),
    )
    overlay_hist(
        {k: np.asarray(v.track_cluster_nhits) for k, v in metrics_by_geom.items()},
        args.output_dir / "cluster_hits_per_linked_track.png",
        xlabel="number of linked hits in particle cluster",
        bins=80,
        logy=True,
    )
    overlay_hist(
        {k: np.asarray(v.track_cluster_energy) for k, v in metrics_by_geom.items()},
        args.output_dir / "cluster_energy_per_linked_track.png",
        xlabel="cluster energy assigned to linked track [GeV]",
        bins=80,
        logy=True,
    )
    overlay_hist(
        {k: np.asarray(v.track_cluster_delta_r) for k, v in metrics_by_geom.items()},
        args.output_dir / "track_cluster_deltaR.png",
        xlabel="deltaR(track direction, cluster centroid)",
        bins=70,
        xrange=(0, 1.5),
        logy=True,
    )
    overlay_hist(
        {k: np.asarray(v.track_momentum_residual) for k, v in metrics_by_geom.items()},
        args.output_dir / "track_momentum_residual.png",
        xlabel="(p_track - p_true)/p_true",
        bins=90,
        xrange=(-1.0, 1.0),
        logy=True,
    )
    overlay_hist(
        {k: np.asarray(v.track_chi2_over_ndf) for k, v in metrics_by_geom.items()},
        args.output_dir / "track_chi2_over_ndf.png",
        xlabel="track chi2/ndf",
        bins=90,
        xrange=(0, 20),
        logy=True,
    )
    overlay_hist(
        {k: np.asarray(v.cluster_spread_rms_mm) for k, v in metrics_by_geom.items()},
        args.output_dir / "cluster_spread_rms_mm.png",
        xlabel="cluster spread RMS around centroid [mm]",
        bins=80,
        logy=True,
    )

    summary_barplot(metrics_by_geom, args.output_dir / "summary_metrics.png")
    plot_cluster_spread_vs_p(metrics_by_geom, args.output_dir / "cluster_spread_vs_trueP.png")
    plot_pion_efficiencies(metrics_by_geom, args.output_dir)

    write_summary(metrics_by_geom, args.output_dir / "summary_metrics.json")

    print(f"Saved plots and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
