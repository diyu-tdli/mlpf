#!/usr/bin/env python3
"""Vergleich von CLD/ARC/ARC_modified mit Ratio-Panels (relativ zu CLD).

Feldreferenzen orientieren sich an:
- notebooks/24_explore_parquet_files.ipynb
- notebooks/arc_gen_extrapolation_check.py

Wichtige erwartete Felder:
- X_track, ygen_track
- X_hit, ygen_hit
- optional: X_gen
- optional: X_cluster

Spaltenkonventionen (wie in den Referenzdateien verwendet):
- X_track[:, 2]   -> eta
- X_track[:, 3:5] -> sin(phi), cos(phi)
- X_track[:, 5]   -> momentumartige Groesse
- X_hit[:, 5]     -> Hit-Energie
- X_hit[:, 6:9]   -> x,y,z
- X_hit[:, -2]+1  -> Hit-Type (Calo ~ {1,2})
- X_gen[:, 8]     -> Teilchenenergie (falls vorhanden)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import importlib.util
import numpy as np

AK_IMPORT_ERROR = None
try:
    import awkward as ak
except ModuleNotFoundError as exc:
    ak = None
    AK_IMPORT_ERROR = exc


GEOMETRIES = {
    "CLD": {"subdir": "05", "pattern": re.compile(r"^pf_tree_(\d+)\.parquet$")},
    "ARC": {"subdir": "arc", "pattern": re.compile(r"^pf_tree_(\d+)_arc\.parquet$")},
    "ARC_modified": {
        "subdir": "arc_modified",
        "pattern": re.compile(r"^pf_tree_(\d+)_arcmod\.parquet$"),
    },
}

GEOM_ORDER = ["CLD", "ARC", "ARC_modified"]
GEOM_COLORS = {"CLD": "#1f77b4", "ARC": "#d62728", "ARC_modified": "#2ca02c"}
GEN_ENERGY_COL = 8


@dataclass
class GeometryStats:
    files_processed: int = 0
    files_failed: int = 0
    events_read: int = 0
    events_processed: int = 0
    events_skipped: int = 0
    missing_fields: Counter = field(default_factory=Counter)


METRIC_SPECS = [
    {
        "key": "n_tracks_per_event",
        "title": "Tracks per Event",
        "xlabel": "Number of Tracks",
        "discrete": True,
        "clip_q": 0.995,
    },
    {
        "key": "n_calo_hits_per_event",
        "title": "Calorimeter Hits per Event",
        "xlabel": "Number of Calorimeter Hits",
        "discrete": True,
        "clip_q": 0.995,
    },
    {
        "key": "n_clusters_per_event",
        "title": "Clusters per Event (gen-ID based)",
        "xlabel": "Number of Clusters",
        "discrete": True,
        "clip_q": 0.995,
    },
    {
        "key": "cluster_size_hits",
        "title": "Cluster Size (Hits per Cluster)",
        "xlabel": "Hits per Cluster",
        "discrete": True,
        "clip_q": 0.995,
    },
    {
        "key": "cluster_r68_hits",
        "title": r"Cluster Radial Size $R_{68}^{\mathrm{hits}}$",
        "xlabel": r"$R_{68}$ in $\Delta R$",
        "discrete": False,
        "bins": 70,
        "clip_q": 0.995,
        "non_negative": True,
        "log_y": True,
    },
    {
        "key": "cluster_r90_hits",
        "title": r"Cluster Radial Size $R_{90}^{\mathrm{hits}}$",
        "xlabel": r"$R_{90}$ in $\Delta R$",
        "discrete": False,
        "bins": 70,
        "clip_q": 0.995,
        "non_negative": True,
        "log_y": True,
    },
    {
        "key": "cluster_r68_energy",
        "title": r"Cluster Radial Size $R_{68}^{E}$",
        "xlabel": r"$R_{68}$ in $\Delta R$",
        "discrete": False,
        "bins": 70,
        "clip_q": 0.995,
        "non_negative": True,
        "log_y": True,
    },
    {
        "key": "cluster_r90_energy",
        "title": r"Cluster Radial Size $R_{90}^{E}$",
        "xlabel": r"$R_{90}$ in $\Delta R$",
        "discrete": False,
        "bins": 70,
        "clip_q": 0.995,
        "non_negative": True,
        "log_y": True,
    },
    {
        "key": "cluster_energy",
        "title": r"Cluster Energy",
        "xlabel": r"Cluster Energy (a.u.)",
        "discrete": False,
        "bins": 60,
        "clip_q": 0.995,
        "non_negative": True,
    },
    {
        "key": "track_momentum",
        "title": r"Track $p$ (from X_track[:,5])",
        "xlabel": r"$p$ (a.u.)",
        "discrete": False,
        "bins": 60,
        "clip_q": 0.995,
        "non_negative": True,
    },
    {
        "key": "delta_r_track_cluster_genmatch",
        "title": r"Track-Cluster $\Delta R$ (same gen-ID)",
        "xlabel": r"$\Delta R$",
        "discrete": False,
        "bins": 70,
        "fixed_range": (0.0, 0.5),
        "non_negative": True,
        "log_y": True,
    },
    {
        "key": "delta_eta_track_cluster_genmatch",
        "title": r"Track-Cluster $\Delta\eta$ (same gen-ID)",
        "xlabel": r"$\Delta\eta$",
        "discrete": False,
        "bins": 80,
        "fixed_range": (-0.2, 0.2),
        "log_y": True,
    },
    {
        "key": "delta_phi_track_cluster_genmatch",
        "title": r"Track-Cluster $\Delta\phi$ (same gen-ID)",
        "xlabel": r"$\Delta\phi$",
        "discrete": False,
        "bins": 80,
        "fixed_range": (-0.2, 0.2),
        "log_y": True,
    },
]

INTERNAL_PROFILE_KEYS = [
    "dr_genmatch_profile_track_p_x",
    "dr_genmatch_profile_track_p_y",
    "dr_genmatch_profile_abseta_x",
    "dr_genmatch_profile_abseta_y",
    "dr_genmatch_profile_particle_energy_x",
    "dr_genmatch_profile_particle_energy_y",
    "dr_genmatch_profile_cluster_energy_x",
    "dr_genmatch_profile_cluster_energy_y",
    "dr_genmatch_tail002_profile_track_p_x",
    "dr_genmatch_tail002_profile_track_p_y",
    "dr_genmatch_tail005_profile_track_p_x",
    "dr_genmatch_tail005_profile_track_p_y",
    "dr_genmatch_tail002_profile_abseta_x",
    "dr_genmatch_tail002_profile_abseta_y",
    "dr_genmatch_tail005_profile_abseta_x",
    "dr_genmatch_tail005_profile_abseta_y",
    "dr_genmatch_tail002_profile_particle_energy_x",
    "dr_genmatch_tail002_profile_particle_energy_y",
    "dr_genmatch_tail005_profile_particle_energy_x",
    "dr_genmatch_tail005_profile_particle_energy_y",
    "dr_genmatch_tail002_profile_cluster_energy_x",
    "dr_genmatch_tail002_profile_cluster_energy_y",
    "dr_genmatch_tail005_profile_cluster_energy_x",
    "dr_genmatch_tail005_profile_cluster_energy_y",
    "cluster_size_profile_cluster_energy_x",
    "cluster_size_profile_cluster_energy_y",
    "cluster_r68_hits_profile_cluster_energy_x",
    "cluster_r68_hits_profile_cluster_energy_y",
    "cluster_r90_hits_profile_cluster_energy_x",
    "cluster_r90_hits_profile_cluster_energy_y",
    "cluster_r68_energy_profile_cluster_energy_x",
    "cluster_r68_energy_profile_cluster_energy_y",
    "cluster_r90_energy_profile_cluster_energy_x",
    "cluster_r90_energy_profile_cluster_energy_y",
]

PROFILE_ARRAY_PAIRS = [
    ("dr_genmatch_profile_track_p_x", "dr_genmatch_profile_track_p_y"),
    ("dr_genmatch_profile_abseta_x", "dr_genmatch_profile_abseta_y"),
    ("dr_genmatch_profile_particle_energy_x", "dr_genmatch_profile_particle_energy_y"),
    ("dr_genmatch_profile_cluster_energy_x", "dr_genmatch_profile_cluster_energy_y"),
    ("dr_genmatch_tail002_profile_track_p_x", "dr_genmatch_tail002_profile_track_p_y"),
    ("dr_genmatch_tail005_profile_track_p_x", "dr_genmatch_tail005_profile_track_p_y"),
    ("dr_genmatch_tail002_profile_abseta_x", "dr_genmatch_tail002_profile_abseta_y"),
    ("dr_genmatch_tail005_profile_abseta_x", "dr_genmatch_tail005_profile_abseta_y"),
    (
        "dr_genmatch_tail002_profile_particle_energy_x",
        "dr_genmatch_tail002_profile_particle_energy_y",
    ),
    (
        "dr_genmatch_tail005_profile_particle_energy_x",
        "dr_genmatch_tail005_profile_particle_energy_y",
    ),
    (
        "dr_genmatch_tail002_profile_cluster_energy_x",
        "dr_genmatch_tail002_profile_cluster_energy_y",
    ),
    (
        "dr_genmatch_tail005_profile_cluster_energy_x",
        "dr_genmatch_tail005_profile_cluster_energy_y",
    ),
    ("cluster_size_profile_cluster_energy_x", "cluster_size_profile_cluster_energy_y"),
    (
        "cluster_r68_hits_profile_cluster_energy_x",
        "cluster_r68_hits_profile_cluster_energy_y",
    ),
    (
        "cluster_r90_hits_profile_cluster_energy_x",
        "cluster_r90_hits_profile_cluster_energy_y",
    ),
    (
        "cluster_r68_energy_profile_cluster_energy_x",
        "cluster_r68_energy_profile_cluster_energy_y",
    ),
    (
        "cluster_r90_energy_profile_cluster_energy_x",
        "cluster_r90_energy_profile_cluster_energy_y",
    ),
]


def _empty_metrics_dict() -> Dict[str, List[float]]:
    keys = [spec["key"] for spec in METRIC_SPECS] + INTERNAL_PROFILE_KEYS
    return {k: [] for k in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/eos/user/v/vriecher/mlpf_events_new/CLD_o2_v05_ARC_ARCmod_test_5k",
        help="Root mit Unterordnern 05, arc, arc_modified",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Ausgabeordner fuer PNGs und summary_metrics.json",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximale Anzahl gemeinsamer Dateien (nach Sortierung).",
    )
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=None,
        help="Maximale Anzahl Events pro Datei.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="DPI fuer PNG-Ausgabe.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Anzahl paralleler Worker (Parallelisierung ueber Geometrien).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Fortschrittslog alle N eingelesenen Events je Geometrie.",
    )
    return parser.parse_args()


def ensure_parquet_backend() -> None:
    if ak is None:
        raise RuntimeError(
            "Fehler: Paket 'awkward' fehlt. Bitte installieren, z.B.:\n"
            "  pip install awkward pyarrow\n"
            "oder in conda:\n"
            "  conda install -c conda-forge awkward pyarrow"
        )
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError(
            "Fehler: 'pyarrow' fehlt. 'ak.from_parquet' benoetigt pyarrow.\n"
            "Bitte installieren, z.B.:\n"
            "  pip install pyarrow\n"
            "oder:\n"
            "  conda install -c conda-forge pyarrow"
        )


def _is_awkward_record(obj) -> bool:
    return ak is not None and isinstance(obj, ak.Record)


def _n_events(obj) -> int:
    if _is_awkward_record(obj):
        for field_name in obj.fields:
            try:
                return len(obj[field_name])
            except Exception:
                continue
        return 0
    try:
        return len(obj)
    except Exception:
        return 0


def _to_numpy(value, dtype=None) -> np.ndarray:
    if ak is not None and isinstance(value, (ak.Array, ak.Record)):
        try:
            arr = ak.to_numpy(value)
        except Exception:
            # Fallback for jagged arrays that cannot be cast to RegularArray.
            arr = np.asarray(ak.to_list(value), dtype=object)
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr)
    if dtype is not None:
        try:
            arr = arr.astype(dtype, copy=False)
        except Exception:
            pass
    return arr


def _as_2d_float(value) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=float)
    arr = _to_numpy(value, dtype=float)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0, 0), dtype=float)
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _as_1d_float(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    arr = _to_numpy(value)
    arr = np.ravel(arr)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    try:
        return arr.astype(float, copy=False)
    except Exception:
        return np.empty(0, dtype=float)


def _gen_energy_array(x_gen: np.ndarray) -> np.ndarray:
    if x_gen.shape[0] == 0 or x_gen.shape[1] <= GEN_ENERGY_COL:
        return np.empty(0, dtype=float)
    return np.nan_to_num(
        x_gen[:, GEN_ENERGY_COL],
        nan=np.nan,
        posinf=np.nan,
        neginf=np.nan,
    )


def _extract_event_field(array, field_name: str, event_idx: int, stats: GeometryStats):
    if field_name not in array.fields:
        stats.missing_fields[field_name] += 1
        return None
    try:
        field = array[field_name]
        if _is_awkward_record(array):
            try:
                n_field = len(field)
            except Exception:
                n_field = 1
            if event_idx < 0 or event_idx >= n_field:
                stats.missing_fields[f"{field_name}__event_oob"] += 1
                return None
            try:
                return field[event_idx]
            except Exception:
                return field
        return field[event_idx]
    except Exception:
        stats.missing_fields[f"{field_name}__read_error"] += 1
        return None


def _wrap_phi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _eta_phi_from_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if xyz.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = 0.5 * np.log((r + z) / (r - z))
    eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.arctan2(y, x)
    return eta, phi


def _safe_nan_array(values: Iterable[float], *, non_negative: bool = False) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        return np.empty(0, dtype=float)
    arr = arr[np.isfinite(arr)]
    if non_negative:
        arr = arr[arr >= 0.0]
    return arr


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or weights.size == 0:
        return float("nan")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(mask):
        return float("nan")
    v = values[mask]
    w = weights[mask]
    if v.size == 0:
        return float("nan")
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return float(np.quantile(v, q))
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / wsum
    idx = int(np.searchsorted(cdf, float(q), side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def _cluster_info_from_hits(
    x_hit: np.ndarray,
    ygen_hit: np.ndarray,
) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[int, float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    int,
    Dict[str, List[float]],
]:
    cluster_profiles = {
        "cluster_size_profile_cluster_energy_x": [],
        "cluster_size_profile_cluster_energy_y": [],
        "cluster_r68_hits_profile_cluster_energy_x": [],
        "cluster_r68_hits_profile_cluster_energy_y": [],
        "cluster_r90_hits_profile_cluster_energy_x": [],
        "cluster_r90_hits_profile_cluster_energy_y": [],
        "cluster_r68_energy_profile_cluster_energy_x": [],
        "cluster_r68_energy_profile_cluster_energy_y": [],
        "cluster_r90_energy_profile_cluster_energy_x": [],
        "cluster_r90_energy_profile_cluster_energy_y": [],
    }
    n = min(x_hit.shape[0], ygen_hit.shape[0])
    if n <= 0:
        return {}, {}, [], [], [], [], [], [], 0, cluster_profiles

    x_hit = x_hit[:n]
    ygen_hit = ygen_hit[:n]

    finite_link = np.isfinite(ygen_hit)
    if not np.any(finite_link):
        return {}, {}, [], [], [], [], [], [], 0, cluster_profiles

    x_hit = x_hit[finite_link]
    gen_ids = np.rint(ygen_hit[finite_link]).astype(int)

    if x_hit.shape[0] == 0:
        return {}, {}, [], [], [], [], [], [], 0, cluster_profiles

    if x_hit.shape[1] >= 2:
        hit_type = np.rint(np.nan_to_num(x_hit[:, -2], nan=-99.0) + 1).astype(int)
        calo_mask = np.isin(hit_type, np.array([1, 2], dtype=int))
    else:
        calo_mask = np.ones(x_hit.shape[0], dtype=bool)

    # Fallback: wenn die Typ-Codierung hier nicht passt, nicht alles verwerfen.
    if not np.any(calo_mask):
        calo_mask = np.ones(x_hit.shape[0], dtype=bool)

    x_hit = x_hit[calo_mask]
    gen_ids = gen_ids[calo_mask]

    if x_hit.shape[0] == 0:
        return {}, {}, [], [], [], [], [], [], 0, cluster_profiles

    if x_hit.shape[1] > 5:
        energy = np.nan_to_num(x_hit[:, 5], nan=0.0, posinf=0.0, neginf=0.0)
    else:
        energy = np.ones(x_hit.shape[0], dtype=float)

    unique_ids, inv, counts = np.unique(gen_ids, return_inverse=True, return_counts=True)
    if unique_ids.size == 0:
        return {}, {}, [], [], [], [], [], [], 0, cluster_profiles

    counts_f = counts.astype(float)
    wsum = np.bincount(inv, weights=energy, minlength=unique_ids.size).astype(float)

    eta_center = np.full(unique_ids.size, np.nan, dtype=float)
    phi_center = np.full(unique_ids.size, np.nan, dtype=float)

    if x_hit.shape[1] > 8:
        xyz = np.nan_to_num(x_hit[:, 6:9], nan=0.0, posinf=0.0, neginf=0.0)
        sum_x = np.bincount(inv, weights=xyz[:, 0], minlength=unique_ids.size)
        sum_y = np.bincount(inv, weights=xyz[:, 1], minlength=unique_ids.size)
        sum_z = np.bincount(inv, weights=xyz[:, 2], minlength=unique_ids.size)
        sum_wx = np.bincount(inv, weights=energy * xyz[:, 0], minlength=unique_ids.size)
        sum_wy = np.bincount(inv, weights=energy * xyz[:, 1], minlength=unique_ids.size)
        sum_wz = np.bincount(inv, weights=energy * xyz[:, 2], minlength=unique_ids.size)

        use_weighted = wsum > 0.0
        denom = np.where(use_weighted, wsum, counts_f)
        center_x = np.divide(
            np.where(use_weighted, sum_wx, sum_x),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0.0,
        )
        center_y = np.divide(
            np.where(use_weighted, sum_wy, sum_y),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0.0,
        )
        center_z = np.divide(
            np.where(use_weighted, sum_wz, sum_z),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0.0,
        )
        eta_center, phi_center = _eta_phi_from_xyz(np.column_stack((center_x, center_y, center_z)))
    elif x_hit.shape[1] > 4:
        eta = np.nan_to_num(x_hit[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
        phi = np.arctan2(
            np.nan_to_num(x_hit[:, 3], nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(x_hit[:, 4], nan=1.0, posinf=1.0, neginf=-1.0),
        )
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        sum_eta = np.bincount(inv, weights=eta, minlength=unique_ids.size)
        sum_sin = np.bincount(inv, weights=sin_phi, minlength=unique_ids.size)
        sum_cos = np.bincount(inv, weights=cos_phi, minlength=unique_ids.size)
        sum_w_eta = np.bincount(inv, weights=energy * eta, minlength=unique_ids.size)
        sum_w_sin = np.bincount(inv, weights=energy * sin_phi, minlength=unique_ids.size)
        sum_w_cos = np.bincount(inv, weights=energy * cos_phi, minlength=unique_ids.size)

        use_weighted = wsum > 0.0
        denom = np.where(use_weighted, wsum, counts_f)
        eta_center = np.divide(
            np.where(use_weighted, sum_w_eta, sum_eta),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0.0,
        )
        phi_center = np.arctan2(
            np.where(use_weighted, sum_w_sin, sum_sin),
            np.where(use_weighted, sum_w_cos, sum_cos),
        )

    centers: Dict[int, Tuple[float, float]] = {}
    finite_center = np.isfinite(eta_center) & np.isfinite(phi_center)
    for gid, eta_c, phi_c in zip(unique_ids[finite_center], eta_center[finite_center], phi_center[finite_center]):
        centers[int(gid)] = (float(eta_c), float(phi_c))

    cluster_energy_by_id = {
        int(gid): max(float(e_sum), 0.0)
        for gid, e_sum in zip(unique_ids, wsum)
        if gid >= 0 and np.isfinite(e_sum)
    }

    nonneg_mask = unique_ids >= 0
    n_clusters_nonneg = int(np.sum(nonneg_mask))
    cluster_sizes = counts_f[nonneg_mask].tolist()
    cluster_energies = np.maximum(wsum[nonneg_mask], 0.0).tolist()
    cluster_energy_nonneg = np.maximum(wsum[nonneg_mask], 0.0)
    cluster_size_nonneg = counts_f[nonneg_mask]
    finite_cluster_size = (
        np.isfinite(cluster_energy_nonneg)
        & (cluster_energy_nonneg > 0.0)
        & np.isfinite(cluster_size_nonneg)
    )
    if np.any(finite_cluster_size):
        cluster_profiles["cluster_size_profile_cluster_energy_x"].extend(
            cluster_energy_nonneg[finite_cluster_size].tolist()
        )
        cluster_profiles["cluster_size_profile_cluster_energy_y"].extend(
            cluster_size_nonneg[finite_cluster_size].tolist()
        )
    cluster_r68_hits: List[float] = []
    cluster_r90_hits: List[float] = []
    cluster_r68_energy: List[float] = []
    cluster_r90_energy: List[float] = []

    hit_eta = np.empty(0, dtype=float)
    hit_phi = np.empty(0, dtype=float)
    if x_hit.shape[1] > 8:
        hit_xyz = np.nan_to_num(x_hit[:, 6:9], nan=0.0, posinf=0.0, neginf=0.0)
        hit_eta, hit_phi = _eta_phi_from_xyz(hit_xyz)
    elif x_hit.shape[1] > 4:
        hit_eta = np.nan_to_num(x_hit[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
        hit_phi = np.arctan2(
            np.nan_to_num(x_hit[:, 3], nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(x_hit[:, 4], nan=1.0, posinf=1.0, neginf=-1.0),
        )

    if hit_eta.size > 0 and hit_phi.size > 0:
        valid_hit_angle = np.isfinite(hit_eta) & np.isfinite(hit_phi)
        order = np.argsort(inv, kind="mergesort")
        offsets = np.concatenate(([0], np.cumsum(counts)))
        for i_gid, gid in enumerate(unique_ids):
            if gid < 0 or not finite_center[i_gid]:
                continue
            cluster_energy_sum = max(float(wsum[i_gid]), 0.0)
            i0 = int(offsets[i_gid])
            i1 = int(offsets[i_gid + 1])
            if i1 <= i0:
                continue
            idx = order[i0:i1]
            idx = idx[valid_hit_angle[idx]]
            if idx.size == 0:
                continue
            deta = hit_eta[idx] - eta_center[i_gid]
            dphi = _wrap_phi(hit_phi[idx] - phi_center[i_gid])
            dr = np.sqrt(deta**2 + dphi**2)
            finite_dr = np.isfinite(dr)
            dr = dr[finite_dr]
            if dr.size == 0:
                continue
            r68_hits = float(np.quantile(dr, 0.68))
            r90_hits = float(np.quantile(dr, 0.90))
            cluster_r68_hits.append(r68_hits)
            cluster_r90_hits.append(r90_hits)
            e = np.clip(energy[idx][finite_dr], 0.0, np.inf)
            r68_energy = _weighted_quantile(dr, e, 0.68)
            r90_energy = _weighted_quantile(dr, e, 0.90)
            cluster_r68_energy.append(r68_energy)
            cluster_r90_energy.append(r90_energy)

            if cluster_energy_sum > 0.0:
                if np.isfinite(r68_hits):
                    cluster_profiles["cluster_r68_hits_profile_cluster_energy_x"].append(cluster_energy_sum)
                    cluster_profiles["cluster_r68_hits_profile_cluster_energy_y"].append(r68_hits)
                if np.isfinite(r90_hits):
                    cluster_profiles["cluster_r90_hits_profile_cluster_energy_x"].append(cluster_energy_sum)
                    cluster_profiles["cluster_r90_hits_profile_cluster_energy_y"].append(r90_hits)
                if np.isfinite(r68_energy):
                    cluster_profiles["cluster_r68_energy_profile_cluster_energy_x"].append(cluster_energy_sum)
                    cluster_profiles["cluster_r68_energy_profile_cluster_energy_y"].append(r68_energy)
                if np.isfinite(r90_energy):
                    cluster_profiles["cluster_r90_energy_profile_cluster_energy_x"].append(cluster_energy_sum)
                    cluster_profiles["cluster_r90_energy_profile_cluster_energy_y"].append(r90_energy)

    return (
        centers,
        cluster_energy_by_id,
        cluster_sizes,
        cluster_energies,
        cluster_r68_hits,
        cluster_r90_hits,
        cluster_r68_energy,
        cluster_r90_energy,
        n_clusters_nonneg,
        cluster_profiles,
    )


def _track_arrays(
    x_track: np.ndarray,
    ygen_track: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(x_track.shape[0], ygen_track.shape[0])
    if n <= 0:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    x_track = x_track[:n]
    ygen_track = ygen_track[:n]

    gen_ids = np.full(n, -999999, dtype=int)
    finite_gid = np.isfinite(ygen_track)
    gen_ids[finite_gid] = np.rint(ygen_track[finite_gid]).astype(int)

    # Wie in arc_gen_extrapolation_check.py bevorzugt: eta/phi aus extrapoliertem Track-XYZ.
    eta = np.full(n, np.nan, dtype=float)
    phi = np.full(n, np.nan, dtype=float)
    if x_track.shape[1] > 14:
        xyz = np.nan_to_num(x_track[:, 12:15], nan=0.0, posinf=0.0, neginf=0.0)
        eta_xyz, phi_xyz = _eta_phi_from_xyz(xyz)
        valid_xyz = np.linalg.norm(xyz, axis=1) > 0.0
        eta[valid_xyz] = eta_xyz[valid_xyz]
        phi[valid_xyz] = phi_xyz[valid_xyz]

    # Fallback auf gespeicherte eta/sin(phi)/cos(phi), falls XYZ nicht brauchbar.
    missing_eta = ~np.isfinite(eta)
    missing_phi = ~np.isfinite(phi)
    if x_track.shape[1] > 2 and np.any(missing_eta):
        eta_store = np.nan_to_num(x_track[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
        eta[missing_eta] = eta_store[missing_eta]
    if x_track.shape[1] > 4 and np.any(missing_phi):
        sin_phi = np.nan_to_num(x_track[:, 3], nan=0.0, posinf=0.0, neginf=0.0)
        cos_phi = np.nan_to_num(x_track[:, 4], nan=1.0, posinf=1.0, neginf=-1.0)
        phi_store = np.arctan2(sin_phi, cos_phi)
        phi[missing_phi] = phi_store[missing_phi]

    if x_track.shape[1] > 5:
        momentum = np.nan_to_num(x_track[:, 5], nan=np.nan, posinf=np.nan, neginf=np.nan)
    else:
        momentum = np.full(n, np.nan, dtype=float)

    return gen_ids, eta, phi, momentum


def _find_geometry_files(dataset_root: Path) -> Dict[str, Dict[int, Path]]:
    out: Dict[str, Dict[int, Path]] = {}
    for geom, spec in GEOMETRIES.items():
        subdir = dataset_root / spec["subdir"]
        if not subdir.is_dir():
            raise FileNotFoundError(f"Unterordner fehlt: {subdir}")
        mapping: Dict[int, Path] = {}
        for fname in os.listdir(subdir):
            match = spec["pattern"].match(fname)
            if match:
                mapping[int(match.group(1))] = subdir / fname
        if not mapping:
            raise FileNotFoundError(f"Keine passenden pf_tree-Dateien in {subdir}")
        out[geom] = mapping
    return out


def _common_file_indices(file_maps: Mapping[str, Mapping[int, Path]]) -> List[int]:
    common = None
    for geom in GEOM_ORDER:
        idx_set = set(file_maps[geom].keys())
        common = idx_set if common is None else (common & idx_set)
    return sorted(common or [])


def _process_geometry(
    geom: str,
    files: Sequence[Path],
    max_events_per_file: Optional[int],
    log_every: int = 1000,
) -> Tuple[Dict[str, List[float]], GeometryStats]:
    stats = GeometryStats()
    metrics: Dict[str, List[float]] = _empty_metrics_dict()

    for file_path in files:
        try:
            arr = ak.from_parquet(str(file_path))
        except Exception as exc:
            stats.files_failed += 1
            print(f"[{geom}] WARN: Datei konnte nicht gelesen werden: {file_path} ({exc})")
            continue

        stats.files_processed += 1
        n_events_total = _n_events(arr)
        n_events = n_events_total if max_events_per_file is None else min(n_events_total, max_events_per_file)

        for iev in range(n_events):
            stats.events_read += 1
            if log_every > 0 and stats.events_read % log_every == 0:
                print(
                    f"[{geom}] Progress: {stats.events_read} Events eingelesen "
                    f"(processed={stats.events_processed}, skipped={stats.events_skipped})"
                )

            x_track = _as_2d_float(_extract_event_field(arr, "X_track", iev, stats))
            ygen_track = _as_1d_float(_extract_event_field(arr, "ygen_track", iev, stats))
            x_hit = _as_2d_float(_extract_event_field(arr, "X_hit", iev, stats))
            ygen_hit = _as_1d_float(_extract_event_field(arr, "ygen_hit", iev, stats))
            x_gen = _as_2d_float(_extract_event_field(arr, "X_gen", iev, stats))

            if x_track.shape[0] == 0 and x_hit.shape[0] == 0:
                stats.events_skipped += 1
                continue

            stats.events_processed += 1

            metrics["n_tracks_per_event"].append(float(x_track.shape[0]))

            if x_hit.shape[0] > 0 and x_hit.shape[1] >= 2:
                htype = np.rint(np.nan_to_num(x_hit[:, -2], nan=-99.0) + 1).astype(int)
                n_calo_hits = int(np.sum(np.isin(htype, np.array([1, 2], dtype=int))))
                if n_calo_hits == 0:
                    n_calo_hits = int(x_hit.shape[0])
            else:
                n_calo_hits = int(x_hit.shape[0])
            metrics["n_calo_hits_per_event"].append(float(n_calo_hits))

            (
                centers,
                cluster_energy_by_id,
                cluster_sizes,
                cluster_energies,
                cluster_r68_hits,
                cluster_r90_hits,
                cluster_r68_energy,
                cluster_r90_energy,
                n_clusters_nonneg,
                cluster_profiles,
            ) = _cluster_info_from_hits(
                x_hit, ygen_hit
            )
            metrics["n_clusters_per_event"].append(float(n_clusters_nonneg))
            metrics["cluster_size_hits"].extend(cluster_sizes)
            metrics["cluster_energy"].extend(cluster_energies)
            metrics["cluster_r68_hits"].extend(cluster_r68_hits)
            metrics["cluster_r90_hits"].extend(cluster_r90_hits)
            metrics["cluster_r68_energy"].extend(cluster_r68_energy)
            metrics["cluster_r90_energy"].extend(cluster_r90_energy)
            for profile_key, profile_values in cluster_profiles.items():
                metrics[profile_key].extend(profile_values)

            gen_track, tr_eta, tr_phi, tr_p = _track_arrays(x_track, ygen_track)
            gen_energy = _gen_energy_array(x_gen)
            p_finite = tr_p[np.isfinite(tr_p)]
            if p_finite.size > 0:
                metrics["track_momentum"].extend(p_finite.tolist())

            if tr_eta.size > 0 and tr_phi.size > 0 and centers:
                track_mask = (gen_track >= 0) & np.isfinite(tr_eta) & np.isfinite(tr_phi)
                if np.any(track_mask):
                    tr_gid_v = gen_track[track_mask]
                    tr_eta_v = tr_eta[track_mask]
                    tr_phi_v = tr_phi[track_mask]
                    tr_p_v = tr_p[track_mask]

                    center_ids = np.array([gid for gid in centers.keys() if gid >= 0], dtype=int)
                    if center_ids.size > 0:
                        center_ids_sorted = np.sort(center_ids)
                        center_eta_sorted = np.array([centers[int(gid)][0] for gid in center_ids_sorted], dtype=float)
                        center_phi_sorted = np.array([centers[int(gid)][1] for gid in center_ids_sorted], dtype=float)
                        center_energy_sorted = np.array(
                            [cluster_energy_by_id.get(int(gid), np.nan) for gid in center_ids_sorted],
                            dtype=float,
                        )

                        idx = np.searchsorted(center_ids_sorted, tr_gid_v)
                        in_bounds = idx < center_ids_sorted.size
                        has_center = np.zeros_like(in_bounds, dtype=bool)
                        has_center[in_bounds] = center_ids_sorted[idx[in_bounds]] == tr_gid_v[in_bounds]

                        if np.any(has_center):
                            idx_m = idx[has_center]
                            tr_gid_m = tr_gid_v[has_center]
                            tr_eta_m = tr_eta_v[has_center]
                            tr_phi_m = tr_phi_v[has_center]
                            tr_p_m = tr_p_v[has_center]
                            cluster_energy_m = np.clip(
                                center_energy_sorted[idx_m],
                                0.0,
                                np.inf,
                            )
                            particle_energy_m = np.full(tr_gid_m.shape, np.nan, dtype=float)
                            valid_particle_energy = (
                                (tr_gid_m >= 0)
                                & (tr_gid_m < gen_energy.size)
                            )
                            if np.any(valid_particle_energy):
                                particle_energy_m[valid_particle_energy] = gen_energy[
                                    tr_gid_m[valid_particle_energy]
                                ]

                            deta = tr_eta_m - center_eta_sorted[idx_m]
                            dphi = _wrap_phi(tr_phi_m - center_phi_sorted[idx_m])
                            dr = np.sqrt(deta**2 + dphi**2)

                            finite_mask = np.isfinite(deta) & np.isfinite(dphi) & np.isfinite(dr)
                            if np.any(finite_mask):
                                deta = deta[finite_mask]
                                dphi = dphi[finite_mask]
                                dr = dr[finite_mask]
                                tr_p_m = tr_p_m[finite_mask]
                                tr_eta_m = tr_eta_m[finite_mask]
                                cluster_energy_m = cluster_energy_m[finite_mask]
                                particle_energy_m = particle_energy_m[finite_mask]

                                metrics["delta_r_track_cluster_genmatch"].extend(dr.tolist())
                                metrics["delta_eta_track_cluster_genmatch"].extend(deta.tolist())
                                metrics["delta_phi_track_cluster_genmatch"].extend(dphi.tolist())

                                prof_mask = np.isfinite(tr_p_m) & np.isfinite(tr_eta_m)
                                if np.any(prof_mask):
                                    p_prof = np.clip(tr_p_m[prof_mask], 0.0, np.inf)
                                    abseta_prof = np.abs(tr_eta_m[prof_mask])
                                    dr_prof = dr[prof_mask]

                                    tail002 = (dr_prof > 0.02).astype(float)
                                    tail005 = (dr_prof > 0.05).astype(float)

                                    metrics["dr_genmatch_profile_track_p_x"].extend(p_prof.tolist())
                                    metrics["dr_genmatch_profile_track_p_y"].extend(dr_prof.tolist())
                                    metrics["dr_genmatch_profile_abseta_x"].extend(abseta_prof.tolist())
                                    metrics["dr_genmatch_profile_abseta_y"].extend(dr_prof.tolist())

                                    metrics["dr_genmatch_tail002_profile_track_p_x"].extend(p_prof.tolist())
                                    metrics["dr_genmatch_tail002_profile_track_p_y"].extend(tail002.tolist())
                                    metrics["dr_genmatch_tail005_profile_track_p_x"].extend(p_prof.tolist())
                                    metrics["dr_genmatch_tail005_profile_track_p_y"].extend(tail005.tolist())

                                    metrics["dr_genmatch_tail002_profile_abseta_x"].extend(abseta_prof.tolist())
                                    metrics["dr_genmatch_tail002_profile_abseta_y"].extend(tail002.tolist())
                                    metrics["dr_genmatch_tail005_profile_abseta_x"].extend(abseta_prof.tolist())
                                    metrics["dr_genmatch_tail005_profile_abseta_y"].extend(tail005.tolist())

                                particle_energy_mask = (
                                    np.isfinite(particle_energy_m) & (particle_energy_m > 0.0)
                                )
                                if np.any(particle_energy_mask):
                                    particle_energy_prof = particle_energy_m[particle_energy_mask]
                                    dr_particle_prof = dr[particle_energy_mask]
                                    tail002_particle = (dr_particle_prof > 0.02).astype(float)
                                    tail005_particle = (dr_particle_prof > 0.05).astype(float)

                                    metrics["dr_genmatch_profile_particle_energy_x"].extend(
                                        particle_energy_prof.tolist()
                                    )
                                    metrics["dr_genmatch_profile_particle_energy_y"].extend(
                                        dr_particle_prof.tolist()
                                    )

                                    metrics["dr_genmatch_tail002_profile_particle_energy_x"].extend(
                                        particle_energy_prof.tolist()
                                    )
                                    metrics["dr_genmatch_tail002_profile_particle_energy_y"].extend(
                                        tail002_particle.tolist()
                                    )
                                    metrics["dr_genmatch_tail005_profile_particle_energy_x"].extend(
                                        particle_energy_prof.tolist()
                                    )
                                    metrics["dr_genmatch_tail005_profile_particle_energy_y"].extend(
                                        tail005_particle.tolist()
                                    )

                                cluster_energy_mask = (
                                    np.isfinite(cluster_energy_m) & (cluster_energy_m > 0.0)
                                )
                                if np.any(cluster_energy_mask):
                                    cluster_energy_prof = cluster_energy_m[cluster_energy_mask]
                                    dr_cluster_prof = dr[cluster_energy_mask]
                                    tail002_cluster = (dr_cluster_prof > 0.02).astype(float)
                                    tail005_cluster = (dr_cluster_prof > 0.05).astype(float)

                                    metrics["dr_genmatch_profile_cluster_energy_x"].extend(
                                        cluster_energy_prof.tolist()
                                    )
                                    metrics["dr_genmatch_profile_cluster_energy_y"].extend(
                                        dr_cluster_prof.tolist()
                                    )

                                    metrics["dr_genmatch_tail002_profile_cluster_energy_x"].extend(
                                        cluster_energy_prof.tolist()
                                    )
                                    metrics["dr_genmatch_tail002_profile_cluster_energy_y"].extend(
                                        tail002_cluster.tolist()
                                    )
                                    metrics["dr_genmatch_tail005_profile_cluster_energy_x"].extend(
                                        cluster_energy_prof.tolist()
                                    )
                                    metrics["dr_genmatch_tail005_profile_cluster_energy_y"].extend(
                                        tail005_cluster.tolist()
                                    )

    return metrics, stats


def _process_geometry_job(
    geom: str,
    files: Sequence[str],
    max_events_per_file: Optional[int],
    log_every: int,
) -> Tuple[str, Dict[str, List[float]], GeometryStats]:
    metrics, stats = _process_geometry(
        geom=geom,
        files=[Path(f) for f in files],
        max_events_per_file=max_events_per_file,
        log_every=log_every,
    )
    return geom, metrics, stats


def _metric_stats(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _compute_plot_bins(
    per_geom_values: Mapping[str, np.ndarray],
    spec: Mapping[str, object],
):
    fixed_range = spec.get("fixed_range")
    bins = int(spec.get("bins", 60))
    clip_q = float(spec.get("clip_q", 0.995))
    is_discrete = bool(spec.get("discrete", False))

    combined_parts = [arr for arr in per_geom_values.values() if arr.size > 0]
    if not combined_parts:
        return None

    combined = np.concatenate(combined_parts)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return None

    if fixed_range is not None:
        lo, hi = float(fixed_range[0]), float(fixed_range[1])
    else:
        lo = float(np.min(combined))
        hi = float(np.quantile(combined, clip_q))
        if not np.isfinite(hi) or hi <= lo:
            hi = float(np.max(combined))
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

    if is_discrete:
        lo_i = int(math.floor(lo))
        hi_i = int(math.ceil(hi))
        if hi_i <= lo_i:
            hi_i = lo_i + 1
        n_bins = hi_i - lo_i + 1
        if n_bins > 140:
            edges = np.linspace(lo_i - 0.5, hi_i + 0.5, 141)
        else:
            edges = np.arange(lo_i - 0.5, hi_i + 1.5, 1.0)
        return edges

    return {"bins": bins, "range": (lo, hi)}


def _normalized_hist(values: np.ndarray, hist_cfg):
    if isinstance(hist_cfg, np.ndarray):
        counts, edges = np.histogram(values, bins=hist_cfg)
    else:
        counts, edges = np.histogram(values, bins=hist_cfg["bins"], range=hist_cfg["range"])
    counts = counts.astype(float)
    s = float(np.sum(counts))
    if s > 0.0:
        counts /= s
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def _plot_metric_with_ratio(
    metric_key: str,
    spec: Mapping[str, object],
    per_geom_values: Mapping[str, np.ndarray],
    output_dir: Path,
    dpi: int,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_cfg = _compute_plot_bins(per_geom_values, spec)
    if hist_cfg is None:
        fig = plt.figure(figsize=(9, 4))
        plt.text(
            0.5,
            0.5,
            f"No valid data for: {spec['title']}",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        out_name = f"{metric_key}_ratio.png"
        out_path = output_dir / out_name
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    hists = {}
    centers_ref = None
    for geom in GEOM_ORDER:
        vals = per_geom_values[geom]
        if vals.size == 0:
            centers = None
            hist = np.array([], dtype=float)
        else:
            centers, hist = _normalized_hist(vals, hist_cfg)
            centers_ref = centers if centers_ref is None else centers_ref
        hists[geom] = hist

    if centers_ref is None or hists["CLD"].size == 0:
        return None

    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.06},
    )

    for geom in GEOM_ORDER:
        hist = hists[geom]
        if hist.size == 0:
            continue
        ax_top.step(centers_ref, hist, where="mid", linewidth=1.8, color=GEOM_COLORS[geom], label=geom)

    ax_top.set_ylabel("Normalized Counts")
    ax_top.set_title(str(spec["title"]))
    if bool(spec.get("log_y", False)):
        pos_parts = [hist[hist > 0.0] for hist in hists.values() if hist.size > 0]
        pos = np.concatenate(pos_parts) if pos_parts else np.empty(0, dtype=float)
        if pos.size > 0:
            ax_top.set_yscale("log")
            y_min = max(float(np.min(pos)) * 0.7, 1e-6)
            y_max = max(float(np.max(pos)) * 1.3, y_min * 10.0)
            ax_top.set_ylim(y_min, y_max)
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="best")

    cld = hists["CLD"]
    ratio_curves = []
    for geom, label in [("ARC", "ARC / CLD"), ("ARC_modified", "ARC_modified / CLD")]:
        if hists[geom].size == 0:
            continue
        ratio = np.divide(
            hists[geom],
            cld,
            out=np.full_like(cld, np.nan, dtype=float),
            where=cld > 0,
        )
        ratio_curves.append(ratio)
        ax_ratio.step(centers_ref, ratio, where="mid", linewidth=1.6, color=GEOM_COLORS[geom], label=label)

    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_ratio.set_xlabel(str(spec["xlabel"]))
    ax_ratio.set_ylabel("Ratio to CLD")
    ax_ratio.grid(alpha=0.25)

    finite_ratio = np.concatenate([r[np.isfinite(r)] for r in ratio_curves]) if ratio_curves else np.empty(0)
    if finite_ratio.size > 0:
        y_max = float(np.quantile(finite_ratio, 0.98))
        y_max = max(1.3, min(5.0, y_max * 1.25))
        ax_ratio.set_ylim(0.0, y_max)
    else:
        ax_ratio.set_ylim(0.0, 2.0)

    if ratio_curves:
        ax_ratio.legend(loc="best")

    out_name = f"{metric_key}_ratio.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_profile_bin_edges(x_all: np.ndarray, *, mode: str) -> Optional[np.ndarray]:
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size < 20:
        return None

    if mode == "track_p":
        x = np.clip(x_all, 0.0, np.inf)
        if np.all(x == 0):
            return None
        q = np.quantile(x, np.linspace(0.0, 1.0, 26))
        edges = np.unique(q)
    elif mode == "abs_eta":
        hi = float(np.quantile(np.clip(x_all, 0.0, np.inf), 0.995))
        hi = min(max(hi, 1.5), 4.0)
        edges = np.linspace(0.0, hi, 25)
    elif mode == "energy":
        x = np.clip(x_all, 0.0, np.inf)
        x = x[x > 0.0]
        if x.size < 20:
            return None
        q = np.quantile(x, np.linspace(0.0, 1.0, 26))
        edges = np.unique(q)
        if edges.size < 4:
            lo = float(np.min(x))
            hi = float(np.quantile(x, 0.995))
            if not np.isfinite(hi) or hi <= lo:
                hi = float(np.max(x))
            if not np.isfinite(hi) or hi <= lo:
                hi = lo * 1.1
            edges = np.linspace(lo, hi, 25)
    else:
        return None

    if edges.size < 4 or not np.all(np.isfinite(edges)):
        return None
    if np.any(np.diff(edges) <= 0):
        edges = np.unique(edges)
    if edges.size < 4:
        return None
    return edges


def _profile_stat_in_bins(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    *,
    stat: str = "median",
    min_count: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    idx = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(edges.size - 1, np.nan, dtype=float)
    cnts = np.zeros(edges.size - 1, dtype=int)
    for ib in range(edges.size - 1):
        m = idx == ib
        n = int(np.sum(m))
        cnts[ib] = n
        if n >= min_count:
            if stat == "median":
                vals[ib] = float(np.median(y[m]))
            elif stat == "mean":
                vals[ib] = float(np.mean(y[m]))
            else:
                raise ValueError(f"Unbekannter Profil-Statistikmodus: {stat}")
    return centers, vals, cnts


def _plot_profile_with_ratio(
    *,
    profile_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    mode: str,
    stat: str,
    per_geom_x: Mapping[str, np.ndarray],
    per_geom_y: Mapping[str, np.ndarray],
    output_dir: Path,
    dpi: int,
    log_x: bool = False,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_all = np.concatenate([arr for arr in per_geom_x.values() if arr.size > 0]) if any(
        arr.size > 0 for arr in per_geom_x.values()
    ) else np.empty(0)
    edges = _build_profile_bin_edges(x_all, mode=mode)
    out_path = output_dir / f"{profile_key}_ratio.png"
    if edges is None:
        fig = plt.figure(figsize=(9, 4))
        plt.text(0.5, 0.5, f"No valid data for: {title}", ha="center", va="center", fontsize=12)
        plt.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    prof_vals = {}
    centers_ref = None
    for geom in GEOM_ORDER:
        centers, vals, _ = _profile_stat_in_bins(per_geom_x[geom], per_geom_y[geom], edges, stat=stat)
        centers_ref = centers
        prof_vals[geom] = vals

    cld = prof_vals["CLD"]
    if cld.size == 0:
        return None

    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.06},
    )
    for geom in GEOM_ORDER:
        y = prof_vals[geom]
        if y.size == 0:
            continue
        ax_top.plot(centers_ref, y, linewidth=1.8, color=GEOM_COLORS[geom], label=geom)
    if log_x and np.all(centers_ref > 0.0):
        ax_top.set_xscale("log")
        ax_ratio.set_xscale("log")
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(title)
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="best")

    ratio_curves = []
    for geom, label in [("ARC", "ARC / CLD"), ("ARC_modified", "ARC_modified / CLD")]:
        y = prof_vals[geom]
        ratio = np.divide(y, cld, out=np.full_like(cld, np.nan, dtype=float), where=np.isfinite(cld) & (cld > 0))
        ratio_curves.append(ratio)
        ax_ratio.plot(centers_ref, ratio, linewidth=1.6, color=GEOM_COLORS[geom], label=label)

    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel("Ratio to CLD")
    ax_ratio.grid(alpha=0.25)
    finite_ratio = np.concatenate([r[np.isfinite(r)] for r in ratio_curves]) if ratio_curves else np.empty(0)
    if finite_ratio.size > 0:
        y_max = float(np.quantile(finite_ratio, 0.98))
        y_max = max(1.3, min(5.0, y_max * 1.25))
        ax_ratio.set_ylim(0.0, y_max)
    else:
        ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.legend(loc="best")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(f"Nicht JSON-serialisierbar: {type(obj)}")


def main() -> int:
    args = parse_args()
    ensure_parquet_backend()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise RuntimeError(f"Fehler: dataset-root existiert nicht: {dataset_root}")

    output_dir = Path(args.output_dir) if args.output_dir else (dataset_root / "comparison_plots_ratio")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_maps = _find_geometry_files(dataset_root)
    common_indices = _common_file_indices(file_maps)
    if not common_indices:
        raise RuntimeError("Fehler: Keine gemeinsamen pf_tree-Indizes zwischen 05/arc/arc_modified gefunden.")

    if args.max_files is not None:
        common_indices = common_indices[: max(0, args.max_files)]

    if not common_indices:
        raise RuntimeError("Fehler: Nach --max-files bleiben keine Dateien uebrig.")

    files_per_geom = {
        geom: [file_maps[geom][idx] for idx in common_indices]
        for geom in GEOM_ORDER
    }

    all_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    all_profiles: Dict[str, Dict[str, np.ndarray]] = {}
    geometry_summaries = {}
    warnings: List[str] = []

    geometry_results: Dict[str, Tuple[Dict[str, List[float]], GeometryStats]] = {}

    if args.workers > 1:
        n_workers = max(1, min(int(args.workers), len(GEOM_ORDER)))
        print(f"Starte Parallelverarbeitung ueber Geometrien mit workers={n_workers}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(
                    _process_geometry_job,
                    geom,
                    [str(p) for p in files_per_geom[geom]],
                    args.max_events_per_file,
                    int(args.log_every),
                ): geom
                for geom in GEOM_ORDER
            }
            for fut in concurrent.futures.as_completed(future_map):
                geom = future_map[fut]
                try:
                    geom_out, metrics_raw, stats = fut.result()
                    geometry_results[geom_out] = (metrics_raw, stats)
                except Exception as exc:
                    msg = f"Parallelverarbeitung fehlgeschlagen fuer '{geom}': {exc}"
                    print(f"WARN: {msg}")
                    warnings.append(msg)
                    geometry_results[geom] = (_empty_metrics_dict(), GeometryStats(files_failed=len(files_per_geom[geom])))
    else:
        for geom in GEOM_ORDER:
            print(f"Verarbeite {geom}: {len(files_per_geom[geom])} Dateien")
            metrics_raw, stats = _process_geometry(
                geom,
                files_per_geom[geom],
                args.max_events_per_file,
                log_every=int(args.log_every),
            )
            geometry_results[geom] = (metrics_raw, stats)

    for geom in GEOM_ORDER:
        metrics_raw, stats = geometry_results[geom]

        converted = {}
        for spec in METRIC_SPECS:
            key = spec["key"]
            converted[key] = _safe_nan_array(
                metrics_raw.get(key, []),
                non_negative=bool(spec.get("non_negative", False)),
            )
        all_metrics[geom] = converted

        profile_dict = {
            key: _safe_nan_array(metrics_raw.get(key, []), non_negative=True)
            for key in INTERNAL_PROFILE_KEYS
        }
        for x_key, y_key in PROFILE_ARRAY_PAIRS:
            n = min(profile_dict[x_key].size, profile_dict[y_key].size)
            profile_dict[x_key] = profile_dict[x_key][:n]
            profile_dict[y_key] = profile_dict[y_key][:n]
        all_profiles[geom] = profile_dict

        geometry_summaries[geom] = {
            "files_requested": len(files_per_geom[geom]),
            "files_processed": stats.files_processed,
            "files_failed": stats.files_failed,
            "events_read": stats.events_read,
            "events_processed": stats.events_processed,
            "events_skipped": stats.events_skipped,
            "missing_fields": dict(stats.missing_fields),
        }

    generated_plots: List[Path] = []
    metrics_summary = {}

    for spec in METRIC_SPECS:
        key = spec["key"]
        per_geom = {geom: all_metrics[geom][key] for geom in GEOM_ORDER}

        try:
            plot_path = _plot_metric_with_ratio(
                metric_key=key,
                spec=spec,
                per_geom_values=per_geom,
                output_dir=output_dir,
                dpi=args.dpi,
            )
            if plot_path is not None:
                generated_plots.append(plot_path)
        except Exception as exc:
            msg = f"Plot fehlgeschlagen fuer '{key}': {exc}"
            print(f"WARN: {msg}")
            warnings.append(msg)

        metric_entry = {geom: _metric_stats(arr) for geom, arr in per_geom.items()}
        cld_mean = metric_entry["CLD"]["mean"]
        ratio_means = {}
        for geom in ["ARC", "ARC_modified"]:
            mean_val = metric_entry[geom]["mean"]
            if np.isfinite(cld_mean) and cld_mean != 0.0 and np.isfinite(mean_val):
                ratio_means[geom] = float(mean_val / cld_mean)
            else:
                ratio_means[geom] = float("nan")
        metric_entry["mean_ratio_to_CLD"] = ratio_means
        metrics_summary[key] = metric_entry

    profile_specs = [
        {
            "profile_key": "delta_r_genmatch_vs_track_p",
            "title": r"Track-Cluster $\Delta R$ (gen-match) vs track $p$",
            "xlabel": r"$p$ (a.u.)",
            "ylabel": r"Median $\Delta R$",
            "mode": "track_p",
            "stat": "median",
            "x_key": "dr_genmatch_profile_track_p_x",
            "y_key": "dr_genmatch_profile_track_p_y",
        },
        {
            "profile_key": "delta_r_genmatch_vs_abs_eta",
            "title": r"Track-Cluster $\Delta R$ (gen-match) vs $|\eta|$",
            "xlabel": r"$|\eta|$",
            "ylabel": r"Median $\Delta R$",
            "mode": "abs_eta",
            "stat": "median",
            "x_key": "dr_genmatch_profile_abseta_x",
            "y_key": "dr_genmatch_profile_abseta_y",
        },
        {
            "profile_key": "delta_r_genmatch_vs_particle_energy",
            "title": r"Track-Cluster $\Delta R$ (gen-match) vs particle energy",
            "xlabel": r"Particle energy (a.u.)",
            "ylabel": r"Median $\Delta R$",
            "mode": "energy",
            "stat": "median",
            "x_key": "dr_genmatch_profile_particle_energy_x",
            "y_key": "dr_genmatch_profile_particle_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "delta_r_genmatch_vs_cluster_energy",
            "title": r"Track-Cluster $\Delta R$ (gen-match) vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"Median $\Delta R$",
            "mode": "energy",
            "stat": "median",
            "x_key": "dr_genmatch_profile_cluster_energy_x",
            "y_key": "dr_genmatch_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p02_vs_track_p",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.02)$ vs track $p$",
            "xlabel": r"$p$ (a.u.)",
            "ylabel": r"$P(\Delta R > 0.02)$",
            "mode": "track_p",
            "stat": "mean",
            "x_key": "dr_genmatch_tail002_profile_track_p_x",
            "y_key": "dr_genmatch_tail002_profile_track_p_y",
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p05_vs_track_p",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.05)$ vs track $p$",
            "xlabel": r"$p$ (a.u.)",
            "ylabel": r"$P(\Delta R > 0.05)$",
            "mode": "track_p",
            "stat": "mean",
            "x_key": "dr_genmatch_tail005_profile_track_p_x",
            "y_key": "dr_genmatch_tail005_profile_track_p_y",
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p02_vs_particle_energy",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.02)$ vs particle energy",
            "xlabel": r"Particle energy (a.u.)",
            "ylabel": r"$P(\Delta R > 0.02)$",
            "mode": "energy",
            "stat": "mean",
            "x_key": "dr_genmatch_tail002_profile_particle_energy_x",
            "y_key": "dr_genmatch_tail002_profile_particle_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p05_vs_particle_energy",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.05)$ vs particle energy",
            "xlabel": r"Particle energy (a.u.)",
            "ylabel": r"$P(\Delta R > 0.05)$",
            "mode": "energy",
            "stat": "mean",
            "x_key": "dr_genmatch_tail005_profile_particle_energy_x",
            "y_key": "dr_genmatch_tail005_profile_particle_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p02_vs_cluster_energy",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.02)$ vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"$P(\Delta R > 0.02)$",
            "mode": "energy",
            "stat": "mean",
            "x_key": "dr_genmatch_tail002_profile_cluster_energy_x",
            "y_key": "dr_genmatch_tail002_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p05_vs_cluster_energy",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.05)$ vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"$P(\Delta R > 0.05)$",
            "mode": "energy",
            "stat": "mean",
            "x_key": "dr_genmatch_tail005_profile_cluster_energy_x",
            "y_key": "dr_genmatch_tail005_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p02_vs_abs_eta",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.02)$ vs $|\eta|$",
            "xlabel": r"$|\eta|$",
            "ylabel": r"$P(\Delta R > 0.02)$",
            "mode": "abs_eta",
            "stat": "mean",
            "x_key": "dr_genmatch_tail002_profile_abseta_x",
            "y_key": "dr_genmatch_tail002_profile_abseta_y",
        },
        {
            "profile_key": "frac_delta_r_genmatch_gt_0p05_vs_abs_eta",
            "title": r"$P(\Delta R_{\mathrm{gen-match}} > 0.05)$ vs $|\eta|$",
            "xlabel": r"$|\eta|$",
            "ylabel": r"$P(\Delta R > 0.05)$",
            "mode": "abs_eta",
            "stat": "mean",
            "x_key": "dr_genmatch_tail005_profile_abseta_x",
            "y_key": "dr_genmatch_tail005_profile_abseta_y",
        },
        {
            "profile_key": "cluster_size_vs_cluster_energy",
            "title": r"Cluster size vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"Median hits per cluster",
            "mode": "energy",
            "stat": "median",
            "x_key": "cluster_size_profile_cluster_energy_x",
            "y_key": "cluster_size_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "cluster_r68_hits_vs_cluster_energy",
            "title": r"Cluster $R_{68}^{\mathrm{hits}}$ vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"Median $R_{68}^{\mathrm{hits}}$",
            "mode": "energy",
            "stat": "median",
            "x_key": "cluster_r68_hits_profile_cluster_energy_x",
            "y_key": "cluster_r68_hits_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "cluster_r90_hits_vs_cluster_energy",
            "title": r"Cluster $R_{90}^{\mathrm{hits}}$ vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"Median $R_{90}^{\mathrm{hits}}$",
            "mode": "energy",
            "stat": "median",
            "x_key": "cluster_r90_hits_profile_cluster_energy_x",
            "y_key": "cluster_r90_hits_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "cluster_r68_energy_vs_cluster_energy",
            "title": r"Cluster $R_{68}^{E}$ vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"Median $R_{68}^{E}$",
            "mode": "energy",
            "stat": "median",
            "x_key": "cluster_r68_energy_profile_cluster_energy_x",
            "y_key": "cluster_r68_energy_profile_cluster_energy_y",
            "log_x": True,
        },
        {
            "profile_key": "cluster_r90_energy_vs_cluster_energy",
            "title": r"Cluster $R_{90}^{E}$ vs cluster energy",
            "xlabel": r"Cluster energy (a.u.)",
            "ylabel": r"Median $R_{90}^{E}$",
            "mode": "energy",
            "stat": "median",
            "x_key": "cluster_r90_energy_profile_cluster_energy_x",
            "y_key": "cluster_r90_energy_profile_cluster_energy_y",
            "log_x": True,
        },
    ]
    profile_summary = {}
    for pspec in profile_specs:
        per_geom_x = {geom: all_profiles[geom][pspec["x_key"]] for geom in GEOM_ORDER}
        per_geom_y = {geom: all_profiles[geom][pspec["y_key"]] for geom in GEOM_ORDER}
        try:
            plot_path = _plot_profile_with_ratio(
                profile_key=pspec["profile_key"],
                title=pspec["title"],
                xlabel=pspec["xlabel"],
                ylabel=pspec["ylabel"],
                mode=pspec["mode"],
                stat=pspec["stat"],
                per_geom_x=per_geom_x,
                per_geom_y=per_geom_y,
                output_dir=output_dir,
                dpi=args.dpi,
                log_x=bool(pspec.get("log_x", False)),
            )
            if plot_path is not None:
                generated_plots.append(plot_path)
        except Exception as exc:
            msg = f"Profil-Plot fehlgeschlagen fuer '{pspec['profile_key']}': {exc}"
            print(f"WARN: {msg}")
            warnings.append(msg)
        profile_summary[pspec["profile_key"]] = {
            geom: int(min(per_geom_x[geom].size, per_geom_y[geom].size)) for geom in GEOM_ORDER
        }

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "max_files": args.max_files,
        "max_events_per_file": args.max_events_per_file,
        "dpi": args.dpi,
        "n_common_files_total": len(_common_file_indices(file_maps)),
        "n_common_files_used": len(common_indices),
        "common_file_indices_used": common_indices,
        "files_used": {
            geom: [str(p) for p in files_per_geom[geom]]
            for geom in GEOM_ORDER
        },
        "geometry_summary": geometry_summaries,
        "metrics": metrics_summary,
        "profile_pairs": profile_summary,
        "generated_plots": [str(p) for p in generated_plots],
        "warnings": warnings,
    }

    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"Fertig. Plots: {len(generated_plots)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Abbruch durch Benutzer (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as exc:
        msg = str(exc)
        print(msg if msg.startswith("Fehler:") else f"Fehler: {msg}")
        print("Das Skript beendet sich ohne Exit-Code 1, damit das Terminal offen bleibt.")
        sys.exit(0)
