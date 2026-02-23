import os

import awkward as ak
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.colors as pc
from plotly.subplots import make_subplots


def _stack_xyz(x, y, z):
    if len(x) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.stack([x, y, z], axis=1).astype(float)


def _eta_phi_from_xyz(xyz):
    if len(xyz) == 0:
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


def _phi_from_sin_cos(sin_phi, cos_phi):
    return np.arctan2(sin_phi, cos_phi)


def _choose_eta_phi(hit_eta, hit_phi, hit_xyz, label="hits"):
    if len(hit_eta) == 0:
        return hit_eta, hit_phi, "stored"
    eta_std = float(np.std(hit_eta))
    phi_std = float(np.std(hit_phi))
    if eta_std < 1e-6 and phi_std < 1e-6:
        eta_xyz, phi_xyz = _eta_phi_from_xyz(hit_xyz)
        print(f"{label}: eta/phi look constant, using XYZ-derived values")
        return eta_xyz, phi_xyz, "xyz"
    return hit_eta, hit_phi, "stored"


def _wrap_phi(dphi):
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _build_color_map(gen_ids):
    colors = pc.qualitative.Dark24
    unique_ids = list(dict.fromkeys(gen_ids))
    return {gid: colors[i % len(colors)] for i, gid in enumerate(unique_ids)}


def _genlink_label(gen_id, gen_pdg):
    if gen_id < 0:
        return "unlinked"
    if gen_pdg is not None and gen_id < len(gen_pdg):
        return f"gen {int(gen_id)} (PDG {int(gen_pdg[gen_id])})"
    return f"gen {int(gen_id)}"


def _build_color_map(gen_ids):
    colors = pc.qualitative.Dark24
    unique_ids = list(dict.fromkeys(gen_ids))
    return {gid: colors[i % len(colors)] for i, gid in enumerate(unique_ids)}


def _genlink_label(gen_id, gen_pdg):
    if gen_id < 0:
        return "unlinked"
    if gen_pdg is not None and gen_id < len(gen_pdg):
        return f"gen {int(gen_id)} (PDG {int(gen_pdg[gen_id])})"
    return f"gen {int(gen_id)}"


def _calo_mask(hit_type, calo_ids=(1, 2)):
    if len(hit_type) == 0:
        return np.zeros(0, dtype=bool)
    return np.isin(hit_type, np.array(calo_ids, dtype=hit_type.dtype))


def _list_parquet_files(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        return sorted(files)
    return [path]


def _resolve_parquet_file(path, file_index=0):
    files = _list_parquet_files(path)
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in {path}")
    if file_index < 0 or file_index >= len(files):
        raise IndexError(f"file_index {file_index} out of range")
    return files[file_index]


def load_event_arrays(parquet_path, event_index=0, file_index=0):
    parquet_file = _resolve_parquet_file(parquet_path, file_index=file_index)
    output = ak.from_parquet(parquet_file)
    if event_index < 0 or event_index >= len(output["X_hit"]):
        raise IndexError(f"event_index {event_index} out of range")

    X_hit = output["X_hit"][event_index]
    X_track = output["X_track"][event_index]

    hit_x = np.array(X_hit[:, 6])
    hit_y = np.array(X_hit[:, 7])
    hit_z = np.array(X_hit[:, 8])
    hit_e = np.array(X_hit[:, 5])
    hit_type = np.array(X_hit[:, -2]) + 1
    hit_genlink = np.array(output["ygen_hit"][event_index])
    hit_eta = np.array(X_hit[:, 2])
    hit_phi = _phi_from_sin_cos(np.array(X_hit[:, 3]), np.array(X_hit[:, 4]))

    track_x = np.array(X_track[:, 12])
    track_y = np.array(X_track[:, 13])
    track_z = np.array(X_track[:, 14])
    track_x_vtx = np.array(X_track[:, 9])
    track_y_vtx = np.array(X_track[:, 10])
    track_z_vtx = np.array(X_track[:, 11])
    track_genlink = np.array(output["ygen_track"][event_index])
    track_eta = np.array(X_track[:, 2])
    track_phi = _phi_from_sin_cos(np.array(X_track[:, 3]), np.array(X_track[:, 4]))

    hit_eta_used, hit_phi_used, hit_src = _choose_eta_phi(
        hit_eta, hit_phi, _stack_xyz(hit_x, hit_y, hit_z), label="hits"
    )
    track_eta_calo, track_phi_calo = _eta_phi_from_xyz(_stack_xyz(track_x, track_y, track_z))
    track_eta_used, track_phi_used, track_src = _choose_eta_phi(
        track_eta_calo, track_phi_calo, _stack_xyz(track_x, track_y, track_z), label="tracks"
    )

    gen_pdg = None
    if "X_gen" in output.fields:
        X_gen = output["X_gen"][event_index]
        if len(X_gen) > 0:
            gen_pdg = np.array(X_gen[:, 0])

    return {
        "hit_xyz": _stack_xyz(hit_x, hit_y, hit_z),
        "hit_e": hit_e,
        "hit_type": hit_type,
        "hit_genlink": hit_genlink,
        "hit_eta": hit_eta_used,
        "hit_phi": hit_phi_used,
        "track_xyz": _stack_xyz(track_x, track_y, track_z),
        "track_xyz_vtx": _stack_xyz(track_x_vtx, track_y_vtx, track_z_vtx),
        "track_genlink": track_genlink,
        "track_eta": track_eta_used,
        "track_phi": track_phi_used,
        "eta_source": {"hits": hit_src, "tracks": track_src},
        "gen_pdg": gen_pdg,
    }


def plot_event_3d(
    parquet_path,
    event_index=0,
    file_index=0,
    show=True,
    save_png=True,
    png_filename=None,
):
    data = load_event_arrays(parquet_path, event_index=event_index, file_index=file_index)
    hit_xyz = data["hit_xyz"]
    hit_e = data["hit_e"]
    hit_type = data["hit_type"]
    hit_genlink = data["hit_genlink"]
    track_xyz = data["track_xyz"]
    track_xyz_vtx = data["track_xyz_vtx"]
    track_genlink = data["track_genlink"]
    gen_pdg = data["gen_pdg"]

    fig = go.Figure()

    calo_mask = _calo_mask(hit_type)
    if np.any(calo_mask):
        hit_genlink_calo = hit_genlink[calo_mask]
        color_map = _build_color_map(hit_genlink_calo)
        for gen_id in np.unique(hit_genlink_calo):
            m = hit_genlink_calo == gen_id
            marker_sizes = 10 * hit_e[calo_mask][m] / hit_e[calo_mask][m]
            fig.add_trace(
                go.Scatter3d(
                    x=hit_xyz[calo_mask][m, 0],
                    y=hit_xyz[calo_mask][m, 1],
                    z=hit_xyz[calo_mask][m, 2],
                    mode="markers",
                    name=_genlink_label(int(gen_id), gen_pdg),
                    marker=dict(
                        size=marker_sizes,
                        color=color_map[gen_id],
                        opacity=0.8,
                        symbol="circle",
                        line=dict(width=0),
                    ),
                )
            )
    else:
        color_map = {}

    if len(track_xyz) > 0:
        if len(color_map) == 0:
            color_map = _build_color_map(track_genlink)
        for gen_id in np.unique(track_genlink):
            m = track_genlink == gen_id
            label = _genlink_label(int(gen_id), gen_pdg)
            # calo track point as cross with black outline
            fig.add_trace(
                go.Scatter3d(
                    x=track_xyz[m, 0],
                    y=track_xyz[m, 1],
                    z=track_xyz[m, 2],
                    mode="markers",
                    name=f"track: {label}",
                    showlegend=True,
                    marker=dict(
                        size=8,
                        color=color_map.get(gen_id, "#444444"),
                        symbol="cross",
                        line=dict(color="#222222", width=2),
                    ),
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
                color="lightgray",
                titlefont=dict(color="lightgray"),
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
                color="lightgray",
                titlefont=dict(color="lightgray"),
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
                color="lightgray",
                titlefont=dict(color="lightgray"),
            ),
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Event {event_index}: calo hits + tracks",
    )

    if save_png:
        if png_filename is None:
            png_filename = f"event_{event_index + 1:04d}_3d.png"
        fig.write_image(png_filename)

    if show:
        fig.show()

    return fig


def plot_eta_phi_clusters_tracks(
    parquet_path,
    event_index=0,
    file_index=0,
    show=True,
    save_png=True,
    png_filename=None,
):
    data = load_event_arrays(parquet_path, event_index=event_index, file_index=file_index)
    hit_eta = data["hit_eta"]
    hit_phi = data["hit_phi"]
    hit_type = data["hit_type"]
    hit_genlink = data["hit_genlink"]
    hit_xyz = data["hit_xyz"]
    hit_e = data["hit_e"]
    track_eta = data["track_eta"]
    track_phi = data["track_phi"]
    track_genlink = data["track_genlink"]
    gen_pdg = data["gen_pdg"]

    fig = go.Figure()

    calo_mask = _calo_mask(hit_type)
    if np.any(calo_mask):
        hit_genlink_calo = hit_genlink[calo_mask]
        color_map = _build_color_map(hit_genlink_calo)
        for gen_id in np.unique(hit_genlink_calo):
            m = hit_genlink_calo == gen_id
            fig.add_trace(
                go.Scatter(
                    x=hit_eta[calo_mask][m],
                    y=hit_phi[calo_mask][m],
                    mode="markers",
                    name=_genlink_label(int(gen_id), gen_pdg),
                    marker=dict(
                        size=6,
                        color=color_map[gen_id],
                        opacity=0.8,
                        symbol="circle",
                    ),
                )
            )
    else:
        color_map = {}

    centers = _cluster_centers_eta_phi(
        hit_eta,
        hit_phi,
        hit_genlink,
        hit_type,
        hit_xyz=hit_xyz,
        hit_e=hit_e,
    )
    for gen_id, (eta_c, phi_c) in centers.items():
        label = _genlink_label(int(gen_id), gen_pdg)
        fig.add_trace(
            go.Scatter(
                x=[eta_c],
                y=[phi_c],
                mode="markers",
                name=f"center: {label}",
                showlegend=False,
                marker=dict(size=7, color=color_map[gen_id], symbol="star", line=dict(color="#222222", width=1) ),
                
            )
        )

    if len(track_eta) > 0:
        if len(color_map) == 0:
            color_map = _build_color_map(track_genlink)
        for gen_id in np.unique(track_genlink):
            m = track_genlink == gen_id
            label = _genlink_label(int(gen_id), gen_pdg)
            fig.add_trace(
                go.Scatter(
                    x=track_eta[m],
                    y=track_phi[m],
                    mode="markers",
                    name=f"track: {label}",
                    showlegend=True,
                    marker=dict(
                        size=8,
                        color=color_map.get(gen_id, "#444444"),
                        symbol="cross",
                        line=dict(color="#222222", width=2),
                    ),
                )
            )

    fig.update_layout(
        xaxis_title="eta",
        yaxis_title="phi",
            title=f"Event {event_index}: calo clusters + tracks",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, b=50, t=40),
    )

    if save_png:
        if png_filename is None:
            png_filename = f"event_{event_index + 1:04d}_etaphi.png"
        fig.write_image(png_filename)

    if show:
        fig.show()

    return fig


def _cluster_centers_eta_phi(
    hit_eta, hit_phi, hit_genlink, hit_type, hit_xyz=None, hit_e=None
):
    lengths = [len(hit_eta), len(hit_phi), len(hit_genlink), len(hit_type)]
    if hit_xyz is not None:
        lengths.append(len(hit_xyz))
    if hit_e is not None:
        lengths.append(len(hit_e))
    n = min(lengths)
    hit_eta = hit_eta[:n]
    hit_phi = hit_phi[:n]
    hit_genlink = hit_genlink[:n]
    hit_type = hit_type[:n]
    if hit_xyz is not None:
        hit_xyz = hit_xyz[:n]
    if hit_e is not None:
        hit_e = hit_e[:n]

    calo_mask = _calo_mask(hit_type)
    if not np.any(calo_mask):
        return {}
    centers = {}
    hit_eta_calo = hit_eta[calo_mask]
    hit_phi_calo = hit_phi[calo_mask]
    hit_xyz_calo = hit_xyz[calo_mask] if hit_xyz is not None else None
    hit_e_calo = hit_e[calo_mask] if hit_e is not None else None
    gen_ids = hit_genlink[calo_mask]
    for gen_id in np.unique(gen_ids):
        m = gen_ids == gen_id
        if hit_e_calo is not None:
            w = hit_e_calo[m]
            wsum = float(np.sum(w))
        else:
            w = None
            wsum = 0.0
        if hit_xyz_calo is not None:
            xyz = hit_xyz_calo[m]
            if w is not None and wsum > 0.0:
                center_xyz = np.sum(xyz * w[:, None], axis=0) / wsum
            else:
                center_xyz = np.mean(xyz, axis=0)
            eta_center, phi_center = _eta_phi_from_xyz(center_xyz[None, :])
            eta_center = float(eta_center[0])
            phi_center = float(phi_center[0])
        elif w is not None and wsum > 0.0:
            eta_center = float(np.sum(w * hit_eta_calo[m]) / wsum)
            phi_center = float(
                np.arctan2(
                    np.sum(w * np.sin(hit_phi_calo[m])),
                    np.sum(w * np.cos(hit_phi_calo[m])),
                )
            )
        else:
            eta_center = float(np.mean(hit_eta_calo[m]))
            phi_center = float(
                np.arctan2(np.mean(np.sin(hit_phi_calo[m])), np.mean(np.cos(hit_phi_calo[m])))
            )
        centers[int(gen_id)] = (eta_center, phi_center)
    return centers


def _cluster_centers_eta_phi_with_energy(
    hit_eta, hit_phi, hit_genlink, hit_type, hit_xyz=None, hit_e=None
):
    lengths = [len(hit_eta), len(hit_phi), len(hit_genlink), len(hit_type)]
    if hit_xyz is not None:
        lengths.append(len(hit_xyz))
    if hit_e is not None:
        lengths.append(len(hit_e))
    n = min(lengths)
    hit_eta = hit_eta[:n]
    hit_phi = hit_phi[:n]
    hit_genlink = hit_genlink[:n]
    hit_type = hit_type[:n]
    if hit_xyz is not None:
        hit_xyz = hit_xyz[:n]
    if hit_e is not None:
        hit_e = hit_e[:n]

    calo_mask = _calo_mask(hit_type)
    if not np.any(calo_mask):
        return {}
    centers = {}
    hit_eta_calo = hit_eta[calo_mask]
    hit_phi_calo = hit_phi[calo_mask]
    hit_xyz_calo = hit_xyz[calo_mask] if hit_xyz is not None else None
    hit_e_calo = hit_e[calo_mask] if hit_e is not None else None
    gen_ids = hit_genlink[calo_mask]
    for gen_id in np.unique(gen_ids):
        m = gen_ids == gen_id
        if hit_e_calo is not None:
            w = hit_e_calo[m]
            wsum = float(np.sum(w))
            e_sum = float(wsum)
        else:
            w = None
            wsum = 0.0
            e_sum = float(np.sum(hit_e_calo[m])) if hit_e_calo is not None else 0.0
        if hit_xyz_calo is not None:
            xyz = hit_xyz_calo[m]
            if w is not None and wsum > 0.0:
                center_xyz = np.sum(xyz * w[:, None], axis=0) / wsum
            else:
                center_xyz = np.mean(xyz, axis=0)
            eta_center, phi_center = _eta_phi_from_xyz(center_xyz[None, :])
            eta_center = float(eta_center[0])
            phi_center = float(phi_center[0])
        elif w is not None and wsum > 0.0:
            eta_center = float(np.sum(w * hit_eta_calo[m]) / wsum)
            phi_center = float(
                np.arctan2(
                    np.sum(w * np.sin(hit_phi_calo[m])),
                    np.sum(w * np.cos(hit_phi_calo[m])),
                )
            )
        else:
            eta_center = float(np.mean(hit_eta_calo[m]))
            phi_center = float(
                np.arctan2(np.mean(np.sin(hit_phi_calo[m])), np.mean(np.cos(hit_phi_calo[m])))
            )
        centers[int(gen_id)] = (eta_center, phi_center, e_sum)
    return centers


def _tracks_eta_phi_by_genlink(track_eta, track_phi, track_genlink):
    tracks = {}
    for gen_id in np.unique(track_genlink):
        m = track_genlink == gen_id
        tracks[int(gen_id)] = (track_eta[m], track_phi[m])
    return tracks


def _align_hit_arrays(hit_eta, hit_phi, hit_genlink, hit_type):
    n = min(len(hit_eta), len(hit_phi), len(hit_genlink), len(hit_type))
    return hit_eta[:n], hit_phi[:n], hit_genlink[:n], hit_type[:n]


def _align_track_arrays(track_eta, track_phi, track_genlink):
    n = min(len(track_eta), len(track_phi), len(track_genlink))
    return track_eta[:n], track_phi[:n], track_genlink[:n]

def _residual_summary(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q68": np.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q68": float(np.quantile(arr, 0.68)),
    }
    
def compute_event_residuals(data):
    hit_eta, hit_phi, hit_genlink, hit_type = _align_hit_arrays(
        data["hit_eta"],
        data["hit_phi"],
        data["hit_genlink"],
        data["hit_type"],
    )
    hit_xyz = data.get("hit_xyz")
    hit_e = data.get("hit_e")
    track_eta, track_phi, track_genlink = _align_track_arrays(
        data["track_eta"],
        data["track_phi"],
        data["track_genlink"],
    )

    centers = _cluster_centers_eta_phi(
        hit_eta, hit_phi, hit_genlink, hit_type, hit_xyz=hit_xyz, hit_e=hit_e
    )
    tracks = _tracks_eta_phi_by_genlink(track_eta, track_phi, track_genlink)
    centers = {gid: val for gid, val in centers.items() if gid >= 0}
    tracks = {gid: val for gid, val in tracks.items() if gid >= 0}
    valid_ids = set(centers.keys()) & set(tracks.keys())

    residuals = []
    for gen_id in valid_ids:
        eta_c, phi_c = centers[gen_id]
        tr_eta, tr_phi = tracks[gen_id]
        if len(tr_eta) == 0:
            continue
        deta = tr_eta - eta_c
        dphi = _wrap_phi(tr_phi - phi_c)
        delta_r = np.sqrt(deta**2 + dphi**2)
        residuals.append(float(np.min(delta_r)))
    return residuals


def compute_residuals_over_events(parquet_path, max_events=200):
    files = _list_parquet_files(parquet_path)
    all_res = []
    remaining = max_events

    for parquet_file in files:
        if remaining <= 0:
            break
        output = ak.from_parquet(parquet_file)
        n_events = len(output["X_hit"])
        take = min(remaining, n_events)
        for iev in range(take):
            track_xyz_calo = _stack_xyz(
                np.array(output["X_track"][iev][:, 12]),
                np.array(output["X_track"][iev][:, 13]),
                np.array(output["X_track"][iev][:, 14]),
            )
            track_eta_calo, track_phi_calo = _eta_phi_from_xyz(track_xyz_calo)
            hit_xyz = _stack_xyz(
                np.array(output["X_hit"][iev][:, 6]),
                np.array(output["X_hit"][iev][:, 7]),
                np.array(output["X_hit"][iev][:, 8]),
            )
            hit_eta = np.array(output["X_hit"][iev][:, 2])
            hit_phi = _phi_from_sin_cos(
                np.array(output["X_hit"][iev][:, 3]),
                np.array(output["X_hit"][iev][:, 4]),
            )
            hit_eta_used, hit_phi_used, _ = _choose_eta_phi(
                hit_eta, hit_phi, hit_xyz, label="residuals hits"
            )
            data = {
                "hit_xyz": hit_xyz,
                "hit_e": np.array(output["X_hit"][iev][:, 5]),
                "hit_type": np.array(output["X_hit"][iev][:, -2]) + 1,
                "hit_genlink": np.array(output["ygen_hit"][iev]),
                "hit_eta": hit_eta_used,
                "hit_phi": hit_phi_used,
                "track_xyz": track_xyz_calo,
                "track_genlink": np.array(output["ygen_track"][iev]),
                "track_eta": track_eta_calo,
                "track_phi": track_phi_calo,
            }
            all_res.extend(compute_event_residuals(data))
        remaining -= take
    return np.array(all_res, dtype=float)


def plot_residuals_compare(
    res_arc,
    res_cld,
    max_dr=0.1,
    bin_width=0.001,
    show=True,
    save_png=True,
    png_filename=None,
    show_stats=True,
    print_stats=True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    res_arc = np.array(res_arc, dtype=float)
    res_cld = np.array(res_cld, dtype=float)

    if max_dr is not None:
        res_arc = res_arc[res_arc <= max_dr]
        res_cld = res_cld[res_cld <= max_dr]

    arc_stats = _residual_summary(res_arc)
    cld_stats = _residual_summary(res_cld)

    if print_stats:
        print(
            f"ARC: N={arc_stats['n']}, mean={arc_stats['mean']:.5f}, "
            f"median={arc_stats['median']:.5f}, q68={arc_stats['q68']:.5f}"
        )
        print(
            f"CLD: N={cld_stats['n']}, mean={cld_stats['mean']:.5f}, "
            f"median={cld_stats['median']:.5f}, q68={cld_stats['q68']:.5f}"
        )

    bins = np.arange(0, max_dr, bin_width)

    plt.figure(figsize=(10, 5))
    plt.hist(res_arc, bins=bins, histtype='step', linewidth=2, color='#d62728', label='ARC')
    plt.hist(res_cld, bins=bins, histtype='step', linewidth=2, color='#1f77b4', label='CLD')
    plt.xlabel("Delta R")
    plt.ylabel("Count")
    plt.title("Calo center vs track residuals (Delta R)")
    plt.legend()
    plt.tight_layout()

    if show_stats:
        stats_text = (
            f"ARC: N={arc_stats['n']}, μ={arc_stats['mean']:.4f}, "
            f"med={arc_stats['median']:.4f}, q68={arc_stats['q68']:.4f}\n"
            f"CLD: N={cld_stats['n']}, μ={cld_stats['mean']:.4f}, "
            f"med={cld_stats['median']:.4f}, q68={cld_stats['q68']:.4f}"
        )
        plt.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                     fontsize=10, ha='right', va='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    if save_png:
        if png_filename is None:
            png_filename = "residuals_arc_vs_cld.png"
        plt.savefig(png_filename)

    if show:
        plt.show()

    return plt.gca()

def compute_linking_stats_over_events(parquet_path, max_events=200, include_unlinked=True):
    files = _list_parquet_files(parquet_path)
    totals = {
        "n_tracks": 0,
        "n_clusters": 0,
        "n_matched_tracks": 0,
        "n_matched_clusters": 0,
    }
    remaining = max_events

    for parquet_file in files:
        if remaining <= 0:
            break
        output = ak.from_parquet(parquet_file)
        n_events = len(output["X_hit"])
        take = min(remaining, n_events)
        for iev in range(take):
            hit_xyz = _stack_xyz(
                np.array(output["X_hit"][iev][:, 6]),
                np.array(output["X_hit"][iev][:, 7]),
                np.array(output["X_hit"][iev][:, 8]),
            )
            hit_e = np.array(output["X_hit"][iev][:, 5])
            hit_type = np.array(output["X_hit"][iev][:, -2]) + 1
            hit_genlink = np.array(output["ygen_hit"][iev])
            hit_eta = np.array(output["X_hit"][iev][:, 2])
            hit_phi = _phi_from_sin_cos(
                np.array(output["X_hit"][iev][:, 3]),
                np.array(output["X_hit"][iev][:, 4]),
            )
            hit_eta, hit_phi, _ = _choose_eta_phi(
                hit_eta, hit_phi, hit_xyz, label="linking stats hits"
            )
            track_genlink = np.array(output["ygen_track"][iev])

            centers = _cluster_centers_eta_phi(
                hit_eta, hit_phi, hit_genlink, hit_type, hit_xyz=hit_xyz, hit_e=hit_e
            )
            cluster_ids = np.array(list(centers.keys()), dtype=int)
            if not include_unlinked:
                cluster_ids = cluster_ids[cluster_ids >= 0]

            if not include_unlinked:
                track_genlink = track_genlink[track_genlink >= 0]

            totals["n_tracks"] += len(track_genlink)
            totals["n_clusters"] += len(cluster_ids)
            if len(track_genlink) > 0 and len(cluster_ids) > 0:
                totals["n_matched_tracks"] += int(np.sum(np.isin(track_genlink, cluster_ids)))
                totals["n_matched_clusters"] += int(np.sum(np.isin(cluster_ids, track_genlink)))
        remaining -= take

    n_tracks = totals["n_tracks"]
    n_clusters = totals["n_clusters"]
    totals["track_efficiency"] = (totals["n_matched_tracks"] / n_tracks) if n_tracks else 0.0
    totals["cluster_purity"] = (
        totals["n_matched_clusters"] / n_clusters
    ) if n_clusters else 0.0
    return totals


def plot_linking_summary_compare(
    stats_arc,
    stats_cld,
    show=True,
    save_png=True,
    png_filename=None,
):
    metrics = ["track_efficiency", "cluster_purity"]
    arc_vals = [stats_arc.get(m, 0.0) for m in metrics]
    cld_vals = [stats_cld.get(m, 0.0) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=arc_vals, name="ARC", marker_color="#d62728"))
    fig.add_trace(go.Bar(x=metrics, y=cld_vals, name="CLD", marker_color="#1f77b4"))
    fig.update_layout(
        title="Linking summary (tracks vs clusters)",
        yaxis_title="Fraction",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, b=50, t=40),
    )

    if save_png:
        if png_filename is None:
            png_filename = "linking_summary_arc_vs_cld.png"
        fig.write_image(png_filename)

    if show:
        fig.show()

    return fig


def compute_residuals_with_energy_over_events(
    parquet_path, max_events=200, include_unlinked=True
):
    files = _list_parquet_files(parquet_path)
    delta_r = []
    cluster_energy = []
    track_eta = []
    track_phi = []
    gen_ids = []
    remaining = max_events

    for parquet_file in files:
        if remaining <= 0:
            break
        output = ak.from_parquet(parquet_file)
        n_events = len(output["X_hit"])
        take = min(remaining, n_events)
        for iev in range(take):
            hit_xyz = _stack_xyz(
                np.array(output["X_hit"][iev][:, 6]),
                np.array(output["X_hit"][iev][:, 7]),
                np.array(output["X_hit"][iev][:, 8]),
            )
            hit_e = np.array(output["X_hit"][iev][:, 5])
            hit_type = np.array(output["X_hit"][iev][:, -2]) + 1
            hit_genlink = np.array(output["ygen_hit"][iev])
            hit_eta = np.array(output["X_hit"][iev][:, 2])
            hit_phi = _phi_from_sin_cos(
                np.array(output["X_hit"][iev][:, 3]),
                np.array(output["X_hit"][iev][:, 4]),
            )
            hit_eta, hit_phi, _ = _choose_eta_phi(
                hit_eta, hit_phi, hit_xyz, label="residuals+energy hits"
            )

            track_xyz_calo = _stack_xyz(
                np.array(output["X_track"][iev][:, 12]),
                np.array(output["X_track"][iev][:, 13]),
                np.array(output["X_track"][iev][:, 14]),
            )
            track_eta_arr, track_phi_arr = _eta_phi_from_xyz(track_xyz_calo)
            track_genlink = np.array(output["ygen_track"][iev])

            centers = _cluster_centers_eta_phi_with_energy(
                hit_eta, hit_phi, hit_genlink, hit_type, hit_xyz=hit_xyz, hit_e=hit_e
            )
            tracks = _tracks_eta_phi_by_genlink(track_eta_arr, track_phi_arr, track_genlink)
            for gen_id, (eta_c, phi_c, e_sum) in centers.items():
                if not include_unlinked and gen_id < 0:
                    continue
                if gen_id not in tracks:
                    continue
                tr_eta, tr_phi = tracks[gen_id]
                if len(tr_eta) == 0:
                    continue
                dphi = _wrap_phi(tr_phi - phi_c)
                dr = np.sqrt((tr_eta - eta_c) ** 2 + dphi ** 2)
                for idx, dr_val in enumerate(dr):
                    delta_r.append(float(dr_val))
                    cluster_energy.append(float(e_sum))
                    track_eta.append(float(tr_eta[idx]))
                    track_phi.append(float(tr_phi[idx]))
                    gen_ids.append(int(gen_id))
        remaining -= take

    return {
        "delta_r": np.array(delta_r, dtype=float),
        "cluster_energy": np.array(cluster_energy, dtype=float),
        "track_eta": np.array(track_eta, dtype=float),
        "track_phi": np.array(track_phi, dtype=float),
        "gen_id": np.array(gen_ids, dtype=int),
    }


def plot_delta_r_vs_cluster_energy_compare(
    res_arc,
    res_cld,
    max_dr=None,
    max_energy=None,
    bins=50,
    show=True,
    save_png=True,
    png_filename=None,
):
    def _filter(res):
        dr = res["delta_r"]
        e = res["cluster_energy"]
        mask = np.ones(len(dr), dtype=bool)
        if max_dr is not None:
            mask &= dr <= max_dr
        if max_energy is not None:
            mask &= e <= max_energy
        return dr[mask], e[mask]

    arc_dr, arc_e = _filter(res_arc)
    cld_dr, cld_e = _filter(res_cld)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["ARC", "CLD"])
    fig.add_trace(
        go.Histogram2d(x=arc_e, y=arc_dr, nbinsx=bins, nbinsy=bins, colorscale="Viridis"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram2d(x=cld_e, y=cld_dr, nbinsx=bins, nbinsy=bins, colorscale="Viridis"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Delta R vs cluster energy",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, b=50, t=40),
    )
    fig.update_xaxes(title_text="Cluster energy", row=1, col=1)
    fig.update_xaxes(title_text="Cluster energy", row=1, col=2)
    fig.update_yaxes(title_text="Delta R", row=1, col=1)
    fig.update_yaxes(title_text="Delta R", row=1, col=2)

    if save_png:
        if png_filename is None:
            png_filename = "delta_r_vs_cluster_energy_arc_vs_cld.png"
        fig.write_image(png_filename)

    if show:
        fig.show()

    return fig


def compute_matching_matrix_over_events(parquet_path, max_events=200, include_unlinked=True):
    files = _list_parquet_files(parquet_path)
    counts = {}
    ids = set()
    remaining = max_events

    for parquet_file in files:
        if remaining <= 0:
            break
        output = ak.from_parquet(parquet_file)
        n_events = len(output["X_hit"])
        take = min(remaining, n_events)
        for iev in range(take):
            hit_xyz = _stack_xyz(
                np.array(output["X_hit"][iev][:, 6]),
                np.array(output["X_hit"][iev][:, 7]),
                np.array(output["X_hit"][iev][:, 8]),
            )
            hit_e = np.array(output["X_hit"][iev][:, 5])
            hit_type = np.array(output["X_hit"][iev][:, -2]) + 1
            hit_genlink = np.array(output["ygen_hit"][iev])
            hit_eta = np.array(output["X_hit"][iev][:, 2])
            hit_phi = _phi_from_sin_cos(
                np.array(output["X_hit"][iev][:, 3]),
                np.array(output["X_hit"][iev][:, 4]),
            )
            hit_eta, hit_phi, _ = _choose_eta_phi(
                hit_eta, hit_phi, hit_xyz, label="matching matrix hits"
            )

            track_xyz_calo = _stack_xyz(
                np.array(output["X_track"][iev][:, 12]),
                np.array(output["X_track"][iev][:, 13]),
                np.array(output["X_track"][iev][:, 14]),
            )
            track_eta, track_phi = _eta_phi_from_xyz(track_xyz_calo)
            track_genlink = np.array(output["ygen_track"][iev])

            centers = _cluster_centers_eta_phi(
                hit_eta, hit_phi, hit_genlink, hit_type, hit_xyz=hit_xyz, hit_e=hit_e
            )
            if len(centers) == 0 or len(track_eta) == 0:
                continue
            center_ids = np.array(list(centers.keys()), dtype=int)
            center_eta = np.array([centers[cid][0] for cid in center_ids], dtype=float)
            center_phi = np.array([centers[cid][1] for cid in center_ids], dtype=float)

            for tr_eta, tr_phi, tr_gid in zip(track_eta, track_phi, track_genlink):
                if not include_unlinked and tr_gid < 0:
                    continue
                dphi = _wrap_phi(tr_phi - center_phi)
                dr = np.sqrt((tr_eta - center_eta) ** 2 + dphi ** 2)
                min_idx = int(np.argmin(dr))
                cl_gid = int(center_ids[min_idx])
                if not include_unlinked and cl_gid < 0:
                    continue
                key = (int(tr_gid), cl_gid)
                counts[key] = counts.get(key, 0) + 1
                ids.add(int(tr_gid))
                ids.add(cl_gid)
        remaining -= take

    labels = sorted(ids)
    index = {gid: i for i, gid in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for (tr_gid, cl_gid), val in counts.items():
        mat[index[tr_gid], index[cl_gid]] += val
    return mat, labels


def plot_matching_matrix_compare(
    mat_arc,
    labels_arc,
    mat_cld,
    labels_cld,
    show=True,
    save_png=True,
    png_filename=None,
):
    all_labels = sorted(set(labels_arc) | set(labels_cld))
    idx = {gid: i for i, gid in enumerate(all_labels)}

    def _expand(mat, labels):
        out = np.zeros((len(all_labels), len(all_labels)), dtype=int)
        for i, tr_gid in enumerate(labels):
            for j, cl_gid in enumerate(labels):
                out[idx[tr_gid], idx[cl_gid]] = mat[i, j]
        return out

    arc_mat = _expand(mat_arc, labels_arc)
    cld_mat = _expand(mat_cld, labels_cld)
    tick_labels = [str(gid) for gid in all_labels]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["ARC", "CLD"])
    fig.add_trace(
        go.Heatmap(z=arc_mat, x=tick_labels, y=tick_labels, colorscale="Viridis"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=cld_mat, x=tick_labels, y=tick_labels, colorscale="Viridis"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Nearest cluster (cols) vs track gen id (rows)",
        xaxis_title="Cluster gen id",
        yaxis_title="Track gen id",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, b=50, t=40),
    )

    if save_png:
        if png_filename is None:
            png_filename = "matching_matrix_arc_vs_cld.png"
        fig.write_image(png_filename)

    if show:
        fig.show()

    return fig