import os

import awkward as ak
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.colors as pc


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
    if gen_pdg is not None and gen_id >= 0 and gen_id < len(gen_pdg):
        return f"PDG {int(gen_pdg[gen_id])}"
    return "particle"


def _build_color_map(gen_ids):
    colors = pc.qualitative.Dark24
    unique_ids = list(dict.fromkeys(gen_ids))
    return {gid: colors[i % len(colors)] for i, gid in enumerate(unique_ids)}


def _genlink_label(gen_id, gen_pdg):
    if gen_pdg is not None and gen_id >= 0 and gen_id < len(gen_pdg):
        return f"PDG {int(gen_pdg[gen_id])}"
    return "particle"


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
            # calo track point as cross with black outline
            fig.add_trace(
                go.Scatter3d(
                    x=track_xyz[m, 0],
                    y=track_xyz[m, 1],
                    z=track_xyz[m, 2],
                    mode="markers",
                    name=f"track {gen_id}",
                    showlegend=False,
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
        fig.add_trace(
            go.Scatter(
                x=[eta_c],
                y=[phi_c],
                mode="markers",
                name=f"center {gen_id}",
                showlegend=False,
                marker=dict(size=7, color="black", symbol="star"),
            )
        )

    if len(track_eta) > 0:
        if len(color_map) == 0:
            color_map = _build_color_map(track_genlink)
        for gen_id in np.unique(track_genlink):
            m = track_genlink == gen_id
            fig.add_trace(
                go.Scatter(
                    x=track_eta[m],
                    y=track_phi[m],
                    mode="markers",
                    name=f"track {gen_id}",
                    showlegend=False,
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
                "hit_xyz": _stack_xyz(
                    np.array(output["X_hit"][iev][:, 6]),
                    np.array(output["X_hit"][iev][:, 7]),
                    np.array(output["X_hit"][iev][:, 8]),
                ),
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
):
    fig = go.Figure()
    res_arc = np.array(res_arc, dtype=float)
    res_cld = np.array(res_cld, dtype=float)

    if max_dr is not None:
        res_arc = res_arc[res_arc <= max_dr]
        res_cld = res_cld[res_cld <= max_dr]

    bins = None
    if bin_width is not None and bin_width > 0:
        bins = dict(start=0.0, end=max_dr if max_dr is not None else None, size=bin_width)

    fig.add_trace(
        go.Histogram(
            x=res_arc,
            name="ARC",
            opacity=0.6,
            marker_color="#d62728",
            xbins=bins,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=res_cld,
            name="CLD",
            opacity=0.6,
            marker_color="#1f77b4",
            xbins=bins,
        )
    )
    fig.update_layout(
        title="Calo center vs track residuals (Delta R)",
        barmode="overlay",
        xaxis_title="Delta R",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, b=50, t=40),
    )

    if save_png:
        if png_filename is None:
            png_filename = "residuals_arc_vs_cld.png"
        fig.write_image(png_filename)

    if show:
        fig.show()

    return fig