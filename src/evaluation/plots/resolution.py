from src.utils.inference.per_particle_metrics import get_mask_id, particle_masses, particle_masses_4_class, safeint
from src.utils.inference.inference_metrics import get_sigma_gaussian
from src.utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix, plot_confusion_matrix_pandora
    , calc_unit_circle_dist, plot_per_energy_resolution2, analyze_fakes, analyze_fakes_PID,
    plot_cm_per_energy, plot_fake_and_missed_energy_regions, quick_plot_mass,
    plot_cm_per_energy_on_overview, calculate_phi
)
import pandas as pd
import torch
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os

def delta_theta_batch(reco, true):
    """
    reco: array-like, shape [N, 3]
    true: array-like, shape [N, 3]

    Returns:
        angles: np.ndarray of shape [N]
                delta_theta per event (rad), NaN where invalid.
    """
    reco = np.asarray(reco, dtype=float)
    true = np.asarray(true, dtype=float)

    if reco.shape != true.shape or reco.shape[1] != 3:
        raise ValueError("Inputs must have shape [N, 3]")

    # norms
    reco_norm = np.linalg.norm(reco, axis=1)
    true_norm = np.linalg.norm(true, axis=1)


    # valid mask (finite and non-zero)
    valid = (
        np.isfinite(reco_norm) & (reco_norm > 0) &
        np.isfinite(true_norm) & (true_norm > 0)
    )

    # initialize output
    angles = np.full(reco.shape[0], np.nan)

    if np.any(valid):
        # normalize only valid rows
        u_reco = reco[valid] / reco_norm[valid][:, None]
        u_true = true[valid] / true_norm[valid][:, None]

        # dot products
        c = np.sum(u_reco * u_true, axis=1)
        c = np.clip(c, -1.0, 1.0)
        angles[valid] = np.arccos(c)

    return angles

def calc_phi_angle(df, pandora=False):
    if pandora:
        assert "pandora_calibrated_pos" in df.columns
    pids = []
    distances = []
    true_e = df.true_showers_E.values
    batch_idx = df.number_batch
    if pandora:
        pred_vect = np.array(df.pandora_calibrated_pos.values.tolist())
        true_vect = (
            np.array(df.true_pos.values.tolist())
            * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
        )
        pred_vect = torch.tensor(pred_vect)
        true_vect = torch.tensor(true_vect)
       
    else:
        pred_vect = np.array(df.pred_pos_matched.values.tolist())
        true_vect = (
            np.array(df.true_pos.values.tolist())
            * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
        )
        pred_vect = torch.tensor(pred_vect)
        true_vect = torch.tensor(true_vect)
       
    
    angles_dist = delta_theta_batch(pred_vect,true_vect )
    return angles_dist


def calculate_response(matched, pandora, log_scale=False, tracks=False, perfect_pid=False, mass_zero=False, ML_pid=False, pid=None, ch=False):

    bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.2))
    mean = []
    variance_om = []
    mean_baseline = []
    variance_om_baseline = []
    mean_true_rec = []
    variance_om_true_rec = []
    mean_errors = []
    variance_om_errors = []
    energy_resolutions = []
    energy_resolutions_reco = []
    distributions = []  # Distributions of E/E_{true} for plotting later
    distributions_reco = []
    mean_pxyz = []
    variance_pxyz = []
    masses = []
    is_track_in_cluster = []
    pxyz_true, pxyz_pred = [], []
    sigma_phi, sigma_theta = [], [] # for the angular resolution vs. energy
    distr_phi, distr_theta = [], []
    mean_cld = []
    variance_om_cld = []
    variance_theta_errors = []
    phi_error  = []
    #binning = 1e-2 * 0.2
    if ch:
        number_bins = 3000 
    else:
        number_bins = 500
    bins_per_binned_E = np.linspace(0, 2, number_bins)

    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
        mask_check = ~pd.isna(matched["pred_showers_E"])
        mask = mask_below * mask_above * mask_check
        true_e = matched.true_showers_E[mask]
        true_rec = matched.reco_showers_E[mask]
        if pandora:
            pred_e = matched.pandora_calibrated_pfo[mask]
            pred_pxyz = np.array(matched.pandora_calibrated_pos[mask].tolist())
        else:
            pred_e = matched.calibrated_E[mask]
            pred_pxyz = np.array(matched.pred_pos_matched[mask].tolist())
        pred_e_nocor = matched.pred_showers_E[mask]
        trk_in_clust = matched.is_track_in_cluster[mask]
        if perfect_pid or mass_zero or ML_pid:
            if len(pred_pxyz):
                pred_pxyz /= np.linalg.norm(pred_pxyz, axis=1).reshape(-1, 1)
            if perfect_pid:
                m = np.array([particle_masses[abs(int(i))] for i in matched.pid[mask]])
            elif ML_pid:
                if pandora:
                    m = np.array([particle_masses[abs(int(i))] for i in matched.pandora_pid[mask]])
                else:
                    m = np.array([particle_masses_4_class.get(safeint(i), 0.0) for i in matched.pred_pid_matched[mask]])
            if mass_zero:
                m = np.array([0 for _ in range(len(matched.pid[mask]))])
            p_squared = (pred_e**2 - m**2).values
            pred_pxyz = np.sqrt(p_squared).reshape(-1, 1) * pred_pxyz
        true_pxyz = np.array(matched.true_pos[mask].tolist())

        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_over_reco = pred_e_nocor / true_rec
            e_cld_plot = (pred_e-true_e) / true_e**2
            distributions.append(e_over_true)
            dist, _, phi_dist, eta_dist = calc_unit_circle_dist(matched[mask], pandora=pandora)
            mean_theta = np.median(eta_dist)
            p16 = np.percentile(eta_dist, 16)
            p84 = np.percentile(eta_dist, 84)
            var_theta = (p84 - p16)/2
            sigma_theta.append(var_theta)
            variance_theta_errors.append(var_theta/np.sqrt(2*len(eta_dist)))
            distr_theta.append(eta_dist)
            angles_dist = calc_phi_angle(matched[mask], pandora=pandora)
            distr_phi.append(angles_dist)
            sigma_phi_var = np.percentile(angles_dist, 68)
            sigma_phi.append(sigma_phi_var)
            phi_error.append(sigma_phi_var/np.sqrt(2*len(angles_dist)))
            distributions_reco.append(e_cld_plot)

            mean_predtotrue = np.median(e_over_true)
            
            p16 = np.percentile(e_over_true, 16)
            p84 = np.percentile(e_over_true, 84)
            var_predtotrue = p84 - p16
            # print(bin_i,bin_i1, len(e_over_true),var_predtotrue/2)
            mean.append(mean_predtotrue)
            variance_om.append(np.abs(var_predtotrue))
            energy_resolutions.append((bin_i1 + bin_i) / 2)
            mean_cld.append(0)
            variance_om_cld.append(np.abs(0))
            sigma = var_predtotrue/(2*mean_predtotrue)
            variance_om_errors.append(sigma/np.sqrt(2*len(e_over_true)))

    return (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
        distributions,
        0,
        variance_om_errors,
        0,
        0,
        [0, 0],
        0,
        phi_error,
        sigma_phi,
        sigma_theta,
        variance_theta_errors,
        distr_phi,
        distributions_reco, 
        mean_cld, 
        variance_om_cld
    )
def get_response_for_id_i(id, matched_pandora, matched_, tracks=False, perfect_pid=False, mass_zero=False, ML_pid=False, pandora=False):
    if id ==[211]:
        ch=True
    else:
        ch = False
    if pandora:
        pids_pandora = np.abs(matched_pandora["pid"].values)
        mask_id = get_mask_id(id, pids_pandora)
        df_id_pandora = matched_pandora[mask_id]
    pids = np.abs(matched_["pid"].values)
    mask_id = get_mask_id(id, pids)
    df_id = matched_[mask_id]
    if pandora:
        (
            mean_p,
            variance_om_p,
            mean_true_rec_p,
            variance_om_true_rec_p,
            energy_resolutions_p,
            energy_resolutions_reco_p,
            mean_baseline,
            variance_om_baseline,
            e_over_e_distr_pandora,
            mean_errors_p,
            variance_errors_p,
            mean_pxyz_pandora, variance_om_pxyz_pandora, masses_pandora, pxyz_true_p, phi_error_p, sigma_phi_pandora, sigma_theta_pandora, distr_phi_pandora, distr_theta_pandora, distr_E_reco_pandora, mp_cld, varp_cld
        ) = calculate_response(df_id_pandora, True, False, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, pid=id[0], ch=ch)
        # Pandora: TODO: do some sort of PID for Pandora
    (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
        e_over_e_distr_model,
        mean_errors,
        variance_errors,
        mean_pxyz, variance_om_pxyz, masses, pxyz_true, phi_error_, sigma_phi, sigma_theta, distr_phi, distr_theta, distr_E_reco, m_cld, var_cld
    ) = calculate_response(df_id, False, False, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, pid=id[0], ch=ch)
    # print("COR:__________________________________")
    # print(variance_om_p)
    # print(variance_om)
    # print("RECO:__________________________________")
    # print(variance_om_true_rec_p)
    # print(variance_om_true_rec)
    dic = {}
    if pandora:
        dic["mean_p"] = mean_p
        dic["variance_om_p"] = variance_om_p
        dic["variance_errors_p"] = np.array(variance_errors_p)
        dic["mean_errors_p"] = np.array(mean_errors_p)
        dic["energy_resolutions_p"] = np.array(energy_resolutions_p)
        dic["mean_p_reco"] = np.array(mean_true_rec_p)
        dic["variance_om_p_reco"] = np.array(variance_om_true_rec_p)
        dic["energy_resolutions_p_reco"] = np.array(energy_resolutions_reco_p)
        dic["distributions_pandora"] = e_over_e_distr_pandora
        dic["distributions_pandora_reco"] = distr_E_reco_pandora
        dic["mp_cld"] = mp_cld
        dic["phi_error_p"] = phi_error_p
        dic["varp_cld"] = varp_cld
        dic["sigma_theta_pandora"] = sigma_theta_pandora
        dic["sigma_phi_pandora"] = sigma_phi_pandora
        dic["sigma_errors_theta_pandora"] = distr_phi_pandora
        dic["distr_angles_pandora"] =distr_theta_pandora


    dic["variance_om"] = variance_om
    dic["mean"] = mean
    dic["mean_errors"] = np.array(mean_errors)
    dic["variance_errors"] = np.array(variance_errors)
    dic["energy_resolutions"] = np.array(energy_resolutions)
    dic["mean_reco"] = mean_true_rec
    dic["variance_om_reco"] = np.array(variance_om_true_rec)
    dic["energy_resolutions_reco"] = np.array(energy_resolutions_reco)
    dic["mean_baseline"] = mean_baseline
    dic["variance_om_baseline"] = np.array(variance_om_baseline)
    dic["distributions_model"] = e_over_e_distr_model
    dic["distributions_model_reco"] = distr_E_reco
    dic["sigma_theta"] = sigma_theta
    dic["distr_angles"] =distr_theta
    dic["sigma_phi"] = sigma_phi
    dic["sigma_errors_theta"] = distr_phi
    dic["m_cld"] = m_cld
    dic["var_cld"] = var_cld
    dic["phi_error"] = phi_error_
    
    return dic
def _centers_to_bins(centers):
    """Reconstruct bin edges from stored bin midpoints using the original log-spaced bins."""
    full_bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.2))
    full_centers = 0.5 * (full_bins[:-1] + full_bins[1:])
    edges = []
    for c in centers:
        idx = np.argmin(np.abs(full_centers - c))
        edges.append(full_bins[idx])
    last_idx = np.argmin(np.abs(full_centers - centers[-1]))
    edges.append(full_bins[last_idx + 1])
    return np.array(edges)

def plot_error_band_resolution(dic, ax, pandora=False, mode="energy", facecolor=None, alpha=0.6):
    """Draw error band for resolution plots. mode: 'energy' or 'angle'."""
    if pandora:
        if mode == "energy":
            eff = 0.5 * np.array(dic["variance_om_p"]) / np.array(dic["mean_p"])
            error_y = np.array(dic["variance_errors_p"]) / 2
        else:
            eff = 1000 * np.array(dic["sigma_phi_pandora"])
            error_y = 1000 * np.array(dic["phi_error_p"]) / 2
        bins = _centers_to_bins(dic["energy_resolutions_p"])
        facecolor = facecolor or "#0F4C5C"
    else:
        if mode == "energy":
            eff = 0.5 * np.array(dic["variance_om"]) / np.array(dic["mean"])
            error_y = np.array(dic["variance_errors"]) / 2
        else:
            eff = 1000 * np.array(dic["sigma_phi"])
            error_y = 1000 * np.array(dic["phi_error"]) / 2
        bins = _centers_to_bins(dic["energy_resolutions"])
        facecolor = facecolor or "#E36414"
    for j in range(len(eff)):
        ax.add_patch(Rectangle(
            (bins[j], eff[j] - error_y[j]),
            bins[j + 1] - bins[j],
            2 * error_y[j],
            facecolor=facecolor, edgecolor=None, alpha=alpha,
        ))

def calculate_resolution(sd_hgb, sd_pandora, PATH_store):
    tracks = True
    # from src.utils.inference.per_particle_metrics import get_response_for_id_i
    perfect_pid = False
    mass_zero = False
    pandora = True
    ML_pid = True
    matched_pandora_h = sd_pandora[(np.abs(sd_pandora.pid)==22)]
    matched_h = sd_hgb[(np.abs(sd_hgb.pid)==22)]
    matched_pandora_hadron= matched_pandora_h[(matched_h.pred_pid_matched==3).values*(matched_pandora_h.pandora_pid==22).values]
    matched_hadron = matched_h[(matched_h.pred_pid_matched==3).values*(matched_pandora_h.pandora_pid==22).values]
    photons_dic = get_response_for_id_i(
        [22], matched_pandora_hadron, matched_hadron, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero,pandora=pandora, 
        ML_pid=ML_pid
    )

    matched_pandora_h = sd_pandora[(np.abs(sd_pandora.pid)==211)+(np.abs(sd_pandora.pid)==2212)]
    matched_h = sd_hgb[(np.abs(sd_hgb.pid)==211)+(np.abs(sd_hgb.pid)==2212)]
    matched_pandora_hadron= matched_pandora_h[(matched_pandora_h.pandora_pid==211).values*(matched_h.pred_pid_matched==1).values]
    matched_hadron = matched_h[(matched_h.pred_pid_matched==1).values*(matched_pandora_h.pandora_pid==211).values]
    # matched_hadron["calibrated_E"]= np.sqrt((matched_hadron["calibrated_E"])**2+(1.3957018E-01**2))
    hadrons_dic2 = get_response_for_id_i(
        [211], matched_pandora_hadron, matched_hadron, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid,pandora=pandora
    )

    matched_pandora_h = sd_pandora[(np.abs(sd_pandora.pid)==2112)+(np.abs(sd_pandora.pid)==130)]
    matched_h = sd_hgb[(np.abs(sd_hgb.pid)==2112)+(np.abs(sd_hgb.pid)==130)]
    matched_pandora_hadron= matched_pandora_h[(matched_pandora_h.pandora_pid==2112).values*(matched_h.pred_pid_matched==2).values]
    matched_hadron = matched_h[(matched_h.pred_pid_matched==2).values*(matched_pandora_h.pandora_pid==2112).values]
    neutrons = get_response_for_id_i(
        [2112], matched_pandora_hadron, matched_hadron, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid,pandora=pandora
    )
    ### E resolution with PID
    photon_edir = photons_dic
    ch_edir = hadrons_dic2
    nh_edir = neutrons
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    size_font = 17

    STYLE_OURS = dict(color="#E36414",  marker='.', linestyle='None', markersize=7) #lw=2.5, 
    STYLE_PANDORA = dict(color="#0F4C5C" ,  marker='s', linestyle='None', markersize=6)
    fig, ax_eff = plt.subplots(2, 3,figsize=(16*2, 10*2))
    bins = [0, 5, 15, 50]
    bin_labels = [f"[{bins[i]},{bins[i + 1]}]" for i in range(len(bins) - 1)]
    ax_eff[0,1].plot(photon_edir["energy_resolutions_p"], 0.5*np.array(photon_edir["variance_om_p"]) / np.array(photon_edir["mean_p"]), **STYLE_PANDORA,label="Pandora")
    # ax_eff[0].plot(photon_edir["energy_resolutions"], photon_edir["variance_om_baseline"] / photon_edir["energy_resolutions"], ".--", c="k", label="Baseline")

    ax_eff[0,1].plot(photon_edir["energy_resolutions"],0.5*np.array(photon_edir["variance_om"]) / np.array(photon_edir["mean"]), label="HitPF", **STYLE_OURS)
    ax_eff[0,1].set_xlabel("Energy [GeV]")
    ax_eff[0,1].set_ylabel("$\sigma_E / E$")
    ax_eff[0,1].set_xticks([2.5, 10.0, 33.0])
    ax_eff[0,1].set_xlim([1e-1,40])
    ax_eff[0,1].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
    ax_eff[0,1].tick_params(axis='x', which='both', direction='inout')
    ax_eff[0,1].legend(fontsize = size_font, title_fontsize=size_font, title=r"photon $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right")
    plot_error_band_resolution(photon_edir, ax_eff[0,1])
    plot_error_band_resolution(photon_edir, ax_eff[0,1], pandora=True)

    ax_eff[0,0].plot(ch_edir["energy_resolutions_p"], 0.5*np.array(ch_edir["variance_om_p"]) / np.array(ch_edir["mean_p"]), **STYLE_PANDORA,label="Pandora")
    # ax_eff[1].plot(ch_edir["energy_resolutions"], ch_edir["variance_om_baseline"] / ch_edir["energy_resolutions"], ".--", c="k", label="Baseline")

    ax_eff[0,0].plot(ch_edir["energy_resolutions"],0.5*np.array(ch_edir["variance_om"])/ np.array(ch_edir["mean"]), label="HitPF", **STYLE_OURS)
    ax_eff[0,0].set_xlabel("Energy [GeV]")
    ax_eff[0,0].set_ylabel("$\sigma_E / E$")
    ax_eff[0,0].set_xticks([2.5, 10.0, 33.0])
    ax_eff[0,0].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
    ax_eff[0,0].tick_params(axis='x', which='both', direction='inout')
    ax_eff[0,0].set_xlim([0.3,40])
    ax_eff[0,0].set_ylim([0,0.03])
    ax_eff[0,0].legend(fontsize = size_font, title_fontsize=size_font, title=r"charged hadron $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right")
    plot_error_band_resolution(ch_edir, ax_eff[0,0])
    plot_error_band_resolution(ch_edir, ax_eff[0,0], pandora=True)

    ax_eff[0,2].plot(nh_edir["energy_resolutions_p"], 0.5*np.array(nh_edir["variance_om_p"]) / np.array(nh_edir["mean_p"]), **STYLE_PANDORA,label="Pandora")
    # ax_eff[2].plot(nh_edir["energy_resolutions"], nh_edir["variance_om_baseline"] / nh_edir["energy_resolutions"], ".--", c="k", label="Baseline")

    ax_eff[0,2].plot(nh_edir["energy_resolutions"],0.5*np.array(nh_edir["variance_om"]) / np.array(nh_edir["mean"]), label="HitPF", **STYLE_OURS)
    ax_eff[0,2].set_xlabel("Energy [GeV]")
    ax_eff[0,2].set_ylabel("$\sigma_E / E$")
    ax_eff[0,2].set_xticks([2.5, 10.0, 33.0])
    ax_eff[0,2].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
    ax_eff[0,2].tick_params(axis='x', which='both', direction='inout')
    plot_error_band_resolution(nh_edir, ax_eff[0,2])
    plot_error_band_resolution(nh_edir, ax_eff[0,2], pandora=True)
    ax_eff[0,2].legend(fontsize = size_font, title_fontsize=size_font, title=r"neutral hadron $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right")
    ax_eff[0,2].set_xlim([1.5,40])
    ax_eff[0,2].set_ylim([0,1.1])


    ax_eff[1,1].plot(photon_edir["energy_resolutions_p"], 1000*np.array(photon_edir["sigma_phi_pandora"]), **STYLE_PANDORA,label="Pandora")
    # ax_eff[0].plot(photon_edir["energy_resolutions"], photon_edir["variance_om_baseline"] / photon_edir["energy_resolutions"], ".--", c="k", label="Baseline")

    ax_eff[1,1].plot(photon_edir["energy_resolutions"],1000*np.array(photon_edir["sigma_phi"]), label="HitPF", **STYLE_OURS)
    ax_eff[1,1].set_xlabel("Energy [GeV]")
    ax_eff[1,1].set_ylabel(r"$\sigma_{ \alpha}$ (mrad)")
    ax_eff[1,1].set_xticks([2.5, 10.0, 33.0])
    ax_eff[1,1].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
    ax_eff[1,1].tick_params(axis='x', which='both', direction='inout')
    ax_eff[1,1].legend(fontsize = size_font, title_fontsize=size_font, title=r"photon $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right")
    ax_eff[1,1].set_xlim([1e-1,40])
    ax_eff[1,1].set_ylim([0,60])
    ax_eff[1,0].plot(ch_edir["energy_resolutions_p"], 1000*np.array(ch_edir["sigma_phi_pandora"]), **STYLE_PANDORA,label="Pandora")
    # ax_eff[1].plot(ch_edir["energy_resolutions"], ch_edir["variance_om_baseline"] / ch_edir["energy_resolutions"], ".--", c="k", label="Baseline")
    plot_error_band_resolution(photon_edir, ax_eff[1,1], mode="angle")
    plot_error_band_resolution(photon_edir, ax_eff[1,1], pandora=True, mode="angle")

    ax_eff[1,0].plot(ch_edir["energy_resolutions"],1000*np.array(ch_edir["sigma_phi"]), label="HitPF", **STYLE_OURS)
    ax_eff[1,0].set_xlabel("Energy [GeV]")
    ax_eff[1,0].set_ylabel(r"$\sigma_{ \alpha}$ (mrad)")
    ax_eff[1,0].set_xticks([2.5, 10.0, 33.0])
    ax_eff[1,0].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
    ax_eff[1,0].tick_params(axis='x', which='both', direction='inout')
    ax_eff[1,0].set_xlim([0.3,40])
    ax_eff[1,0].set_ylim([0,60])
    ax_eff[1,0].legend(fontsize = size_font, title_fontsize=size_font, title=r"charged hadron $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right")
    plot_error_band_resolution(ch_edir, ax_eff[1,0], mode="angle")
    plot_error_band_resolution(ch_edir, ax_eff[1,0], pandora=True, mode="angle")
    ax_eff[1,2].plot(nh_edir["energy_resolutions_p"], 1000*np.array(nh_edir["sigma_phi_pandora"]) , **STYLE_PANDORA,label="Pandora")
    # ax_eff[2].plot(nh_edir["energy_resolutions"], nh_edir["variance_om_baseline"] / nh_edir["energy_resolutions"], ".--", c="k", label="Baseline")

    ax_eff[1,2].plot(nh_edir["energy_resolutions"],1000*np.array(nh_edir["sigma_phi"]) , label="HitPF", **STYLE_OURS)
    ax_eff[1,2].set_xlabel("Energy [GeV]")
    ax_eff[1,2].set_ylabel(r"$\sigma_{ \alpha}$ (mrad)")
    ax_eff[1,2].set_xticks([2.5, 10.0, 33.0])
    ax_eff[1,2].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
    ax_eff[1,2].tick_params(axis='x', which='both', direction='inout')

    ax_eff[1,2].legend(fontsize = size_font, title_fontsize=size_font, title=r"neutral hadron $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right")
    ax_eff[1,2].set_xlim([1.5,40])
    ax_eff[1,2].set_ylim([0,60])
    plot_error_band_resolution(nh_edir, ax_eff[1,2], mode="angle")
    plot_error_band_resolution(nh_edir, ax_eff[1,2], pandora=True, mode="angle")


    for ax in ax_eff.flatten():
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        # ax.legend(fontsize = size_font)
        ax.set_xscale("log")
        ax.tick_params(axis='both', which='major', labelsize=size_font)
        ax.tick_params(axis='both', which='minor', labelsize=size_font)
        # ax.set_xlim([0.5,70])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = size_font
    plt.rcParams['axes.labelsize'] = size_font
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    plt.rcParams['legend.fontsize'] = size_font
    fig.tight_layout()
    fig.savefig(os.path.join(PATH_store, "e_angular_resolution.pdf"), bbox_inches="tight")