from src.utils.inference.efficiency_calc_and_plots import create_eff_dic_pandora, create_eff_dic, limit_error_bars
import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from matplotlib.patches import Rectangle

hep.style.use("CMS")
matplotlib.rc('font', size=15)
particles = {
    "photons":   {"pid": 22,  "name": "Photons",           "group": "Electromagnetic", "row": 0},
    "electrons": {"pid": 11,  "name": "Electrons",         "group": "Electromagnetic", "row": 3},
    "pions":     {"pid": 211, "name": "Charged hadrons",   "group": "Hadronic",        "row": 1},
    "kaons":     {"pid": 130, "name": "Neutral hadrons",   "group": "Hadronic",        "row": 2},
    "muons":     {"pid": 13,  "name": "Muons",             "group": "Hadronic",        "row": 4},
}

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

def _draw_error_band(ax, eff, yerr_lower, yerr_upper, bins, facecolor, alpha, edgecolor):
    for j in range(len(eff)):
        ax.add_patch(Rectangle(
            (bins[j], eff[j] - yerr_lower[j]),
            bins[j + 1] - bins[j],
            yerr_lower[j] + yerr_upper[j],
            facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
        ))

def plot_error_band_fakes(dic, ax, i=0, pandora=False, energy_pct=False, facecolor=None, alpha=0.6, edgecolor=None):
    if pandora:
        eff = np.array(dic["fake_percent_energy_p" if energy_pct else "fakes_p"])
        error_y = np.array(dic["fakes_errors_energy_p" if energy_pct else "fakes_errors_p"]) / 2
        bins = _centers_to_bins(dic["energy_fakes_p"])
        facecolor = facecolor or "#0F4C5C"
    else:
        eff = np.array(dic["fake_percent_energy_" + str(i) if energy_pct else "fakes_" + str(i)])
        error_y = np.array(dic["fakes_errors_energy" + str(i) if energy_pct else "fakes_errors" + str(i)]) / 2
        bins = _centers_to_bins(dic["energy_fakes_" + str(i)])
        facecolor = facecolor or "#E36414"
    _draw_error_band(ax, eff, error_y, error_y, bins, facecolor, alpha, edgecolor)

def plot_error_band_eff(dic, ax, add, i=0, pandora=False, bins=None, facecolor=None, alpha=0.6, edgecolor=None):
    if pandora:
        eff = np.array(dic["eff_p" + add][:-1])
        error_y = np.array(dic["errors_p" + add])[:-1]
        facecolor = facecolor or "#0F4C5C"
    else:
        eff = np.array(dic["eff" + add + "_" + str(i)][:-1])
        error_y = np.array(dic["errors" + add + "_" + str(i)])[:-1]
        facecolor = facecolor or "#E36414"
    yerr_lower, yerr_upper = limit_error_bars(eff, error_y / 2, upper_limit=1)
    _draw_error_band(ax, eff, yerr_lower, yerr_upper, bins, facecolor, alpha, edgecolor)

def calculate_eff(sd_pandora,sd_hgb):
    calc_fakes = sd_pandora is not None
    pandora = calc_fakes
    eff_dic = {
    k: create_eff_dic_pandora(sd_pandora, v["pid"]) if pandora else {}
    for k, v in particles.items()
    }
    # Fill dictionaries from HGB dataframes
    df_list = [sd_hgb]
    for var_i, sd_hgb in enumerate(df_list):
        for k, v in particles.items():
            eff_dic[k] = create_eff_dic(
                eff_dic[k], sd_hgb, v["pid"],
                var_i=var_i,
                calc_fakes=calc_fakes
            )   
    return eff_dic 

def plot_efficiency(eff_dic, path_store):

    ################################### Efficiency plot #####################################################################################################################################################################
    ######################################################################################################################################################################################################
    fig_eff, ax_eff = plt.subplots(2, 3, figsize=(16*2, 10*2))
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    size_font = 17
    STYLE_OURS = dict(color="#E36414",  marker='.', linestyle='None', markersize=7) #lw=2.5, 
    STYLE_PANDORA = dict(color="#0F4C5C" ,  marker='s', linestyle='None', markersize=6)
    ls =0.2
    htp = 0.4
    ################################### Efficiency plot  FOR THE APPENDIX WITH PID AND FAKE ENERGY#####################################################################################################################################################################
    ######################################################################################################################################################################################################
    ax_eff[0,1].plot(eff_dic["photons"]["energy_eff_" + str(0)][:-1],
                eff_dic["photons"]["eff_pid" + "_" + str(0)][:-1],label="HitPF", **STYLE_OURS)
    ax_eff[0,1].plot(eff_dic["photons"]["energy_eff_p"][:-1],
                eff_dic["photons"]["eff_p_pid"][:-1], label="Pandora", **STYLE_PANDORA)
    ax_eff[0,1].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[0,1].set_ylabel("Efficiency",fontsize=size_font)
    ax_eff[0,1].set_ylim([0,1.01])
    ax_eff[0,1].legend(fontsize = size_font, title_fontsize=size_font, title=r"photons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower right",labelspacing=ls,handletextpad=htp)
    ax_eff[0,1].set_xlim([1e-1,40])
    plot_error_band_eff(eff_dic["photons"], ax_eff[0,1], "_pid", bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2)))
    plot_error_band_eff(eff_dic["photons"], ax_eff[0,1], "_pid", pandora=True, bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2)))
    #################################
    ax_eff[0,0].plot(eff_dic["pions"]["energy_eff_" + str(0)][:-1],
                eff_dic["pions"]["eff_pid" + "_" + str(0)][:-1],label="HitPF", **STYLE_OURS)
    ax_eff[0,0].plot(eff_dic["pions"]["energy_eff_p"][:-1],
                eff_dic["pions"]["eff_p_pid"][:-1], label="Pandora", **STYLE_PANDORA)
    plot_error_band_eff(eff_dic["pions"], ax_eff[0,0], "_pid", bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[1:])
    plot_error_band_eff(eff_dic["pions"], ax_eff[0,0], "_pid", pandora=True, bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[1:])
    # ax_eff[0,1].set_title("Charged Hadrons",fontsize=size_font)
    ax_eff[0,0].legend(fontsize = size_font, title_fontsize=size_font, title=r"charged hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower right",labelspacing=ls,handletextpad=htp)
    ax_eff[0,0].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[0,0].set_ylabel("Efficiency",fontsize=size_font)
    ax_eff[0,0].set_ylim([0,1.01])
    ax_eff[0,0].set_xlim([0.3,40])
    #################################


    ax_eff[0,2].plot(eff_dic["kaons"]["energy_eff_" + str(0)],
                eff_dic["kaons"]["eff_pid" + "_" + str(0)],label="HitPF", **STYLE_OURS)
    ax_eff[0,2].plot(eff_dic["kaons"]["energy_eff_p"],
                eff_dic["kaons"]["eff_p_pid"], label="Pandora", **STYLE_PANDORA)
    ax_eff[0,2].legend(fontsize = size_font,  title_fontsize=size_font,title=r"neutral hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower right",labelspacing=ls,handletextpad=htp)

    plot_error_band_eff(eff_dic["kaons"], ax_eff[0,2], "_pid", bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[8:])
    plot_error_band_eff(eff_dic["kaons"], ax_eff[0,2], "_pid", pandora=True, bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[8:])
    ax_eff[0,2].set_xlabel("Energy (GeV)",fontsize=size_font)
    # ax_eff[0,2].set_title("Neutral Hadrons",fontsize=size_font)
    ax_eff[0,2].set_ylabel("Efficiency",fontsize=size_font)
    ax_eff[0,2].set_ylim([0,1.01])
    ax_eff[0,2].set_xlim([1.5,40])
    ##Fake rate #################################
    ax_eff[1,1].plot(eff_dic["photons"]["energy_fakes_" + str(0)],
                eff_dic["photons"]["fakes_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[1,1].plot(eff_dic["photons"]["energy_fakes_p"],
            eff_dic["photons"]["fakes_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1,1].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[1,1].set_ylabel("Fake rate",fontsize=size_font)
    # ax_eff[1,1].set_title(r"$\gamma$",fontsize=size_font)
    ax_eff[1,1].set_ylim([1e-4,1])
    ax_eff[1,1].set_xlim([1e-1,40])
    ax_eff[1,1].legend(fontsize = size_font,  title_fontsize=size_font,title=r"photons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right",labelspacing=ls,handletextpad=htp)
    plot_error_band_fakes(eff_dic["photons"], ax_eff[1,1], i=0)
    plot_error_band_fakes(eff_dic["photons"], ax_eff[1,1], i=0, pandora=True)
    ax_eff[1,1].set_yscale("log")
    #################################
    ax_eff[1,0].plot(eff_dic["pions"]["energy_fakes_" + str(0)],
        eff_dic["pions"]["fakes_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[1,0].plot(eff_dic["pions"]["energy_fakes_p"],
            eff_dic["pions"]["fakes_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1,0].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[1,0].set_ylabel("Fake rate",fontsize=size_font)
    # ax_eff[1,0].set_title("Chardged Hadrons",fontsize=size_font)
    ax_eff[1,0].legend(fontsize = size_font,  title_fontsize=size_font,title=r"charged hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right",labelspacing=ls,handletextpad=htp)
    plot_error_band_fakes(eff_dic["pions"], ax_eff[1,0], i=0)
    plot_error_band_fakes(eff_dic["pions"], ax_eff[1,0], i=0, pandora=True)
    ax_eff[1,0].set_ylim([1e-4,1])
    ax_eff[1,0].set_xlim([0.3,40])
    ax_eff[1,0].set_yscale("log")
    ax_eff[1,2].plot(eff_dic["kaons"]["energy_fakes_" + str(0)],
        eff_dic["kaons"]["fakes_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[1,2].plot(eff_dic["kaons"]["energy_fakes_p"],
            eff_dic["kaons"]["fakes_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1,2].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[1,2].set_ylabel("Fake rate",fontsize=size_font)
    # ax_eff[1,2].set_title("Neutral Hadrons",fontsize=size_font)
    ax_eff[1,2].legend(fontsize = size_font, title_fontsize=size_font, title=r"neutral hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower left",labelspacing=ls,handletextpad=htp)
    plot_error_band_fakes(eff_dic["kaons"], ax_eff[1,2], i=0)
    plot_error_band_fakes(eff_dic["kaons"], ax_eff[1,2], i=0, pandora=True)
    ax_eff[1,2].set_ylim([1e-4,1])
    ax_eff[1,2].set_xlim([1.5,40])
    ax_eff[1,2].set_yscale("log")
    for ax in ax_eff.flatten():
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.tick_params(axis='both', which='major', labelsize=size_font)
        ax.tick_params(axis='both', which='minor', labelsize=size_font)

    plt.rcParams['font.size'] = size_font
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    fig_eff.tight_layout()
    fig_eff.savefig(os.path.join(path_store, "overview_Efficiency_FakeRate.pdf"))


def plot_clustering_eff(eff_dic, PATH_store_summary_plots):
    fig_eff, ax_eff = plt.subplots(1, 3, figsize=(16*2, 5*2))
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    size_font = 17
    STYLE_OURS = dict(color="#E36414", marker='.', linestyle='None', markersize=7)
    STYLE_PANDORA = dict(color="#0F4C5C", marker='s', linestyle='None', markersize=6)
    ls, htp = 0.2, 0.4

    ax_eff[1].plot(eff_dic["photons"]["energy_eff_" + str(0)][:-1],
                eff_dic["photons"]["eff_" + str(0)][:-1], **STYLE_OURS, label="HitPF")
    ax_eff[1].plot(eff_dic["photons"]["energy_eff_p"][:-1],
                eff_dic["photons"]["eff_p"][:-1], **STYLE_PANDORA, label="Pandora")
    ax_eff[1].set_xlabel("Energy (GeV)", fontsize=size_font)
    ax_eff[1].set_ylabel("Clustering Efficiency", fontsize=size_font)
    ax_eff[1].set_ylim([0, 1.01])
    ax_eff[1].legend(fontsize=size_font, title_fontsize=size_font, title=r"photons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower right", labelspacing=ls, handletextpad=htp)
    plot_error_band_eff(eff_dic["photons"], ax_eff[1], "", bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2)))
    plot_error_band_eff(eff_dic["photons"], ax_eff[1], "", pandora=True, bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2)))

    ax_eff[0].plot(eff_dic["pions"]["energy_eff_" + str(0)][:-1],
                eff_dic["pions"]["eff_" + str(0)][:-1], label="HitPF", **STYLE_OURS)
    ax_eff[0].plot(eff_dic["pions"]["energy_eff_p"][:-1],
                eff_dic["pions"]["eff_p"][:-1], label="Pandora", **STYLE_PANDORA)
    ax_eff[0].set_xlabel("Energy (GeV)", fontsize=size_font)
    ax_eff[0].set_ylabel("Clustering Efficiency", fontsize=size_font)
    ax_eff[0].set_ylim([0, 1.01])
    ax_eff[0].set_xlim([1e-1, 40])
    ax_eff[0].legend(fontsize=size_font, title_fontsize=size_font, title=r"charged hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower right", labelspacing=ls, handletextpad=htp)
    plot_error_band_eff(eff_dic["pions"], ax_eff[0], "", bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[1:])
    plot_error_band_eff(eff_dic["pions"], ax_eff[0], "", pandora=True, bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[1:])

    ax_eff[2].plot(eff_dic["kaons"]["energy_eff_" + str(0)][:-1],
                eff_dic["kaons"]["eff_" + str(0)][:-1], label="HitPF", **STYLE_OURS)
    ax_eff[2].plot(eff_dic["kaons"]["energy_eff_p"][:-1],
                eff_dic["kaons"]["eff_p"][:-1], label="Pandora", **STYLE_PANDORA)
    ax_eff[2].set_xlabel("Energy (GeV)", fontsize=size_font)
    ax_eff[2].set_ylabel("Clustering Efficiency", fontsize=size_font)
    ax_eff[2].set_ylim([0, 1.01])
    ax_eff[2].set_xlim([1.5, 40])
    ax_eff[2].legend(fontsize=size_font, title_fontsize=size_font, title=r"neutral hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower right", labelspacing=ls, handletextpad=htp)
    plot_error_band_eff(eff_dic["kaons"], ax_eff[2], "", bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[8:])
    plot_error_band_eff(eff_dic["kaons"], ax_eff[2], "", pandora=True, bins=np.exp(np.arange(np.log(0.1), np.log(80), 0.2))[8:])

    for ax in ax_eff.flatten():
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.tick_params(axis='both', which='major', labelsize=size_font)
        ax.tick_params(axis='both', which='minor', labelsize=size_font)
    plt.rcParams['font.size'] = size_font
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    fig_eff.tight_layout()
    fig_eff.savefig(os.path.join(PATH_store_summary_plots, "overview_Efficiency_clustering_track_removal.pdf"))

def plot_fake_energy(eff_dic, PATH_store_summary_plots):
    fig_eff, ax_eff = plt.subplots(1, 3, figsize=(16*2, 5*2))
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    size_font = 17
    STYLE_OURS = dict(color="#E36414", marker='.', linestyle='None', markersize=7)
    STYLE_PANDORA = dict(color="#0F4C5C", marker='s', linestyle='None', markersize=6)
    ls, htp = 0.2, 0.4

    ax_eff[1].plot(eff_dic["photons"]["energy_fakes_" + str(0)],
                eff_dic["photons"]["fake_percent_energy_" + str(0)], label="HitPF", **STYLE_OURS)
    ax_eff[1].plot(eff_dic["photons"]["energy_fakes_p"],
                eff_dic["photons"]["fake_percent_energy_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1].set_xlabel("Energy (GeV)", fontsize=size_font)
    ax_eff[1].set_ylabel("Fake Energy %", fontsize=size_font)
    ax_eff[1].legend(fontsize=size_font, title_fontsize=size_font, title=r"photons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right", labelspacing=ls, handletextpad=htp)
    ax_eff[1].set_ylim([1e-4, 5])
    ax_eff[1].set_xlim([1e-1, 40])
    ax_eff[1].set_yscale("log")
    plot_error_band_fakes(eff_dic["photons"], ax_eff[1], i=0, energy_pct=True)
    plot_error_band_fakes(eff_dic["photons"], ax_eff[1], i=0, pandora=True, energy_pct=True)

    ax_eff[0].plot(eff_dic["pions"]["energy_fakes_" + str(0)],
                eff_dic["pions"]["fake_percent_energy_" + str(0)], label="HitPF", **STYLE_OURS)
    ax_eff[0].plot(eff_dic["pions"]["energy_fakes_p"],
                eff_dic["pions"]["fake_percent_energy_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[0].set_xlabel("Energy (GeV)", fontsize=size_font)
    ax_eff[0].set_ylabel("Fake Energy %", fontsize=size_font)
    ax_eff[0].legend(fontsize=size_font, title_fontsize=size_font, title=r"charged hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="upper right", labelspacing=ls, handletextpad=htp)
    ax_eff[0].set_ylim([1e-4, 5])
    ax_eff[0].set_xlim([1e-1, 40])
    ax_eff[0].set_yscale("log")
    plot_error_band_fakes(eff_dic["pions"], ax_eff[0], i=0, energy_pct=True)
    plot_error_band_fakes(eff_dic["pions"], ax_eff[0], i=0, pandora=True, energy_pct=True)

    ax_eff[2].plot(eff_dic["kaons"]["energy_fakes_" + str(0)],
                eff_dic["kaons"]["fake_percent_energy_" + str(0)], label="HitPF", **STYLE_OURS)
    ax_eff[2].plot(eff_dic["kaons"]["energy_fakes_p"],
                eff_dic["kaons"]["fake_percent_energy_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[2].set_xlabel("Energy (GeV)", fontsize=size_font)
    ax_eff[2].set_ylabel("Fake Energy %", fontsize=size_font)
    ax_eff[2].legend(fontsize=size_font, title_fontsize=size_font, title=r"neutral hadrons $Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV", loc="lower left", labelspacing=ls, handletextpad=htp)
    ax_eff[2].set_ylim([1e-4, 5])
    ax_eff[2].set_xlim([1.5, 40])
    ax_eff[2].set_yscale("log")
    plot_error_band_fakes(eff_dic["kaons"], ax_eff[2], i=0, energy_pct=True)
    plot_error_band_fakes(eff_dic["kaons"], ax_eff[2], i=0, pandora=True, energy_pct=True)

    for ax in ax_eff.flatten():
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.tick_params(axis='both', which='major', labelsize=size_font)
        ax.tick_params(axis='both', which='minor', labelsize=size_font)
    plt.rcParams['font.size'] = size_font
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    fig_eff.tight_layout()
    fig_eff.savefig(os.path.join(PATH_store_summary_plots, "FakeEnergy_track_removal.pdf"))