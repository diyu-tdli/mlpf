
import matplotlib
import sys
sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import mplhep as hep
import os
import matplotlib.pyplot as plt
from src.utils.inference.efficiency_calc_and_plots import create_eff_dic_pandora,create_eff_dic, limit_error_bars
fs = 15
font = {'size': fs}
matplotlib.rc('font', **font)
hep.style.use("CMS")


size_font=15
particles = {
    "photons":   {"pid": 22,  "name": "Photons",           "group": "Electromagnetic", "row": 0},
    "electrons": {"pid": 11,  "name": "Electrons",         "group": "Electromagnetic", "row": 3},
    "pions":     {"pid": 211, "name": "Charged hadrons",   "group": "Hadronic",        "row": 1},
    "kaons":     {"pid": 130, "name": "Neutral hadrons",   "group": "Hadronic",        "row": 2},
    "muons":     {"pid": 13,  "name": "Muons",             "group": "Hadronic",        "row": 4},
}

colors_list = ["#0F4C5C", "#E36414", "#E36414"]
def plot_error_bars(photons_dic, ax, add, i =0 ):
    energy = photons_dic["energy_eff_" + str(i)]
    eff = photons_dic["eff"+ add + "_" + str(i)]
    error_y = photons_dic["errors"+ add + "_" + str(i)]
    yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    ax.errorbar(energy, eff ,yerr= [yerr_lower, yerr_upper], ecolor=colors_list[i],elinewidth=2,
    alpha=1,
    capsize=6,
    linestyle="none")

def plot_error_bars_pandora(photons_dic, ax, add, i =0 ):
    energy = photons_dic["energy_eff_p"]
    eff = photons_dic["eff_p"+add]
    error_y = photons_dic["errors_p"+add]
    yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    ax.errorbar(energy, eff ,yerr= [yerr_lower, yerr_upper], ecolor=colors_list[1],elinewidth=2,
    alpha=1,
    capsize=6,
    linestyle="none")
def pot_error_bar_fakes(photons_dic, ax, i=0):
    energy = photons_dic["energy_fakes_" + str(i)]
    eff = photons_dic["fakes_" + str(i)]
    error_y = photons_dic["fakes_errors"+ str(i)]
    # yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    error_y = np.array(error_y)/2
    ax.errorbar(energy, eff ,yerr= [error_y, error_y], ecolor=colors_list[i],elinewidth=2,
    alpha=1,
    capsize=6,
    linestyle="none")
def pot_error_bar_fakes_pandora(photons_dic, ax, i=0):
    energy = photons_dic["energy_fakes_p"]
    eff = photons_dic["fakes_p"]
    error_y = photons_dic["fakes_errors_p"]
    # yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    error_y = np.array(error_y)/2
    ax.errorbar(energy, eff ,yerr= [error_y, error_y], ecolor=colors_list[1],elinewidth=2,
    alpha=1,
    capsize=6,
    linestyle="none")


def plot_clustering(sd_hgb, sd_pandora,PATH_store_summary_plots):
    # Initialise dictionaries
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

    ################################### Efficiency plot #####################################################################################################################################################################
    ######################################################################################################################################################################################################
    fig_eff, ax_eff = plt.subplots(2, 3, figsize=(16, 10))
    import matplotlib
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    
    colors_list = ["#0F4C5C", "#E36414", "#E36414"]
    STYLE_OURS = dict(color="#0F4C5C",  marker='.', linestyle='None', markersize=10) #lw=2.5, 
    STYLE_PANDORA = dict(color="#E36414",  marker='.', linestyle='None', markersize=10)
    ax_eff[0,0].plot(eff_dic["photons"]["energy_eff_" + str(0)],
                eff_dic["photons"]["eff" + "_" + str(0)], **STYLE_OURS)
    ax_eff[0,0].plot(eff_dic["photons"]["energy_eff_p"],
                eff_dic["photons"]["eff_p"], **STYLE_PANDORA)
    ax_eff[0,0].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[0,0].set_ylabel("Efficiency",fontsize=size_font)
    ax_eff[0,0].set_title(r"$\gamma$",fontsize=size_font)
    plot_error_bars(eff_dic["photons"],ax_eff[0,0], "")
    plot_error_bars_pandora(eff_dic["photons"],ax_eff[0,0], "")
    #################################
    ax_eff[0,1].plot(eff_dic["pions"]["energy_eff_" + str(0)],
                eff_dic["pions"]["eff" + "_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[0,1].plot(eff_dic["pions"]["energy_eff_p"],
                eff_dic["pions"]["eff_p"], label="Pandora", **STYLE_PANDORA)
    plot_error_bars(eff_dic["pions"],ax_eff[0,1], "")
    ax_eff[0,1].set_title("Charged Hadrons",fontsize=size_font)
    ax_eff[0,1].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[0,1].set_ylabel("Efficiency",fontsize=size_font)
    plot_error_bars_pandora(eff_dic["pions"],ax_eff[0,1], "")
    #################################
    ax_eff[0,2].plot(eff_dic["kaons"]["energy_eff_" + str(0)],
                eff_dic["kaons"]["eff" + "_" + str(0)], label="HitPF", **STYLE_OURS)
    ax_eff[0,2].plot(eff_dic["kaons"]["energy_eff_p"],
                eff_dic["kaons"]["eff_p"], label="Pandora", **STYLE_PANDORA)
    plot_error_bars(eff_dic["kaons"],ax_eff[0,2], "")
    plot_error_bars_pandora(eff_dic["kaons"],ax_eff[0,2], "")
    ax_eff[0,2].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[0,2].set_title("Neutral Hadrons",fontsize=size_font)
    ax_eff[0,2].set_ylabel("Efficiency",fontsize=size_font)

    ##Fake rate #################################
    ax_eff[1,0].plot(eff_dic["photons"]["energy_fakes_" + str(0)],
                eff_dic["photons"]["fakes_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[1,0].plot(eff_dic["photons"]["energy_fakes_p"],
            eff_dic["photons"]["fakes_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1,0].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[1,0].set_ylabel("Fake rate",fontsize=size_font)
    ax_eff[1,0].set_title(r"$\gamma$",fontsize=size_font)
    pot_error_bar_fakes(eff_dic["photons"], ax_eff[1,0], i=0)
    pot_error_bar_fakes_pandora(eff_dic["photons"], ax_eff[1,0], i=0)
    #################################
    ax_eff[1,1].plot(eff_dic["pions"]["energy_fakes_" + str(0)],
        eff_dic["pions"]["fakes_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[1,1].plot(eff_dic["pions"]["energy_fakes_p"],
            eff_dic["pions"]["fakes_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1,1].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[1,1].set_ylabel("Fake rate",fontsize=size_font)
    ax_eff[1,1].set_title("Chardged Hadrons",fontsize=size_font)
    pot_error_bar_fakes(eff_dic["pions"], ax_eff[1,1], i=0)
    pot_error_bar_fakes_pandora(eff_dic["pions"], ax_eff[1,1], i=0)

    ax_eff[1,2].plot(eff_dic["kaons"]["energy_fakes_" + str(0)],
        eff_dic["kaons"]["fakes_" + str(0)],  label="HitPF", **STYLE_OURS)
    ax_eff[1,2].plot(eff_dic["kaons"]["energy_fakes_p"],
            eff_dic["kaons"]["fakes_p"], label="Pandora", **STYLE_PANDORA)
    ax_eff[1,2].set_xlabel("Energy (GeV)",fontsize=size_font)
    ax_eff[1,2].set_ylabel("Fake rate",fontsize=size_font)
    ax_eff[1,2].set_title("Neutral Hadrons",fontsize=size_font)
    pot_error_bar_fakes(eff_dic["kaons"], ax_eff[1,2], i=0)
    pot_error_bar_fakes_pandora(eff_dic["kaons"], ax_eff[1,2], i=0)


    legend_elements = [
        Line2D([0], [0], **STYLE_OURS, label="HitPF"),
        Line2D([0], [0], **STYLE_PANDORA, label="PandoraPFA"),
    ]


    for ax in ax_eff.flatten():
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize = size_font)
        ax.set_xscale("log")
        ax.tick_params(axis='both', which='major', labelsize=size_font)
        ax.tick_params(axis='both', which='minor', labelsize=size_font)
    plt.rcParams['font.size'] = size_font
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    fig_eff.tight_layout()
    fig_eff.savefig(os.path.join(PATH_store_summary_plots, "overview_Efficiency_FakeRate.pdf"))


