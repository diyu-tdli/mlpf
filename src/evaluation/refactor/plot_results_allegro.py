
import matplotlib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.inference.per_particle_metrics import plot_per_energy_resolution, reco_hist_2, \
    plot_mass_contribution_per_category, plot_mass_contribution_per_PID
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
# import mplhep as hep
from src.utils.inference.pandas_helpers import open_mlpf_dataframe, concat_with_batch_fix
from src.utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix, plot_confusion_matrix_pandora,
    plot_efficiency_all, calc_unit_circle_dist, plot_per_energy_resolution2, analyze_fakes, analyze_fakes_PID,
    plot_cm_per_energy, plot_fake_and_missed_energy_regions, quick_plot_mass,
    plot_cm_per_energy_on_overview
)
from src.utils.inference.track_cluster_eff_plots import plot_track_assignation_eval
from src.utils.inference.event_Ks import get_decay_type
import matplotlib.pyplot as plt
import torch
import pickle
from src.evaluation.refactor.preprocess import preprocess_dataframe, renumber_batch_idx
fs = 15
font = {'size': fs}
matplotlib.rc('font', **font)
# hep.style.use("CMS")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    help="Path to the folder with the training in which checkpoints are saved",
                    default="/eos/home-g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files")
parser.add_argument("--preprocess", type=str, help="Comma-separated list of scripts to apply",
                    default="")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory (just the name of the folder, nested under the input path")
parser.add_argument("--mass-only", action="store_true", help="Only quickly plot mass in the energy resolution plots")
# parser.add_argument("--exclude-gt-clusters") # TODO: implement

args = parser.parse_args()
print("Preprocess:", args.preprocess)
PATH_store = os.path.join(args.path, args.output_dir)
if not os.path.exists(PATH_store):
    os.makedirs(PATH_store)
import sys
class Logger(object):
    def __init__(self, filename='log.txt'):
        self.terminal = sys.stdout
        self.log = open(os.path.join(PATH_store, filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
sys.stdout = Logger()
sys.stderr = Logger("err.txt")

PATH_store_individual_plots = os.path.join(PATH_store, "individual_plots")
PATH_store_summary_plots = os.path.join(PATH_store, "summary_plots")
if not os.path.exists(PATH_store_individual_plots):
    os.makedirs(PATH_store_individual_plots)
if not os.path.exists(PATH_store_summary_plots):
    os.makedirs(PATH_store_summary_plots)


dir_top = args.path

sd_hgb1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/allegro_3_46000_gen_test_DCP_CLD0_0_None.pt"), False, False)
sd_hgb = sd_hgb1 #concat_with_batch_fix([sd_hgb1, sd_hgb2, sd_hgb3])

# sd_hgb1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/1_100_0_None_pandora.pt"), False, False)
# sd_hgp = sd_hgb1 #concat_with_batch_fix([sd_hgb1, sd_hgb2, sd_hgb3])
# sd_hgb = sd_hgb[(np.abs(sd_hgb.pid)==211)+(np.abs(sd_hgb.pid)==22)+(np.abs(sd_hgb.pid)==11)+(np.abs(sd_hgb.pid)==130)+(np.abs(sd_hgb.pid)==2112)]
# sd_hgp = sd_hgp[(np.abs(sd_hgp.pid)==211)+(np.abs(sd_hgp.pid)==22)+(np.abs(sd_hgp.pid)==11)+(np.abs(sd_hgp.pid)==130)+(np.abs(sd_hgp.pid)==2112)]
# quick_plot_mass(sd_hgb, sd_hgp, PATH_store_summary_plots)

plot_track_assignation_eval(sd_hgb, sd_hgb, PATH_store_summary_plots)

fig_eff, ax = plt.subplots(4, 2, figsize=(10, 10 ))  # The overview figure of efficiencies #
plot_efficiency_all(None, [sd_hgb], PATH_store_individual_plots, ["ML"], ax=ax)
reco_hist_2(sd_hgb, ax=ax)

fig_eff.tight_layout()
fig_eff.savefig(os.path.join(PATH_store_summary_plots, "overview_Efficiency_FakeRate.pdf"))


# current_dir = PATH_store_individual_plots
# current_dir_detailed = PATH_store_summary_plots
# if not os.path.exists(current_dir):
#     os.makedirs(current_dir)
# if not os.path.exists(current_dir_detailed):
#     os.makedirs(current_dir_detailed)
# print("plot_per_energy_resolution2_multiple")
# plot_per_energy_resolution2_multiple(
#     None,
#     {"ML": sd_hgb},
#     current_dir,
#     tracks=True,
#     perfect_pid=False,
#     mass_zero=True,
#     ML_pid=False,
#     PATH_store_detailed_plots=current_dir_detailed
# )
