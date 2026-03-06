import matplotlib
matplotlib.rc('font', size=15)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.inference.pandas_helpers import open_mlpf_dataframe, concat_with_batch_fix
from src.evaluation.plots.efficiency import calculate_eff, plot_efficiency, plot_clustering_eff, plot_fake_energy
from src.evaluation.plots.mass import plot_mass
from src.evaluation.plots.resolution import calculate_resolution

PATH_store = os.path.join("/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/041225_arc_05/", "paper_plots")
PATH_store_summary_plots = os.path.join(PATH_store, "summary_plots")
# os.makedirs(PATH_store_summary_plots, exist_ok=True)

dir_top = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/041225_arc_05/"

sd_hgb1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__600_8000_0_None.pt"), False, False)
# sd_hgb2, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__400_6000_0_None.pt"), False, False)
# sd_hgb3, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__100_4000_0_None.pt"), False, False)
# sd_hgb4, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__800_10000_0_None.pt"), False, False)
sd_hgb = sd_hgb1 #concat_with_batch_fix([sd_hgb1, sd_hgb2, sd_hgb3,sd_hgb4 ])

sd_pandora1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__600_8000_0_None_pandora.pt"), False, False)
# sd_pandora2, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__400_6000_0_None_pandora.pt"), False, False)
# sd_pandora3, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__100_4000_0_None_pandora.pt"), False, False)
# sd_pandora4, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/05_EPID_times1_9000_040226_basic_ecor24000_removaltracks__800_10000_0_None_pandora.pt"), False, False)
sd_pandora = sd_pandora1 #concat_with_batch_fix([sd_pandora1, sd_pandora2, sd_pandora3, sd_pandora4])


mask = (sd_hgb.pred_pid_matched==4)*(sd_hgb.calibrated_E<1)
sd_hgb.loc[mask, "pred_pid_matched"]=1

## Define soft probabilities in the outputs of the PID if needed
# def softmax(logits):
#     exp = np.exp(logits - np.max(logits))
#     sf = exp / exp.sum()
#     return sf[-1]
    
# sd_hgb['muon_pbb'] = sd_hgb['matched_extra_features'].apply(
#     lambda x: softmax(np.array([x[2], x[3], x[6]]))
# )

mask = sd_hgb.pred_pid_matched==1
sd_hgb.loc[mask, "calibrated_E"] =  np.sqrt((sd_hgb[mask]["calibrated_E"])**2+(1.3957018E-01**2))
mask = sd_hgb.pred_pid_matched==0
sd_hgb.loc[mask, "calibrated_E"] =  np.sqrt((sd_hgb[mask]["calibrated_E"])**2+(5.10998902E-04**2))

### EFFICIENTY PLOTS
eff_dic = calculate_eff(sd_pandora,sd_hgb)
plot_efficiency(eff_dic, PATH_store_summary_plots)
plot_clustering_eff(eff_dic,PATH_store_summary_plots)
plot_fake_energy(eff_dic,PATH_store_summary_plots)


### MASS PLOT
plot_mass(sd_hgb, sd_pandora, PATH_store_summary_plots)


calculate_resolution(sd_hgb, sd_pandora, PATH_store_summary_plots)