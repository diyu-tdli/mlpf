import torch
import sys
import os.path as osp
import os
import sys
import numpy as np

sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
from src.dataset.dataset import SimpleIterDataset
from src.utils.utils import to_filelist
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_scatter import scatter_sum
import matplotlib.pyplot as plt
import pickle
import numpy as np
import mplhep as hep


hep.style.use("CMS")
import matplotlib
matplotlib.rc('font', size=13)

#"/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia_evaluation/Zcard/reco_Zcard_1.root"
datasets = {
    "test": "/eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/train/gun_dr_logE_211125_test/out_reco_edm4hep_1_0.parquet",
    "train": "/eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/train/gun_dr_logE_211125_test/out_reco_edm4hep_1_0.parquet",
}

class Args:
    def __init__(self, datasets):
        self.data_train = [datasets]
        self.data_val = [datasets]
        #self.data_train = files_train
        self.data_config = '/afs/cern.ch/work/m/mgarciam/private/mlpf/config_files/config_hits_track_v4.yaml'
        self.extra_selection = None
        self.train_val_split = 1
        self.data_fraction = 1
        self.file_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 0.1
        self.steps_per_epoch = None
        self.in_memory = False
        self.local_rank = None
        self.copy_inputs = False
        self.no_remake_weights = False
        self.batch_size = 1
        self.num_workers = 0
        self.demo = False
        self.laplace = False
        self.diffs = False
        self.class_edges = False
        self.allegro = True
        self.truth_tracking = True


args = {key: Args(value) for key, value in datasets.items()}

datas = {}
files_dict = {}
for key in datasets:
    train_range = (0, args[key].train_val_split)
    train_file_dict, train_files = to_filelist(args[key], 'val')
    train_data = SimpleIterDataset(train_file_dict, args[key].data_config, for_training=False,
                                   extra_selection=args[key].extra_selection,
                                   remake_weights=True,
                                   load_range_and_fraction=(train_range, args[key].data_fraction),
                                   file_fraction=args[key].file_fraction,
                                   fetch_by_files=args[key].fetch_by_files,
                                   fetch_step=args[key].fetch_step,
                                   infinity_mode=False,
                                   in_memory=args[key].in_memory,
                                   async_load=False,
                                   args_parse=Args(""),
                                   name='train')
    datas[key] = train_data
    files_dict[key] = train_files

import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
itera = iter(train_data)
g, y = next(itera)
len(y.pid), torch.unique(g.ndata["particle_number"])


import pandas as pd
import plotly.express as px
mask = (g.ndata['hit_type'] ==-1)    
tidx =  1*(g.ndata['particle_number'][mask].view(-1,1))
#tidx =    1*(g.ndata['hit_link_modified'][mask].view(-1,1))+1
features =  20*(g.ndata['e_hits'][mask].view(-1,1)) +g.ndata["h"][mask][:,-1].view(-1,1)
X = g.ndata["pos_hits_xyz"][mask] #[mask]
data = {
            "X":X[:, 0].view(-1, 1).detach().cpu().numpy(),
            "Y": X[:, 1].view(-1, 1).detach().cpu().numpy(),
            "Z": X[:, 2].view(-1, 1).detach().cpu().numpy(),
            "tIdx": tidx.view(-1, 1).detach().cpu().numpy(),
            "features": features.view(-1, 1).detach().cpu().numpy(),
        }
hoverdict = {}
df = pd.DataFrame(
np.concatenate([data[k] for k in data], axis=1),
columns=[k for k in data],
)
rdst = np.random.RandomState(1234567890)  # all the same
# shuffle_truth_colors(df, "tIdx", rdst)

hover_data = ["tIdx"] #+ [k for k in hoverdict.keys()]
# if nidx is not None:
#     hover_data.append("av_same")
fig = px.scatter_3d(
df,
x="X",
y="Y",
z="Z",
color="tIdx",
size="features",
hover_data=hover_data,
template="simple_white",
)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.write_html(
               "plot.html"
            )