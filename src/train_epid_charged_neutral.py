import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import awkward as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
import pyarrow.parquet as pq
from train_epid_charged import plot_cm_per_energy

# -------------------------------------------------
# Config
# -------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENERGY_BINS = torch.tensor([0., 1., 5,10.,20, 30, 100])  # GeV

import matplotlib.pyplot as plt


def calculate_phi(x, y, z=None):
    return np.arctan2(y, x)

def calculate_eta(x, y, z):
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    return -np.log(np.tan(theta / 2))


pid_conversion_dict = {
    11: 0, -11: 0,
    211: 1, -211: 1,
    130: 2, -130: 2,
    2112: 2, -2112: 2,
    22: 3,
    321: 1, -321: 1,
    2212: 1, -2212: 1,
    310: 2, -310: 2,
    3122: 2, -3122: 2,
    3212: 2, -3212: 2,
    3112: 1, -3112: 1,
    3222: 1, -3222: 1,
    3224: 1, -3224: 1,
    3312: 2, -3312: 2,
    13: 4, -13: 4,
    3322: 2, -3322: 2,
    3334: 1, -3334: 1,
}

# -------------------------------------------------
# Dataset
# -------------------------------------------------
def load_dataset_parquet(
    files=None,
    cache_path= None,
    max_files=None,
    true_energy_index=3,
    charged=False  # <-- CHANGE if needed
):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["X"], data["y"], data["e_true"],  data["y_true"]
    print("Cache not found — building dataset from parquet")
    X_list = []
    y_list = []
    e_true_list = []
    y_true_list = []

    
    if max_files is not None:
        files = files[:max_files]

    for fname in tqdm(files, desc="Loading parquet data"):
        data = ak.from_parquet(fname)

        X_hit_all = data["X_hit"]
        X_track_all = data["X_track"]
        X_gen_all = data["X_gen"]
        ygen_hit_all = data["ygen_hit"]
        ygen_track_all = data["ygen_track"]
        n_events = len(X_hit_all)

        for evt in range(n_events):
            X_hit = np.array(X_hit_all[evt])
            X_track = np.array(X_track_all[evt])
            X_gen = np.array(X_gen_all[evt])
            ygen_hit = np.array(ygen_hit_all[evt])
            ygen_track = np.array(ygen_track_all[evt])
            hit_type_feature_hit = X_hit[:,10]+1
            e_hits = X_hit[:,5]
            pos_xyz_hits_hits = X_hit[:,6:9]
            pos_xyz_hits_tracks = X_track[:,12:15] #(referencePoint_calo.i)
            
            if len(X_hit) == 0:
                continue

            # loop over generator particles
            for igen in np.unique(ygen_hit):
                if igen < 0:
                    continue
                pid = np.abs(X_gen[igen,0])
                if charged:
                    pid_mask = np.sum((pid==11)+(pid==211)+(pid==13)+(pid==2212))
                else:
                    pid_mask = np.sum((pid==22)+(pid==130)+(pid==2112))
                if pid_mask>0:
                    hit_mask = ygen_hit == igen
    
                    hits = X_hit[hit_mask]
                    e_hits_p = e_hits[hit_mask]
                    ecal_e = np.sum(e_hits_p[hit_type_feature_hit[hit_mask]==2])
                    if np.sum(hit_type_feature_hit[hit_mask]==2)>0:
                        per_graph_e_hits_ecal_dispersion = np.std(e_hits_p[hit_type_feature_hit[hit_mask]==2])**2
                    else:
                        per_graph_e_hits_ecal_dispersion = 0
                    sum_e = np.sum(e_hits_p)
                    hcal_e = np.sum(e_hits_p[hit_type_feature_hit[hit_mask]==3])
                    if np.sum(hit_type_feature_hit[hit_mask]==3)>0:
                        per_graph_e_hits_hcal_dispersion = np.std(e_hits_p[hit_type_feature_hit[hit_mask]==3])**2
                    else:
                        per_graph_e_hits_hcal_dispersion = 0
                    
                    num_hits = np.sum(hit_mask)
                    per_graph_e_hits_muon = np.sum(e_hits_p[hit_type_feature_hit[hit_mask]==4])
                    per_graph_n_hits_muon = np.sum(hit_type_feature_hit[hit_mask]==4)
                    if charged:
                        track_mask = ygen_track == igen
                        num_tracks = np.sum(track_mask)
                        tracks = X_track[track_mask]
                        if len(tracks)>0:
                            track_p = np.max(tracks[:,5])
                            chis_tracks = np.max(tracks[:,15]/ tracks[:,16])
                        else:
                            track_p = 0
                            chis_tracks = 0
                    else:
                        num_tracks = 0
                        chis_tracks = 0
                        track_p = 0


                    if hits.shape[0] == 0:
                        continue
                    if charged:
                        pos_xyz_hits = np.concatenate((pos_xyz_hits_hits[hit_mask], pos_xyz_hits_tracks[track_mask]), axis=0)
                    else:
                        pos_xyz_hits = pos_xyz_hits_hits[hit_mask]
                    pos_xyz_hits = pos_xyz_hits/ 3300
                    node_features_avg = np.mean(pos_xyz_hits, axis=0)
                    eta, phi = calculate_eta(
                        node_features_avg[0],
                        node_features_avg[1],
                        node_features_avg[2],
                    ), calculate_phi(node_features_avg[0], node_features_avg[1])
                
                        # no per_graph_e_hits_ecal_dispersion
                        # no per_graph_e_hits_hcal_dispersion
                        
                    particle_features = [ecal_e / sum_e,
                            hcal_e / sum_e,
                            num_hits, track_p,
                            per_graph_e_hits_ecal_dispersion, 
                            per_graph_e_hits_hcal_dispersion,
                            sum_e, num_tracks, np.clip(chis_tracks,-5,5),
                            per_graph_e_hits_muon,
                            per_graph_n_hits_muon, 
                            node_features_avg[0],
                            node_features_avg[1],
                            node_features_avg[2], 
                            eta, 
                            phi
                            ]
    
                    
                    
                    e_true = float(X_gen[igen][8])
        
                    # ---- truth info from generator
                    pdg = int(abs(X_gen[igen][0]))
                    if charged:
                        append_if = (num_tracks>0)*(len(particle_features)>0)
                    else:
                        append_if = len(particle_features)>0
                    if append_if==True:
                        X_list.append(torch.tensor(particle_features, dtype=torch.float32))
                        y_list.append(pid_conversion_dict.get(pdg, -1))
                        e_true_list.append(e_true)
                        y_true_list.append(pdg)

    # ---- stack
    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.long)
    e_true = torch.tensor(e_true_list, dtype=torch.float32)
    y_true=torch.tensor(y_true_list, dtype=torch.float32)


    # ---- apply same filtering logic as before
    if charged:
        mask  = (y == -1) | (y == 2) | (y == 3) 
    else:
        mask = (y == -1) | (y == 0) | (y == 1) | (y == 4)
    X = X[~mask]
    y = y[~mask]
    e_true = e_true[~mask]
    y_true = y_true[~mask]
    if charged:
        recovered_E = X[:,6]/X[:,3]
        X = torch.cat((X,recovered_E.view(-1,1) ), dim=1)

    # merge classes (as in your original code)
    if charged:
         y[y == 4] = 2
    else:
        y[y == 2] = 0
        y[y == 3] = 1
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "X": X.cpu(),
            "y": y.cpu(),
            "e_true": e_true.cpu(),
            "y_true": y_true.cpu()
        },
        cache_path,
    )
    return X, y, e_true, y_true

# -------------------------------------------------
# Model
# -------------------------------------------------
class PIDClassifier(nn.Module):
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)

# class PIDClassifier(nn.Module):
#     def __init__(self, n_features, n_classes):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_features, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),

#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),

#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Linear(64, n_classes),
#         )

#     def forward(self, x):
#         return self.net(x)

# -------------------------------------------------
# Compute PID × Energy-bin weights
# -------------------------------------------------
def compute_pid_energy_weights(y, e_true, n_classes):
    bin_idx = torch.bucketize(e_true, ENERGY_BINS) - 1
    bin_idx = torch.clamp(bin_idx, 0, len(ENERGY_BINS) - 2)

    n_bins = len(ENERGY_BINS) - 1
    counts = torch.zeros(n_classes, n_bins)

    for c, b in zip(y, bin_idx):
        counts[c, b] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean()  # normalize

    return weights.to(DEVICE)


# -------------------------------------------------
# Training
# -------------------------------------------------
def train(model, X_train, y_train, e_train, X_val, y_val, e_val, args, pid_energy_weights):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss(reduction="none")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0

        for i in range(0, len(X_train), args.batch_size):
            idx = perm[i:i + args.batch_size]

            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            eb = e_train[idx].to(DEVICE)

            logits = model(xb)
            ce = criterion(logits, yb)
            # energy bin
            bin_idx = torch.bucketize(eb, ENERGY_BINS.to(DEVICE)) - 1
            bin_idx = torch.clamp(bin_idx, 0, pid_energy_weights.shape[1] - 1)

            # PID × energy weight
            w = pid_energy_weights[yb, bin_idx]
            loss = (ce *w).mean()
            loss_nw = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss_nw.item() * len(xb)

        epoch_loss /= len(X_train)

        # ---- validation
        model.eval()
        with torch.no_grad():
            yval = y_val.to(DEVICE)
            preds = model(X_val.to(DEVICE))
            ce_val = criterion(preds, yval)
            acc = accuracy_score(y_val.cpu().numpy(), preds.argmax(dim=1).cpu().numpy())

        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.out, "model_best_fine_energy_bin_2_06012026_neutral_allf_v1.pt"))

        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_val": ce_val.mean(),
            "val/accuracy": acc,
        })

        print(f"Epoch {epoch:4d} | loss {epoch_loss:.4f} | val acc {acc:.4f}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", nargs="*",default=[])
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--charged", default=False, action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    if args.charged:
        cache_path = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/data_charged_allfeatures_v1.pt"
        n_classes = 3
    else:
        cache_path = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/data_neutral_allfeatures_v1.pt"
        n_classes = 2
    os.makedirs(args.out, exist_ok=True)
    if args.train==True:
        X, y, e_true, y_true = load_dataset_parquet(args.dataset_path,cache_path=cache_path, max_files= args.max_files, charged=args.charged)
        wandb.init(
            project="mlpf_debug",
            entity="ml4hep",
            name=os.path.basename(args.out)+"pid_gun_CLD_all_inputs_exteriortraining",
            config=vars(args),
        )

        # # ---- load data
       

        # X = X[:,[0,1]]
        print("Filtered class counts:", torch.bincount(y))
        X_train, X_val, y_train, y_val, e_train, e_val = train_test_split(
            X, y, e_true, test_size=0.1, stratify=y, random_state=42, #train_size=1000
        )
        mask = torch.sum(torch.isnan(X_train),dim=1)>0
        print("is nan", X_train[mask])
        # ---- normalize
        mean = X_train.mean(dim=0, keepdim=True)
        
        std = X_train.std(dim=0, keepdim=True)
        std[std == 0] = 1.
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        print(mean)
        print(std)
        print("Filtered class counts:", torch.bincount(y))
        # ---- weights
        pid_energy_weights = compute_pid_energy_weights(
            y_train, e_train, n_classes=n_classes
        )
        print("pid_energy_weights",pid_energy_weights)

        # ---- model
        
        model = PIDClassifier(X.shape[1], n_classes=n_classes).to(DEVICE)
        wandb.watch(model, log="gradients", log_freq=100)

        # ---- train
        
        train(
            model,
            X_train,
            y_train,
            e_train,
            X_val,
            y_val,
            e_val,
            args,
            pid_energy_weights,
        )

        wandb.finish()
    else:
        X, y, e_true, _ = load_dataset_parquet(args.dataset_path,cache_path=cache_path, max_files= args.max_files, charged=args.charged)
        # X = X[:,[0,1]]
        X_train, X_val, y_train, y_val, e_train, e_val = train_test_split(
            X, y, e_true, test_size=0.1, stratify=y, random_state=42
        )
        
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        print(mean)
        print(std)
        # tensor([[ 6.0678e-01,  3.9312e-01,  2.0669e+02,  1.1078e+01,  1.9059e-03,
        #   7.1710e-03,  7.6348e+00,  1.0059e+00,  1.5275e+00,  2.2525e-04,
        #   1.6156e+00,  6.5176e-03, -3.1530e-04,  1.1045e-03,  1.6097e-03,
        #  -1.9319e-04,  8.1974e-01]])
        # tensor([[3.8463e-01, 3.8452e-01, 2.3513e+02, 5.9792e+01, 6.3708e-03, 2.8647e-02,
        #         1.0735e+01, 7.7793e-02, 1.0541e+00, 1.0687e-03, 3.0680e+00, 5.1672e-01,
        #         5.1578e-01, 4.1414e-01, 5.8136e-01, 1.8053e+00, 5.0277e-01]]

        # ---- normalize
        # mean = X_train.mean(dim=0, keepdim=True)
        # std = X_train.std(dim=0, keepdim=True)

        # old model 
        if args.charged:
            mean = torch.Tensor([6.3685e-01,  3.6311e-01,  2.4663e+02,  9.9379e+00,  2.2094e-03,
                    8.6935e-03,  9.4012e+00,  1.0016e+00,  1.5576e+00,  1.1574e-04,
                    8.1473e-01,  6.9357e-03, -1.9688e-03,  1.7545e-03,  2.4900e-03,
                    -2.3013e-03,  9.2218e-01])
            std = torch.Tensor([3.7872e-01, 3.7867e-01, 2.5531e+02, 1.1613e+01, 5.5376e-03, 2.8849e-02,
            1.1680e+01, 4.0465e-02, 1.0855e+00, 1.3119e-03, 2.2130e+00, 5.0969e-01,
            5.0856e-01, 3.9794e-01, 5.4018e-01, 1.8046e+00, 2.5514e-01])
            # new model:

            mean = torch.Tensor([6.0678e-01,  3.9312e-01,  2.0669e+02,  1.1078e+01,  1.9059e-03,
            7.1710e-03,  7.6348e+00,  1.0059e+00,  1.5275e+00,  2.2525e-04,
            1.6156e+00,  6.5176e-03, -3.1530e-04,  1.1045e-03,  1.6097e-03,
            -1.9319e-04,  8.1974e-01])
            std = torch.Tensor([3.8463e-01, 3.8452e-01, 2.3513e+02, 5.9792e+01, 6.3708e-03, 2.8647e-02,
            1.0735e+01, 7.7793e-02, 1.0541e+00, 1.0687e-03, 3.0680e+00, 5.1672e-01,
            5.1578e-01, 4.1414e-01, 5.8136e-01, 1.8053e+00, 5.0277e-01])
            file = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/041225_arc_arc_EPID_w_accum_v6_savedata/model_best_fine_energy_bin_2.pt"
            # new model
            file = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/model_best_fine_energy_bin_2_v2_06012026.pt"
            
        else:
            mean = torch.Tensor([0.8313, 0.1687])
            std = torch.Tensor([0.3245, 0.3244])
            mean = torch.Tensor([8.3130e-01, 1.6866e-01, 1.7389e+02, 0.0000e+00, 2.2800e-03, 6.7901e-03,
                    6.8350e+00, 0.0000e+00, 0.0000e+00, 2.3338e-05, 6.0927e-02, 5.0594e-03,
                    9.7549e-05, 1.8372e-03, 2.7192e-03, 1.0284e-03])
            std = torch.Tensor([3.2446e-01, 3.2440e-01, 2.4837e+02, 1.0000e+00, 1.3243e-02, 3.0682e-02,
                    1.0980e+01, 1.0000e+00, 1.0000e+00, 5.9734e-04, 8.6192e-01, 4.9072e-01,
                    4.9035e-01, 3.9050e-01, 5.5450e-01, 1.8071e+00])
            file = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/model_best_fine_energy_bin_2_v2_06012026_neutral_allf.pt"
        
        std[std == 0] = 1.
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        model = PIDClassifier(X.shape[1], n_classes=n_classes).to(DEVICE)
        # old model
        state_dict = torch.load(file, map_location="cpu")

        model.load_state_dict(state_dict)
        energy_bins = [(0, 1), (1, 10), (10, 100)]
        plot_cm_per_energy(model, X_val, y_val, e_val, energy_bins, ["EM", "Charged Hadron", "Neutral Hadron"])
       
       
if __name__ == "__main__":
    main()
