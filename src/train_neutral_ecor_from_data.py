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
from train_epid_charged_neutral import load_dataset_parquet
# -------------------------------------------------
# Config
# -------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENERGY_BINS = torch.tensor([0., 1., 5,10.,20, 30, 100])  # GeV

import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def extract_shower_features_xyzE(hits, radial_bins=(0.0, 10.0, 30.0, 60.0, 120.0)):
    """
    hits: np.ndarray of shape (N_hits, 4)
          columns = x, y, z, E

    returns: dict of scalar shower features
    """

    x, y, z, E = hits.T
    E_tot = E.sum()
    n_hits = len(E)

    # -----------------------------
    # 1. Energy-weighted centroid
    # -----------------------------
    w = E / E_tot
    x0 = np.sum(w * x)
    y0 = np.sum(w * y)
    z0 = np.sum(w * z)

    coords = np.stack([x - x0, y - y0, z - z0], axis=1)

    # -----------------------------
    # 2. Shower axis via PCA
    # -----------------------------
    C = (coords * w[:, None]).T @ coords
    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis /= np.linalg.norm(axis)

    # -----------------------------
    # 3. Longitudinal & transverse projections
    # -----------------------------
    s = coords @ axis
    r = np.linalg.norm(coords - np.outer(s, axis), axis=1)

    # -----------------------------
    # 4. Longitudinal features
    # -----------------------------
    s_mean = np.sum(w * s)
    s_rms  = np.sqrt(np.sum(w * (s - s_mean)**2))
    s_maxE = s[np.argmax(E)]

    order_s = np.argsort(s)
    E_cum_s = np.cumsum(E[order_s])
    s_90 = s[order_s][np.searchsorted(E_cum_s, 0.9 * E_tot)]

    # -----------------------------
    # 5. Transverse features
    # -----------------------------
    r_mean = np.sum(w * r)
    r_rms  = np.sqrt(np.sum(w * r**2))
    r_max  = r.max()

    r_90 = np.percentile(r, 90)

    # -----------------------------
    # 6. Radial energy profile
    # -----------------------------
    radial_profile = {}
    for i in range(len(radial_bins) - 1):
        rmin, rmax = radial_bins[i], radial_bins[i + 1]
        mask = (r >= rmin) & (r < rmax)
        radial_profile[f"E_r_{rmin}_{rmax}"] = E[mask].sum() / E_tot

    # -----------------------------
    # 7. Hit energy statistics
    # -----------------------------
    E_mean_hit = E.mean()
    E_std_hit  = E.std()
    E_max_hit  = E.max()
    frac_highE_hits = np.sum(E > 0.05 * E_max_hit) / n_hits

    # -----------------------------
    # 8. Collect features
    # -----------------------------
    features = {
        "E_total": E_tot,
        "n_hits": n_hits,

        "x_cog": x0,
        "y_cog": y0,
        "z_cog": z0,

        "s_mean": s_mean,
        "s_rms": s_rms,
        "s_maxE": s_maxE,
        "s_90pct": s_90,

        "r_mean": r_mean,
        "r_rms": r_rms,
        "r_max": r_max,
        "r_90pct": r_90,

        "E_mean_hit": E_mean_hit,
        "E_std_hit": E_std_hit,
        "E_max_hit": E_max_hit,
        "frac_highE_hits": frac_highE_hits,
    }

    features.update(radial_profile)
    return features


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

# -------------------------------------------------
# Model
# -------------------------------------------------
class PIDClassifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)


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
def train(model, X_train, y_train, e_train, X_val, y_val, e_val, y_true_train, y_true_val, args, pid_energy_weights):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_ce_val = 1

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0

        for i in range(0, len(X_train), args.batch_size):
            idx = perm[i:i + args.batch_size]

            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            eb = e_train[idx].to(DEVICE)

            ep = model(xb)
            ep = ep.view(-1)
            eps = 1e-6
            ce = F.l1_loss(ep, eb,reduction="none")

            # energy bin
            bin_idx = torch.bucketize(eb, ENERGY_BINS.to(DEVICE)) - 1
            bin_idx = torch.clamp(bin_idx, 0, pid_energy_weights.shape[1] - 1)

            # PID × energy weight
            w = pid_energy_weights[yb, bin_idx]
            loss = (ce * w).mean()
            loss_nw = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss_nw.item() * len(xb)

        epoch_loss /= len(X_train)

        # ---- validation
        model.eval()
        with torch.no_grad():
            eval = e_val.to(DEVICE)
            preds = model(X_val.to(DEVICE))
            preds = preds.view(-1)
            ce_val = F.l1_loss(preds,eval,reduction="none")
        
        if ce_val.mean() < best_ce_val:
            best_ce_val = ce_val.mean()
            torch.save(model.state_dict(), os.path.join(args.out, "model_best_ecor_070126.pt"))

        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_val": ce_val.mean(),
            "loss val neutrons": ce_val[y_true_val==2112].mean()
        })

        print(f"Epoch {epoch:4d} | loss {epoch_loss:.4f} | val ce_val {ce_val.mean():.4f}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", nargs="*",default=[])
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(args.train)
    if args.train==True:
        X, y, e_true, y_list_true = load_dataset_parquet(args.dataset_path,cache_path="/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/data_neutral_allfeatures_v1.pt", max_files= args.max_files)
        wandb.init(
            project="mlpf_debug",
            entity="ml4hep",
            name=os.path.basename(args.out)+"neutral",
            config=vars(args),
        )

        # # ---- load data
       

    
        print("Filtered class counts:", torch.bincount(y))
        X_train, X_val, y_train, y_val, e_train, e_val, y_true_train, y_true_val = train_test_split(
            X, y, e_true, y_list_true, test_size=0.01, stratify=y, random_state=42
        )

        # ---- normalize
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        # std[std == 0] = 1.
       
        # X_train = (X_train - mean) / std
        # X_val = (X_val - mean) / std
        print("Filtered class counts:", torch.bincount(y))
        # ---- weights
        pid_energy_weights = compute_pid_energy_weights(
            y_train, e_train, n_classes=2
        )
        print("pid_energy_weights",pid_energy_weights)

        # ---- model
        model = PIDClassifier(X.shape[1]).to(DEVICE)
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
            y_true_train,
            y_true_val,
            args,
            pid_energy_weights,
        )

        wandb.finish()
    else:
        X, y, e_true = load_dataset_parquet(args.dataset_path, args.max_files)
        X_train, X_val, y_train, y_val, e_train, e_val = train_test_split(
            X, y, e_true, test_size=0.9, stratify=y, random_state=42
        )
        # mean = X_train.mean(dim=0, keepdim=True)
        # std = X_train.std(dim=0, keepdim=True)
        mean = torch.Tensor([ 6.3685e-01,  3.6311e-01,  2.4663e+02,  9.9379e+00,  2.2094e-03,
          8.6935e-03,  9.4012e+00,  1.0016e+00,  1.5576e+00,  1.1574e-04,
          8.1473e-01,  6.9357e-03, -1.9688e-03,  1.7545e-03,  2.4900e-03,
         -2.3013e-03,  9.2218e-01])
        std = torch.Tensor([3.7872e-01, 3.7867e-01, 2.5531e+02, 1.1613e+01, 5.5376e-03, 2.8849e-02,
         1.1680e+01, 4.0465e-02, 1.0855e+00, 1.3119e-03, 2.2130e+00, 5.0969e-01,
         5.0856e-01, 3.9794e-01, 5.4018e-01, 1.8046e+00, 2.5514e-01])

        # ---- normalize
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        std[std == 0] = 1.
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        model = PIDClassifier(X.shape[1]).to(DEVICE)
        state_dict = torch.load("/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/041225_arc_arc_EPID_w_accum_v6_savedata/model_best_fine_energy_bin_2_.pt", map_location="cpu")

        model.load_state_dict(state_dict)
        energy_bins = [(0, 1), (1, 10), (10, 100)]
       
if __name__ == "__main__":
    main()
