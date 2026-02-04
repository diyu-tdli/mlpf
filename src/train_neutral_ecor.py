import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------------------------------
# Config
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENERGY_BINS = torch.tensor([0., 1., 5,10.,20, 30, 100])  # GeV

import matplotlib.pyplot as plt
import numpy as np



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
def load_dataset(path, max_files=None):
    X, y, e_true_list = [], [], []

    files = [f for f in os.listdir(path) if f.endswith(".pkl")]
    if max_files:
        files = files[:max_files]

    for fname in tqdm(files, desc="Loading data"):
        with open(os.path.join(path, fname), "rb") as f:
            data = pickle.load(f)

        x = data["x"]
        e_true = torch.tensor(data["e_true"], dtype=torch.float32)
        pid_tensor = torch.abs(data["pid_y"])

        pid_mapped = torch.tensor(
            [pid_conversion_dict.get(int(p), -1) for p in pid_tensor],
            dtype=torch.long,
        )

        # merge muons into neutral hadrons
        # pid_mapped[pid_mapped == 2] = 0
        # pid_mapped[pid_mapped == 3] = 1
        if len(x) == 0:
            continue

        X.append(torch.tensor(x, dtype=torch.float32))
        y.append(pid_mapped)
        e_true_list.append(e_true)
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    e_true = torch.cat(e_true_list, dim=0) 
    mask  = (y ==-1)+(y ==0)+(y ==1)+(y ==4)
    X = X[~mask] 
    y =y[~mask]
    e_true = e_true[~mask]

    y[y == 2] = 0
    y[y == 3] = 1
    return (
        X,
        y,
        e_true,
    )


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
def train(model, X_train, y_train, e_train, X_val, y_val, e_val, args, pid_energy_weights):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = F.l1_loss(reduction="none")

    best_ce_val = 0.0

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
            ce = criterion(ep, eb)

            # energy bin
            bin_idx = torch.bucketize(eb, ENERGY_BINS.to(DEVICE)) - 1
            bin_idx = torch.clamp(bin_idx, 0, pid_energy_weights.shape[1] - 1)

            # PID × energy weight
            w = pid_energy_weights[yb, bin_idx]

            loss = (ce * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_train)

        # ---- validation
        model.eval()
        with torch.no_grad():
            eval = e_val.to(DEVICE)
            preds = model(X_val.to(DEVICE))
            ce_val = criterion(preds, eval)
           
        if ce_val > best_ce_val:
            best_ce_val = ce_val
            torch.save(model.state_dict(), os.path.join(args.out, "model_best_ecor.pt"))

        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_val": ce_val.mean(),
        })

        print(f"Epoch {epoch:4d} | loss {epoch_loss:.4f} | val ce_val {ce_val:.4f}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(args.train)
    if args.train==True:
        wandb.init(
            project="mlpf_debug",
            entity="ml4hep",
            name=os.path.basename(args.out)+"neutral",
            config=vars(args),
        )

        # ---- load data
        X, y, e_true = load_dataset(args.dataset_path, args.max_files)

        # ---- split
        # ---- remove classes with < 2 samples
        
        print("Filtered class counts:", torch.bincount(y))
        X_train, X_val, y_train, y_val, e_train, e_val = train_test_split(
            X, y, e_true, test_size=0.1, stratify=y, random_state=42
        )

        # ---- normalize
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        std[std == 0] = 1.
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        print(mean)
        print(std)
        # ---- weights
        pid_energy_weights = compute_pid_energy_weights(
            y_train, e_train, n_classes=3
        )

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
            args,
            pid_energy_weights,
        )

        wandb.finish()
    else:
        X, y, e_true = load_dataset(args.dataset_path, args.max_files)
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
        state_dict = torch.load("/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/041225_arc_arc_EPID_w_accum_v6_savedata/model_best_fine_energy_bin_2.pt", map_location="cpu")

        model.load_state_dict(state_dict)
        energy_bins = [(0, 1), (1, 10), (10, 100)]
        plot_cm_per_energy(model, X_val, y_val, e_val, energy_bins, ["EM", "Charged Hadron", "Neutral Hadron"])

if __name__ == "__main__":
    main()
