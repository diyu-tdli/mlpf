# MLPF Clustering + Properties (Vincent Scripts)

> Obsidian-Wissensgraph (neu): [[obsidian_mlpf/MLPF_HighLevel_Map]]
> 
> Schnellstart:
> - [[obsidian_mlpf/01_Training_Entry_and_Pipeline]]
> - [[obsidian_mlpf/02_Clustering_Step]]
> - [[obsidian_mlpf/03_Determination_Step_Energy_PID]]
> - [[obsidian_mlpf/99_Code_Index]]
> - [[obsidian_mlpf/17_Agent_Context_HandOff]]

## Scope
Diese Note dokumentiert die zwei Skripte:
- `training_mlpf_cld_arc_05.sh` (Clustering-Training)
- `train_properties.sh` (Properties/Energy+PID-Training auf Basis von Clustern)

Zusatz:
- `training_mlpf_cld_arc.sh` ist aktuell leer (0 Bytes).

## 1) High-Level Architektur
Es gibt einen gemeinsamen Python-Entry-Point:
- `python -m src.train_lightning1`

Von dort aus läuft der Ablauf:
1. Argumente parsen (`parser_args.py`)
2. Daten laden (`train_utils.train_load` -> `SimpleIterDataset` -> `create_graph`)
3. Modell bauen (`example_mode_gatr_noise.py` -> `Gatr_pf_e_noise.ExampleWrapper`)
4. Lightning Trainer bauen (`build_trainer`)
5. Training:
   - Immer wird zuerst Clustering-Ausgabe `(cluster_coords, beta)` gerechnet
   - Falls `--correction` aktiv: danach Energy/PID-Kopf (`EnergyCorrection`)

## 2) Was unterscheidet die zwei Skripte?

### A) Clustering-Skript (`training_mlpf_cld_arc_05.sh`)
Ziel: primär Objektkondensations-/Clustering-Training.
Wichtige Flags:
- `--condensation`
- kein `--correction`
- `--frac_cluster_loss 0`, `--qmin 3`, `--use-average-cc-pos 0.98`

Effekt:
- Verlust kommt aus `object_condensation_loss2(...)`
- kein Energy/PID-Head aktiv

### B) Properties-Skript (`train_properties.sh`)
Ziel: Energy+PID+Positions-Properties trainieren (auf Clustern).
Wichtige Flags:
- `--correction`
- `--freeze-clustering`
- `--use-gt-clusters`
- `--regress-pos`, `--regress-unit-p`
- `--separate-PID-GATr`, `--n-layers-PID-head 3`
- `--restrict_PID_charge`, `--PID-4-class`, `--balance-pid-classes`

Effekt:
- Clustering-Backbone wird per Callback eingefroren
- Clusterbildung für EC kann optional Ground Truth nutzen
- Loss = hauptsächlich Energy- und PID-Loss

## 3) Aufrufkette (Call Graph)

### Shell -> Python
- `scripts/scripts_vincent/training_mlpf_cld_arc_05.sh`
- `scripts/scripts_vincent/train_properties.sh`
beide rufen auf:
- `src/train_lightning1.py`

### In `src/train_lightning1.py`
- `main()`
  - `args = parser.parse_args()`
  - `train_load(args)`
  - `model_setup(args, data_config)`
  - `trainer.fit(model, train_loader, val_loader)`

### Model-Bau
- `train_utils.model_setup(...)` lädt Netzwerkmodul aus `--network-config`
- hier: `src/models/wrapper/example_mode_gatr_noise.py`
- dort `get_model(...)` -> `GraphTransformerNetWrapper` -> `Gatr_pf_e_noise.ExampleWrapper`

### Training Step (zentral)
In `Gatr_pf_e_noise.ExampleWrapper.training_step(...)`:
1. `result = self(batch_g, y, batch_idx)`
2. `object_condensation_loss2(...)` für Clustering-Loss
3. wenn `args.correction`: `energy_correction.get_loss(...)`

## 4) Datenfluss technisch

### 4.1 Dataloader
`train_utils.train_load` baut `SimpleIterDataset` und PyTorch `DataLoader`.

`SimpleIterDataset`:
- lädt Parquet-Chunks asynchron (`_load_next`)
- macht Sampling/Shuffle
- baut pro Event DGL-Graph via `create_graph(...)`

`create_graph(...)` erzeugt Node-Features u.a.:
- `g.ndata["h"] = [xyz, onehot(hit_type), e_hit, p_hit]`
- `g.ndata["particle_number"] = hit_link + 1` (0 = noise)
- zusätzliche Felder wie `chi_squared_tracks`, `pos_hits_xyz`, ...

Collation:
- `graph_batch_func(...)` batched DGL-Graphs + GT-Partikelcontainer.

### 4.2 Clustering Forward
`ExampleWrapper.forward(...)` (ohne GT-Cluster):
1. Normierung + Einbettung von Hit-Position + Hit-Typ
2. GATr forward (mit block-diagonal attention mask pro Event)
3. Projektion auf
   - `x_cluster_coord` (3D Cluster Space)
   - `beta` (Kondensationsscore)
4. Ausgabe `x = [coord3, beta]`

Wenn `args.correction` aktiv: `EnergyCorrection.forward_correction(...)` wird danach aufgerufen.

### 4.3 Matching + Post-clustering Features (Properties-Pfad)
`EnergyCorrection.clustering_and_global_features(...)`:
1. `obtain_clustering_for_matched_showers(...)`
   - entweder vorhergesagte Cluster (DPC_custom_CLD)
   - oder GT-Cluster bei `--use-gt-clusters`
   - Matching pred cluster <-> true showers (`match_showers`)
   - optional fakes hinzufügen
2. pro Shower graph-level Features (`get_post_clustering_features`)
   - z.B. ECAL/HCAL Energieanteile, Anzahl Hits, Track-Infos, Chi2, ...
3. Aufteilen in charged vs neutral via `num_tracks`

### 4.4 Energy/PID Heads
`EnergyCorrectionWrapper` (charged + neutral separat):
- kleiner GATr + MLP für Energie
- optional PID-Head (linear oder MLP)
- bei `--regress-pos`: zusätzlich Richtungs-/Positionsgrößen

### 4.5 Loss im Properties-Training
`EnergyCorrection.get_loss(...)`:
- neutraler Energie-L1-Loss (inkl. filtering)
- PID-Loss charged/neutral (`pid_loss_weighted`)
- optional fake-score loss (aktuell deaktiviert)

In `training_step` wird bei `--correction` gesetzt:
- finaler Loss = `loss_EC + loss_neutral_pid + loss_charged_pid`
- der vorher berechnete Clustering-Loss wird dabei überschrieben

## 5) Was macht `--freeze-clustering` genau?
Über Callback `FreezeClustering` werden eingefroren:
- `ScaledGooeyBatchNorm2_1`
- `gatr`
- `clustering`
- `beta`

Damit werden im Properties-Run primär nur EC/PID-Köpfe trainiert.

## 6) Pfad-Mapping (GPFS -> EOS)
Deine `scripts_vincent` nutzen GPFS-Pfade (anderer Cluster). Lokal im aktuellen Repo sind sinnvolle Äquivalente:

- Code:
  - GPFS: `/gpfs/scratch/ehpc399/vincent/code/mlpf`
  - hier: `/eos/user/v/vriecher/mlpf_arc/mlpf`

- Daten (Beispiel aus vorhandenem Launch-Skript):
  - GPFS: `/gpfs/scratch/ehpc399/vincent/data/5k_mix/05/`
  - EOS-Beispiel: `/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/05/`

- Modelle/W&B/Logs: lokal unter `/eos/user/v/vriecher/...` oder Job-Workdir anlegen.

Hinweis: in `train_lightning1.py` wird für Training `glob(args.data_train[0] + "*.parquet")` verwendet. D.h. `--data-train` sollte auf ein Verzeichnis zeigen, das direkt `.parquet` enthält.

## 7) Konkrete Flag-Interpretation für deine zwei Skripte

### Clustering-Skript
- `--condensation`: nutzt OC-Training
- `--frac_cluster_loss 0`: optionaler Pairwise-Anteil aus
- `--qmin`, `--use-average-cc-pos`: OC-Hyperparameter
- `--tracks`: Track-Info im Graph verfügbar

### Properties-Skript
- `--correction`: aktiviert EC/PID Pfad
- `--freeze-clustering`: friert Clustering-Backbone ein
- `--use-gt-clusters`: Matching/EC mit Truth-Clustern
- `--regress-pos`, `--regress-unit-p`: Richtung/Position statt nur Energie
- `--separate-PID-GATr`: eigener GATr für PID
- `--restrict_PID_charge`: charged/neutral Klassen strikt getrennt

## 8) Wichtige Dateien (Lesereihenfolge)
1. `scripts/scripts_vincent/train_properties.sh`
2. `scripts/scripts_vincent/training_mlpf_cld_arc_05.sh`
3. `src/train_lightning1.py`
4. `src/utils/train_utils.py`
5. `src/models/wrapper/example_mode_gatr_noise.py`
6. `src/models/GATr/Gatr_pf_e_noise.py`
7. `src/models/energy_correction_NN_v1.py`
8. `src/layers/utils_training.py`
9. `src/utils/post_clustering_features.py`
10. `src/dataset/functions_graph.py`, `src/dataset/dataset.py`

## 9) Praktische Checks, wenn du es laufen lässt
- Stimmen `--data-train` Pfade auf echte `*.parquet`?
- Ist `--network-config` relativ zum Repo-Root korrekt?
- Bei 1 GPU: `--gpus 0`, `#SBATCH --gres=gpu:1`
- Bei 4 GPU: `--gpus 0,1,2,3`, entsprechendes SLURM-Setup
- Wenn offline: `WANDB_MODE=offline` ist korrekt
