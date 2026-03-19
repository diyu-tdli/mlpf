# Path Mapping and Run Configs

## Kontext
Die Vincent-Skripte sind für einen Cluster mit GPFS-Pfaden geschrieben. Lokal arbeitest du hier auf EOS.

Direkter Sprung:
- Konkrete Beispiel-CLI: [[MLPF_HighLevel_Map]]
- Arg- und Runmode-Bedeutung: [[15_Args_and_RunModes_DeepDive]]

## Typisches Mapping
- Code:
  - GPFS: `/gpfs/scratch/ehpc399/vincent/code/mlpf`
  - EOS hier: `/eos/user/v/vriecher/mlpf_arc/mlpf`

- Daten:
  - GPFS-Beispiel: `/gpfs/scratch/ehpc399/vincent/data/5k_mix/05/`
  - EOS-Beispiel: `/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_ecort/05/`

- Outputs:
  - GPFS: `/gpfs/scratch/ehpc399/vincent/models/...`
  - EOS: frei wählbar unter `/eos/user/v/vriecher/...`

## Bestehende Skripte
- Vincent:
  - `scripts/scripts_vincent/training_mlpf_cld_arc_05.sh`
  - `scripts/scripts_vincent/train_properties.sh`
- ähnliche Referenzen im Repo:
  - `slurm/launch_clustering_training.sh`
  - `slurm/launch_energy_PID_training.sh`

## Flag-Checkliste
1. `--data-train` zeigt auf Verzeichnis mit `*.parquet`.
2. `--network-config` ist relativ zum Repo root.
3. `--gpus` passt zu `#SBATCH --gres=gpu:N`.
4. Bei offline Umgebungen: `WANDB_MODE=offline`.

Direkter Sprung:
- Training-Infrastruktur (Callbacks/Trainer): [[14_Training_Callbacks_Optimizer_Scheduler]]

## Empfohlene Lernreihenfolge
1. [[MLPF_HighLevel_Map]]
2. [[01_Training_Entry_and_Pipeline]]
3. [[02_Clustering_Step]]
4. [[03_Determination_Step_Energy_PID]]
5. [[99_Code_Index]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
