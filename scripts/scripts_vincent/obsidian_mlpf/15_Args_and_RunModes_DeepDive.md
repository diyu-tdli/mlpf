# Args and Run Modes Deep Dive

## Rolle
Die CLI-Flags in `parser_args.py` sind der zentrale Steuermechanismus für Modellverhalten.

Direkter Sprung:
- Pipeline mit Arg-Verarbeitung: [[01_Training_Entry_and_Pipeline]]

## Wichtige Mode-Schalter
- `--correction`: aktiviert Determination/EC/PID-Pfad
- `--freeze-clustering`: friert Clusteringmodule ein
- `--use-gt-clusters`: nutzt Truth-Cluster statt pred-Cluster in Matching/EC
- `--predict`: eval/predict statt training

## Clustering-relevante Flags
- `--qmin`
- `--frac_cluster_loss`
- `--use-average-cc-pos`
- `--L_attractive_weight`
- `--L_repulsive_weight`
- `--fill_loss_weight`

## Determination-relevante Flags
- `--regress-pos`
- `--regress-unit-p`
- `--separate-PID-GATr`
- `--n-layers-PID-head`
- `--restrict_PID_charge`
- `--PID-4-class`

## Daten-/I/O-relevante Flags
- `--data-train`, `--data-val`, `--data-test`
- `--fetch-by-files`, `--fetch-step`
- `--train-val-split`
- `--num-workers`
- `--train-batches`

## Praktischer Modusvergleich
### Clustering-Run
- `--correction` aus
- OC-loss dominiert

Direkter Sprung:
- OC-Loss-Details: [[05_Object_Condensation_Loss]]
- Clustering-Ablauf: [[02_Clustering_Step]]

### Determination-Run
- `--correction` an
- oft mit `--freeze-clustering` und `--use-gt-clusters`

Direkter Sprung:
- Determination-Ablauf: [[03_Determination_Step_Energy_PID]]
- EnergyCorrection intern: [[11_EnergyCorrection_DeepDive]]

## Weiterführende Links
- Pipeline: [[01_Training_Entry_and_Pipeline]]
- Clustering: [[02_Clustering_Step]]
- Determination: [[03_Determination_Step_Energy_PID]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
