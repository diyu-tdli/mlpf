# Code Index

## Trainingsskripte
- `scripts/scripts_vincent/training_mlpf_cld_arc_05.sh`
- `scripts/scripts_vincent/train_properties.sh`

## Python Entry und Orchestrierung
- `src/train_lightning1.py`
- `src/utils/parser_args.py`
- `src/utils/train_utils.py`
- `src/utils/callbacks.py`

## Modell-Wrapper und Kernmodell
- `src/models/wrapper/example_mode_gatr_noise.py`
- `src/models/GATr/Gatr_pf_e_noise.py`
- `src/models/energy_correction_NN_v1.py`

## Clustering/Matching/Loss
- `src/layers/object_cond.py`
- `src/layers/utils_training.py`
- `src/utils/post_clustering_features.py`

## Datenebene
- `src/dataset/dataset.py`
- `src/dataset/functions_graph.py`
- `config_files/config_hits_track_v4.yaml`

## Themenlinks
- Pipeline: [[01_Training_Entry_and_Pipeline]]
- Clustering: [[02_Clustering_Step]]
- Determination: [[03_Determination_Step_Energy_PID]]
- GATr: [[04_GATr_in_MLPF]]
- OC-Loss: [[05_Object_Condensation_Loss]]
- Matching: [[06_Cluster_Matching_and_GT_Clusters]]
- Datenfluss: [[07_Data_Dataloader_and_Graph_Building]]
- Pfadmapping: [[08_Path_Mapping_Cluster_Configs]]
- ExampleWrapper Deep Dive: [[10_ExampleWrapper_DeepDive]]
- EnergyCorrection Deep Dive: [[11_EnergyCorrection_DeepDive]]
- Post-clustering Features Deep Dive: [[12_PostClustering_Features_DeepDive]]
- Inference OC Deep Dive: [[13_InferenceOC_Functions_DeepDive]]
- Training Infra Deep Dive: [[14_Training_Callbacks_Optimizer_Scheduler]]
- Args Deep Dive: [[15_Args_and_RunModes_DeepDive]]
- Extending Clustering: [[16_Extending_New_Clustering_Method]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
