# MLPF High-Level Map

## Ziel dieses Wissens-Graphen
Diese Seite ist der Einstieg für das Verständnis des MLPF-Ablaufs:
1. **Clustering Step**: Hits/Tracks werden in Shower-Objekte gruppiert.
2. **Determination Step**: Für diese Objekte werden Eigenschaften bestimmt (Energie, Richtung/Position, PID).

Die Details sind in verlinkten Unterseiten dokumentiert.

## Big Picture
- Einstiegspunkt des Trainings: [[01_Training_Entry_and_Pipeline]]
- Datenfluss bis zum Graph: [[07_Data_Dataloader_and_Graph_Building]]
- Clustering-Schritt: [[02_Clustering_Step]]
- Determination/Properties-Schritt: [[03_Determination_Step_Energy_PID]]

## Kernideen der zwei Stufen
### 1) Clustering
- Konzept: Lerne einen Cluster-Space und einen Kondensationsscore (`beta`) pro Node.
- Technisch: Modell produziert pro Node `x=[cluster_coord(3), beta(1)]`.
- Loss: Objektkondensation ([[05_Object_Condensation_Loss]]).

Anschaulich:
- Stell dir jeden Hit als Punkt in einer Wolke vor.
- Das Modell „zieht“ Punkte derselben Shower zusammen und markiert zentrale Punkte mit hohem `beta`.

### 2) Determination
- Konzept: Auf Shower-Ebene aus Clusterobjekten physikalische Eigenschaften regressieren/klassifizieren.
- Technisch: Matching pred-vs-truth Shower, Feature-Aggregation, dann Energy/PID-Modelle.
- Details: [[03_Determination_Step_Energy_PID]], [[06_Cluster_Matching_and_GT_Clusters]].

Anschaulich:
- Nach dem Clustering werden aus „Punktwolken“ einzelne Objekte.
- Dann bekommt jedes Objekt einen Steckbrief: Energie, Richtung, Teilchentyp.

## Beispielaufrufe
Clustering-Run (typisch):
```bash
python -m src.train_lightning1 \
  --data-train /path/to/parquets/ \
  --data-config config_files/config_hits_track_v4.yaml \
  --network-config src/models/wrapper/example_mode_gatr_noise.py \
  --model-prefix /path/to/out/clustering_run/ \
  --gpus 0 --batch-size 20 --num-epochs 10 \
  --condensation --qmin 3 --use-average-cc-pos 0.98 \
  --fetch-by-files --fetch-step 4 --train-batches 12000
```

Determination-Run (typisch):
```bash
python -m src.train_lightning1 \
  --data-train /path/to/parquets/ \
  --data-config config_files/config_hits_track_v4.yaml \
  --network-config src/models/wrapper/example_mode_gatr_noise.py \
  --model-prefix /path/to/out/props_run/ \
  --gpus 0 --batch-size 8 --num-epochs 1 \
  --correction --freeze-clustering --use-gt-clusters \
  --regress-pos --regress-unit-p \
  --separate-PID-GATr --n-layers-PID-head 3 \
  --restrict_PID_charge --PID-4-class --train-batches 50
```

## Zentrale Modellkomponente
- GATr-Modul als Geometric Algebra Transformer: [[04_GATr_in_MLPF]]

## Praktische Setup-/Pfadseite
- GPFS vs EOS Mapping, wichtige CLI-Flags und typische Run-Konfigurationen: [[08_Path_Mapping_Cluster_Configs]]

## Code-Navigation
- Kompakter Index relevanter Dateien: [[99_Code_Index]]

## Deep Dive Ebene
- Lightning-Kernklasse: [[10_ExampleWrapper_DeepDive]]
- EnergyCorrection intern: [[11_EnergyCorrection_DeepDive]]
- Post-clustering Features: [[12_PostClustering_Features_DeepDive]]
- Inference/Matching-Funktionen: [[13_InferenceOC_Functions_DeepDive]]
- Training-Infra (Callbacks/Optimizer/Scheduler): [[14_Training_Callbacks_Optimizer_Scheduler]]
- Flags und Runmodes: [[15_Args_and_RunModes_DeepDive]]
- Erweiterungspfad neues Clustering: [[16_Extending_New_Clustering_Method]]
- Agent-Übergabe: [[17_Agent_Context_HandOff]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
