# Clustering Step

## Konzept
Ziel ist, aus Hit-/Track-Nodes Objektkandidaten zu formen.

Das Modell lernt pro Node:
- eine Position im Clusterraum (`cluster_coord`, 3D)
- einen Kondensationswert (`beta`)

Intuition:
- Nodes gleicher physikalischer Shower sollen im Clusterraum nah liegen.
- `beta` markiert repräsentative „Seed“-Nodes für Objekte.

Direkter Sprung:
- GATr-Hintergrund: [[04_GATr_in_MLPF]]
- Implementierungsklasse: [[10_ExampleWrapper_DeepDive]]

Anschaulich:
- Wie bei Tropfenbildung: viele einzelne Punkte „kondensieren“ zu wenigen Tropfen (Objekten).
- `beta` entspricht grob der Frage: „Ist dieser Punkt ein guter Mittelpunkt für einen Tropfen?“

## Technische Umsetzung
### Forward-Pfad
- Datei: `src/models/GATr/Gatr_pf_e_noise.py`
- Klasse: `ExampleWrapper`
- Methode: `forward(self, g, y, step_count, ...)`

Wichtige Schritte:
1. Eingaben aus Graph holen (`pos_hits_xyz`, `hit_type`, `e_hits`, `p_hits`).
2. Positions-Normalisierung via BatchNorm (`ScaledGooeyBatchNorm2_1`).
3. Embedding in Geometric-Algebra-Form.
4. GATr-Forward mit block-diagonal Attention-Mask pro Event.
5. Heads:
   - `self.clustering` -> 3D Clusterkoordinaten
   - `self.beta` -> 1D Beta
6. Ausgabe `x = [coord_x, coord_y, coord_z, beta]`.

Direkter Sprung:
- OC-Loss intern: [[05_Object_Condensation_Loss]]
- Matching/Folgepipeline: [[06_Cluster_Matching_and_GT_Clusters]]

### Loss
- In `training_step`: `object_condensation_loss2(...)`
- Flags:
  - `--qmin`
  - `--frac_cluster_loss`
  - `--use-average-cc-pos`
  - `--L_attractive_weight`, `--L_repulsive_weight`, `--fill_loss_weight`

Details: [[05_Object_Condensation_Loss]]

## Mini-Beispiel (Outputform)
Wenn ein Batch-Graph `N` Nodes hat, ist die Clustering-Ausgabe:
- `x.shape = (N, 4)`
- `x[:, 0:3]` = Clusterraum-Koordinaten
- `x[:, 3]` = Beta-Logit pro Node

Nach `sigmoid` gilt:
- `beta in [0,1]`, hohe Werte -> Kandidaten für Kondensationszentren.

## Typischer Debug-Aufruf (kurz)
```bash
python -m src.train_lightning1 \
  --data-train /path/to/parquets/ \
  --data-config config_files/config_hits_track_v4.yaml \
  --network-config src/models/wrapper/example_mode_gatr_noise.py \
  --model-prefix /tmp/mlpf_debug_clustering/ \
  --gpus 0 --batch-size 8 --num-epochs 1 \
  --condensation --qmin 3 --use-average-cc-pos 0.98 \
  --fetch-by-files --fetch-step 1 --train-batches 30
```

## Inferenz/Validierung
In `validation_step` wird je nach Modus gespeichert/weitergereicht:
- reine Clustering-Ausgaben
- oder bei `--correction` zusätzliche Determination-Ausgaben.

## Typischer Script-Kontext
- `training_mlpf_cld_arc_05.sh` nutzt diesen Pfad ohne `--correction`.

## Weiterführende Links
- GATr-Details: [[04_GATr_in_MLPF]]
- Determination-Fortsetzung: [[03_Determination_Step_Energy_PID]]
- Matching: [[06_Cluster_Matching_and_GT_Clusters]]
- Funktions-Deep-Dive: [[10_ExampleWrapper_DeepDive]]
- Inference-Funktionsdetails: [[13_InferenceOC_Functions_DeepDive]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
