# Deep Dive: `Gatr_pf_e_noise.ExampleWrapper`

## Rolle im Gesamtsystem
Diese Klasse ist das zentrale Lightning-Modul für:
- Clustering-Output (`coord + beta`)
- optionalen Determination-Followup (`EnergyCorrection`)
- Training/Validation-Loop

Direkter Sprung:
- Clustering-Kontext: [[02_Clustering_Step]]
- Determination-Kontext: [[03_Determination_Step_Energy_PID]]
- EnergyCorrection intern: [[11_EnergyCorrection_DeepDive]]

## Wichtige Methoden
### `__init__(...)`
Konzept:
- baut Clustering-Backbone + Heads
- optional EC/PID-Subsystem bei `args.correction`

Technisch:
- `self.gatr = GATr(...)`
- `self.ScaledGooeyBatchNorm2_1 = BatchNorm1d(3)`
- `self.clustering = Linear(3 -> 3)`
- `self.beta = Linear(2 -> 1)`
- wenn `args.correction`: `self.energy_correction = EnergyCorrection(self)`

### `forward(g, y, step_count, ..., use_gt_clusters=False)`
Konzept:
- berechnet Node-Level Clusterrepräsentation
- optional ruft danach EC/PID-Pfad auf

Technisch:
1. `inputs = g.ndata["pos_hits_xyz"]`
2. Normierung + Embedding
3. Attention-Mask via `build_attention_mask`
4. `self.gatr(...)`
5. Heads auf GATr-Ausgabe -> `x_cluster_coord`, `beta`
6. `x = cat([coord, beta])`
7. wenn `args.correction`: `self.energy_correction.forward_correction(...)`

Direkter Sprung:
- `forward_correction` im Detail: [[11_EnergyCorrection_DeepDive]]

Sonderfall:
- bei `use_gt_clusters=True` wird intern ein Dummy-`x` gesetzt, weil Cluster aus GT gelesen werden.

### `training_step(batch, batch_idx)`
Konzept:
- standardmäßig OC-Loss
- bei Correction-Mode: EC/PID-Loss dominiert

Technisch:
1. `result = self(batch_g, y, batch_idx)`
2. `object_condensation_loss2(...)`
3. falls `args.correction`: `energy_correction.get_loss(...)`
4. finaler Loss:
   - ohne correction: OC-Loss
   - mit correction: `loss_EC + loss_pid_neutral + loss_pid_charged`

Direkter Sprung:
- OC-Loss intern: [[05_Object_Condensation_Loss]]
- Args, die das steuern: [[15_Args_and_RunModes_DeepDive]]

### `validation_step(...)`
Konzept:
- evaluiert Vorhersagen und erzeugt optionale Output-DataFrames

Technisch:
- ruft im predict/eval-Pfad `create_and_store_graph_output(...)` auf
- unterstützt `use_gt_clusters`, `pandora`, `truth_tracking`

Direkter Sprung:
- Output/Matching-Funktionen: [[13_InferenceOC_Functions_DeepDive]]

### `configure_optimizers()`
Technisch:
- Optimizer: `Adam(lr=args.start_lr)`
- Scheduler: `CosineAnnealingThenFixedScheduler`

## Subtile Implementationsdetails
- `make_mom_zero()` setzt BatchNorm-Momentum nach frühen Epochen auf 0.
- im correction-Mode wird OC-Loss zwar berechnet, aber final überschrieben.

## Weiterführende Links
- Clustering: [[02_Clustering_Step]]
- Determination: [[03_Determination_Step_Energy_PID]]
- Scheduler: [[14_Training_Callbacks_Optimizer_Scheduler]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
