# Training Callbacks, Optimizer, Scheduler

## Callback-Ebene
### Wo
- `src/utils/callbacks.py`
- `src/layers/utils_training.py` (`FreezeClustering`)

### `get_callbacks(args)`
- `ModelCheckpoint` alle 500 train-steps
- `LearningRateMonitor`
- `TQDMProgressBar`
- optional `FreezeClustering()` bei `--freeze-clustering`

### `FreezeClustering`
Friert vor Training ein:
- `ScaledGooeyBatchNorm2_1`
- `gatr`
- `clustering`
- `beta`

Effekt:
- Determination-Training ändert primär EC/PID-Komponenten.

Direkter Sprung:
- Warum das fachlich so ist: [[03_Determination_Step_Energy_PID]]
- Wo eingefrorene Module definiert sind: [[10_ExampleWrapper_DeepDive]]

## Optimizer/Scheduler Ebene
### Wo
- `src/models/GATr/Gatr_pf_e_noise.py`

### `configure_optimizers()`
- `Adam(self.parameters(), lr=args.start_lr)`
- Scheduler: `CosineAnnealingThenFixedScheduler`

### `CosineAnnealingThenFixedScheduler`
- Phase 1: Cosine Annealing bis `T_max`
- Phase 2: fixe Lernrate

Direkter Sprung:
- Klassenkontext mit `configure_optimizers`: [[10_ExampleWrapper_DeepDive]]

## Trainer-Strategie
In `train_lightning1.build_trainer(...)`:
- correction runs: `strategy="auto"`
- sonst training: `strategy="ddp"`

Direkter Sprung:
- Gesamtpipeline: [[01_Training_Entry_and_Pipeline]]

## Weiterführende Links
- Pipeline: [[01_Training_Entry_and_Pipeline]]
- ExampleWrapper Deep Dive: [[10_ExampleWrapper_DeepDive]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
