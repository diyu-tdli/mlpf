# Training Entry and Pipeline

## Konzeptionell
Der Python-Entry `src.train_lightning1` orchestriert den gesamten Trainingslauf:
1. Argumente parsen.
2. Datensätze/Dataloader bauen.
3. Modell über Netzwerk-Wrapper instanziieren.
4. Lightning `Trainer` bauen.
5. `fit()` ausführen.

Diese Orchestrierung ist identisch für Clustering-only und Determination-Runs. Die Unterschiede kommen primär über Flags (`--correction`, `--freeze-clustering`, `--use-gt-clusters`).

Direkter Sprung:
- Flag-Details: [[15_Args_and_RunModes_DeepDive]]
- Kernmodul im Training: [[10_ExampleWrapper_DeepDive]]

## Technisch im Code
- Datei: `src/train_lightning1.py`
- `main()`:
  - `parser.parse_args()`
  - `get_samples_steps_per_epoch(args)`
  - `train_load(args)` aus `src.utils.train_utils`
  - `model_setup(args, data_config)` aus `src.utils.train_utils`
  - `build_trainer(...)`
  - `trainer.fit(...)`

Wichtige Details:
- `args.data_train` wird auf `*.parquet` expandiert.
- `build_trainer` setzt bei Correction-Runs `strategy="auto"`, sonst im Training `ddp`.
- Callbacks werden über `src.utils.callbacks.get_callbacks(args)` eingebunden.

Direkter Sprung:
- Callback/Optimizer/Scheduler im Detail: [[14_Training_Callbacks_Optimizer_Scheduler]]

## Wie die beiden Skripte hier reinlaufen
- Clustering: `scripts/scripts_vincent/training_mlpf_cld_arc_05.sh`
- Properties: `scripts/scripts_vincent/train_properties.sh`

Beide rufen:
- `python -m src.train_lightning1 ...`

Beispielaufrufe:
- Clustering + Determination: [[MLPF_HighLevel_Map]]

## Weiterführende Links
- Datenfluss: [[07_Data_Dataloader_and_Graph_Building]]
- Clustering: [[02_Clustering_Step]]
- Determination: [[03_Determination_Step_Energy_PID]]
- Index: [[99_Code_Index]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
