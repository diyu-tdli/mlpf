# Agent Context Hand-Off

## Zweck
Diese Seite ist die komprimierte Übergabe an einen Coding-Agenten, damit er gezielt Änderungen am MLPF-System durchführen kann.

## Systemsummary
- Zwei-Stufen-Architektur:
  1. Clustering (`coord + beta` pro Node)
  2. Determination (Energie/PID/Position pro Shower)

- Entry:
  - `python -m src.train_lightning1`

- Kernmodule:
  - `Gatr_pf_e_noise.ExampleWrapper`
  - `EnergyCorrection` / `EnergyCorrectionWrapper`
  - `obtain_clustering_for_matched_showers`
  - `match_showers`

## Kritische APIs (nicht unbedacht brechen)
1. Outputformat von `ExampleWrapper.forward(...)`
2. Rückgabeformat von `EnergyCorrection.forward_correction(...)`
3. Labelkonvention (`0 = noise`) in Clustering/Matching
4. Keys in prediction dict (`pred_energy_corr`, `pred_pos`, `pred_PID`, ...)

Direkter Sprung:
- API-Ursprung in `ExampleWrapper`: [[10_ExampleWrapper_DeepDive]]
- API-Ursprung in `EnergyCorrection`: [[11_EnergyCorrection_DeepDive]]

## Wo welche Änderung typischerweise passiert
- Neuer Clustering-Algorithmus:
  - `src/layers/inference_oc.py`
  - `src/layers/utils_training.py`
  - optional neue parser flags in `src/utils/parser_args.py`

- Neue Determination-Features:
  - `src/utils/post_clustering_features.py`
  - ggf. `EnergyCorrectionWrapper` input dims anpassen

- Loss-Änderungen:
  - Clustering: `src/layers/object_cond.py`
  - Determination: `src/models/energy_correction_NN_v1.py::get_loss`

## Lesepfad für neue Agenten
1. [[MLPF_HighLevel_Map]]
2. [[01_Training_Entry_and_Pipeline]]
3. [[10_ExampleWrapper_DeepDive]]
4. [[11_EnergyCorrection_DeepDive]]
5. [[13_InferenceOC_Functions_DeepDive]]
6. [[16_Extending_New_Clustering_Method]]
7. [[99_Code_Index]]

## Definition of Done für eine Architekturänderung
1. Training läuft ohne Crash auf kleinem Debug-Run.
2. Prediction/Evaluation-Pfad funktioniert weiter.
3. Keine Inkonsistenz zwischen train/infer Labeling.
4. Outputs in `create_and_store_graph_output` weiterhin konsistent nutzbar.

## Standard-Prompt-Vorlage für einen Agenten
```text
Nutze zuerst [[MLPF_HighLevel_Map]] und dann [[16_Extending_New_Clustering_Method]].
Ziel: Ersetze den aktuellen Clustering-Labeler durch <NAME>.
Constraints:
- Noise label muss 0 bleiben.
- use_gt_clusters Verhalten darf nicht brechen.
- train und predict Pfad müssen denselben neuen Labeler verwenden.
- Rückgabeformate (EnergyCorrection/Output-DataFrame) unverändert lassen.
Liefere:
1) Code-Änderungen
2) kurze Risikoanalyse
3) Debug-Kommando für einen kurzen Testlauf
```

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
