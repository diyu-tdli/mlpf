# Deep Dive: `EnergyCorrection` und `EnergyCorrectionWrapper`

## Rolle
Dieses Modul macht aus gematchten Shower-Objekten die Determination:
- Energie (hauptsächlich neutral)
- optional Richtung/Position
- PID (charged/neutral, getrennt)

## Architektur
- `EnergyCorrection` (Orchestrator)
- `EnergyCorrectionWrapper` (jeweils Modell für charged oder neutral)

Direkter Sprung:
- Determination-High-Level: [[03_Determination_Step_Energy_PID]]
- GATr-Hintergrund: [[04_GATr_in_MLPF]]

## `EnergyCorrection` zentrale Methoden
### `get_PID_categories(...)`
Konzept:
- definiert PID-Klassenräume

Technisch:
- nutzt Flags wie `restrict_PID_charge`, `is_muons`
- setzt `self.pids_charged`, `self.pids_neutral`

Direkter Sprung:
- Flag-Bedeutungen: [[15_Args_and_RunModes_DeepDive]]

### `get_energy_correction(...)`
Konzept:
- instanziiert charged/neutral Modelle

Technisch:
- `self.model_charged = EnergyCorrectionWrapper(..., charged=True, ...)`
- `self.model_neutral = EnergyCorrectionWrapper(..., charged=False, ...)`

### `clustering_and_global_features(g, x, y, add_fakes=True)`
Konzept:
1. Shower-Matching
2. graph-level Features
3. Split charged vs neutral

Technisch:
- ruft `obtain_clustering_for_matched_showers(...)`
- normiert Koordinaten (`/3300`)
- erstellt globale Features via `get_post_clustering_features(...)`
- extra Debug/Fake Features via `get_extra_features(...)`
- charged/neutral Split über `num_tracks`

Direkter Sprung:
- Matching-Funktion intern: [[06_Cluster_Matching_and_GT_Clusters]]
- Post-Cluster Features intern: [[12_PostClustering_Features_DeepDive]]

### `forward_correction(...)`
Konzept:
- führt charged/neutral Vorhersagen aus und merged Outputs

Technisch:
- charged: `model_charged.charged_prediction(...)`
- neutral: `model_neutral.neutral_prediction(...)`
- optional `pred_pos`, `pred_ref_pt`, `pred_PID`
- Rückgabeformat hängt von `return_train` und `explain_ec` ab

### `get_loss(...)`
Konzept:
- Determination-Trainingsziel

Technisch:
- neutraler L1-Energieloss mit Filterung
- PID-Losses (`obtain_PID_charged`, `obtain_PID_neutral`, `pid_loss_weighted`)
- optional fake-score loss

Direkter Sprung:
- Trainingsloop, der diesen Loss nutzt: [[10_ExampleWrapper_DeepDive]]

## `EnergyCorrectionWrapper.predict(...)`
Konzept:
- Showergraph -> latente repräsentation -> Energie/PID/(optional Position)

Technisch:
- eigener GATr auf Showergraph-Nodes
- `scatter_sum` auf Shower-Level
- concat mit global features
- MLP (`Net`) für Energie
- optional PID-Head und Positionspfad

## Wichtig für Erweiterungen
- charged/neutral Split ist zentral in mehreren Stellen kodiert.
- Rückgabe-Dictionary-Felder (`pred_energy_corr`, `pred_pos`, `pred_PID`, ...) sind API für weitere Schritte.

## Weiterführende Links
- Determination-Überblick: [[03_Determination_Step_Energy_PID]]
- Matching: [[06_Cluster_Matching_and_GT_Clusters]]
- Feature Engineering: [[12_PostClustering_Features_DeepDive]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
