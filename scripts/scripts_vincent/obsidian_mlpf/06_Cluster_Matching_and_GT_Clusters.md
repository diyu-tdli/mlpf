# Cluster Matching and GT Clusters

## Konzept
Der Determination-Step arbeitet auf Showerobjekten. Dafür muss man aus Node-Level-Clustern Shower-Graphen bauen und mit Truth-Showers matchen.

Es gibt zwei Modi:
- **Predicted clustering**: Clusterlabels aus Modelloutput.
- **GT clustering** (`--use-gt-clusters`): Truth-Clusterlabels direkt verwenden.

Direkter Sprung:
- Labeling-/Matching-Implementierung: [[13_InferenceOC_Functions_DeepDive]]
- EC-Orchestrierung darüber: [[11_EnergyCorrection_DeepDive]]

## Wo im Code
- Datei: `src/layers/utils_training.py`
- Funktion: `obtain_clustering_for_matched_showers(...)`

## Technischer Ablauf
1. Pro Event:
   - falls nicht GT: `coords` + `beta` aus Modelloutput setzen
   - Labels über `DPC_custom_CLD(...)` bestimmen
2. Optional Track-Bereinigung:
   - `remove_bad_tracks_from_cluster(...)`
3. Pred-Cluster gegen Truth-Partikel matchen:
   - `match_showers(...)`
4. Für Matches:
   - eigene Shower-Teilgraphen aufbauen
   - true/reco Energien, PID, Koordinaten sammeln
5. Optional Fake-Cluster hinzufügen (`add_fakes`)
6. Rückgabe als gebatchte Showergraphen + Zielgrößen

## Warum `--use-gt-clusters`?
Damit kann man EC/PID isoliert evaluieren/trainieren, ohne Fehler aus dem Clustering-Schritt.

Direkter Sprung:
- Runmode-Flags: [[15_Args_and_RunModes_DeepDive]]

## Kopplung an Determination
`EnergyCorrection.clustering_and_global_features(...)` ruft dieses Matching auf und erzeugt danach graph-level Features.

## Weiterführende Links
- Determination-Step: [[03_Determination_Step_Energy_PID]]
- Clustering davor: [[02_Clustering_Step]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
