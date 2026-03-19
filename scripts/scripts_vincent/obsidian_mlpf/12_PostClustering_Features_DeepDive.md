# Deep Dive: Post-Clustering Features

## Rolle
Nach dem Matching werden pro Shower aggregierte Features gebaut. Diese sind Input für die EC/PID-Modelle.

Direkter Sprung:
- Matching davor: [[06_Cluster_Matching_and_GT_Clusters]]
- EC-Verbrauch dieser Features: [[11_EnergyCorrection_DeepDive]]

## Wo im Code
- `src/utils/post_clustering_features.py`
- aufgerufen in `EnergyCorrection.clustering_and_global_features(...)`

## `get_post_clustering_features(...)`
Konzept:
- kompakte, physikalisch motivierte Shower-Zusammenfassung

Technisch (typisch):
- ECAL/HCAL Energieanteile relativ zu `sum_e`
- Anzahl Hits
- gemittelter Track-Impuls
- Energie-Dispersionen (varianzartig über `scatter_std^2`)
- Anzahl Tracks
- optional Track-`chi_squared` und muon-spezifische Features

Warum wichtig:
- reduziert Node-Level Komplexität auf Objekt-Level Signale für Regression/Klassifikation.

## `get_extra_features(...)`
Konzept:
- Zusatzfeatures für Debug/Fake-Analyse

Technisch:
- u.a. Top-k Beta pro Shower + Anzahl Nodes

## Designhinweis
Bei neuen Features immer prüfen:
1. numerische Stabilität (`nan_to_num`)
2. Konsistenz zwischen Training/Inference
3. ob charged/neutral Splitting weiter valide bleibt

Direkter Sprung:
- Erweiterungspfad für Architekturänderungen: [[17_Agent_Context_HandOff]]

## Weiterführende Links
- Determination: [[03_Determination_Step_Energy_PID]]
- EC Deep Dive: [[11_EnergyCorrection_DeepDive]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
