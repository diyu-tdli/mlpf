# Deep Dive: `inference_oc.py` Kernfunktionen

## Rolle
Dieses Modul enthält die Laufzeitlogik für:
- Labeling von Clustern
- Matching predicted showers gegen truth showers
- Erzeugen persistierter Auswerte-DataFrames

Direkter Sprung:
- Clustering-Kontext: [[02_Clustering_Step]]
- Determination-Kontext: [[03_Determination_Step_Energy_PID]]

## `DPC_custom_CLD(X, g, device)`
Konzept:
- Dichtebasierte Clusterzuweisung im Clusterraum

Technisch:
- Distanzmatrix `D`
- lokale Dichte gewichtet mit Hit-Energie
- Center-Selektion (`rho_min`, `delta_min`)
- Core-Points + Labelmapping, Label 0 für Noise

Nutzen:
- robuste Clusterlabels ohne feste Clusterzahl.

Direkter Sprung:
- Neuer Clusterer einbauen: [[16_Extending_New_Clustering_Method]]

## `remove_bad_tracks_from_cluster(...)`
Konzept:
- fehlerhafte Trackzuordnung aus Clustern entfernen

Technisch:
- prüft z.B. Inkonsistenz zwischen Clusterenergie und Track-Impuls
- setzt schlechte Tracknodes auf Label 0 (Noise)

## `match_showers(...)`
Konzept:
- ordnet predicted showers den truth-Partikeln zu

Technisch:
1. Intersections/Unions bauen
2. IoU-Matrix berechnen
3. IoU thresholding
4. Hungarian matching (`linear_sum_assignment`)

Output:
- matched index-paare (`row_ind`, `col_ind`)
- IoU/Hitgewichtsinformationen

Direkter Sprung:
- Matching-High-Level: [[06_Cluster_Matching_and_GT_Clusters]]

## `create_and_store_graph_output(...)`
Konzept:
- end-to-end Auswerteobjekte pro Batch/Event erzeugen

Technisch:
- pro Event labeln (pred/gt/pandora)
- matchen
- Shower-DataFrames bauen
- optional speichern via `store_at_batch_end(...)`

## Warum für Erweiterungen zentral?
Wenn neues Clustering eingeführt wird, ist dies die Integrationsstelle für:
- Labeler austauschen
- Matching-Kriterien ändern
- Metriken/Ausgabeformat erweitern

## Weiterführende Links
- Matching Überblick: [[06_Cluster_Matching_and_GT_Clusters]]
- Neue Clustering-Methoden einbauen: [[16_Extending_New_Clustering_Method]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
