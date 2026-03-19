# Data, Dataloader and Graph Building

## Konzept
Die Pipeline liest Parquet-Events und konvertiert sie in DGL-Graphs:
- Nodes = Hits/Tracks
- Nodefeatures enthalten Geometrie, Typ, Energie/Impuls, Zusatzinfos

Direkter Sprung:
- Training-Orchestrierung darüber: [[01_Training_Entry_and_Pipeline]]
- Nutzung im Clustering-Forward: [[10_ExampleWrapper_DeepDive]]

## Technische Kette
1. `train_utils.train_load(args)`
2. `SimpleIterDataset` (iterables, async prefetch)
3. `create_graph(...)` in `src/dataset/functions_graph.py`
4. `graph_batch_func(...)` für DGL-Batching + GT-Container

## Wichtige Implementationdetails
### SimpleIterDataset
- Datei: `src/dataset/dataset.py`
- asynchrones Laden über `ThreadPoolExecutor`
- `fetch_by_files`/`fetch_step` steuern I/O-Verhalten
- `infinity_mode` für steps-per-epoch basiertes Training

### Graphaufbau
- Datei: `src/dataset/functions_graph.py`
- zentrale Felder:
  - `g.ndata["h"] = [pos_xyz, hit_type_onehot, e_hits, p_hits]`
  - `g.ndata["particle_number"] = hit_particle_link + 1`
  - `g.ndata["chi_squared_tracks"]` (wenn verfügbar)

### Label-Container
- `Particles_GT` wird parallel gehalten und beim Batching zusammengeführt.

## Konfigurationsdatei
- `config_files/config_hits_track_v4.yaml` definiert Inputs/Variablen auf Datenebene.
- Praktisch wichtig: `train_lightning1.py` erwartet train-seitig `*.parquet` im angegebenen Ordner.

Direkter Sprung:
- Pfad-/Run-Beispiele: [[08_Path_Mapping_Cluster_Configs]]

## Warum das wichtig ist
Viele Trainingsprobleme sind in Wirklichkeit Daten-/Graph-Build-Probleme:
- falsche Pfade
- falsche Dateiformate
- unerwartete Featureverteilungen

## Weiterführende Links
- Pipeline-Einstieg: [[01_Training_Entry_and_Pipeline]]
- Clustering-Step: [[02_Clustering_Step]]
- Determination-Step: [[03_Determination_Step_Energy_PID]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
