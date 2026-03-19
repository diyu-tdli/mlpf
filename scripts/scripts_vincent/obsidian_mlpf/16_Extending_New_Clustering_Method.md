# Extending: New Clustering Method

## Ziel
Diese Seite beschreibt, wie man eine neue Clusteringmethode (z.B. statt DPC_custom_CLD) sauber integriert.

Direkter Sprung:
- Ausgangsarchitektur: [[MLPF_HighLevel_Map]]
- Kern-Labelingcode: [[13_InferenceOC_Functions_DeepDive]]

## Konzeptuelle Integrationspunkte
1. **Label-Erzeugung** aus Node-Embeddings (`coords`, `beta`).
2. **Matching** gegen Truth (`match_showers` bleibt meist gleich).
3. **Weitergabe** an Determination-Pfad (`obtain_clustering_for_matched_showers`).

## Technische Integrationsstellen
### A) Inferenz-/Matching-Ebene
- Datei: `src/layers/inference_oc.py`
- aktuelle Labeler:
  - `DPC_custom_CLD`
  - `clustering_obtain_labels`
  - `hfdb_obtain_labels`

Vorgehen:
1. Neue Funktion hinzufügen, z.B. `my_new_cluster_labels(X, g, device, ...)`.
2. In Aufrufern austauschbar machen:
   - `obtain_clustering_for_matched_showers(...)` (in `utils_training.py`)
   - `create_and_store_graph_output(...)` (in `inference_oc.py`)

Beispiel-Skizze:
```python
if args.cluster_algo == "myalgo":
    labels = my_new_cluster_labels(X, g, device)
elif args.cluster_algo == "dpc":
    labels = DPC_custom_CLD(X, g, device)
elif args.cluster_algo == "hdbscan":
    labels = hfdb_obtain_labels(X, device)
```

### B) Konfigurierbar per Flag machen
- Datei: `src/utils/parser_args.py`
- neuen Flag ergänzen, z.B. `--cluster-algo {dpc,hdbscan,myalgo}`
- dann in den obigen Aufrufern branchen.

### C) Trainingskonsistenz sicherstellen
- `train` und `predict` Pfad sollen identischen Labeler nutzen (außer explizit anders gewünscht).
- bei `--use-gt-clusters` den neuen Labeler bewusst umgehen.

## Qualitätscheckliste bei neuer Methode
1. Labelkonvention konsistent (`0 = noise`).
2. Stabil bei kleinen/rauschigen Events.
3. Matching-IoU nicht regressiv verschlechtert.
4. Determination-Loss bleibt stabil (keine NaNs in Features/Indices).

Direkter Sprung:
- Determination-Losskontext: [[11_EnergyCorrection_DeepDive]]

## Minimaler Implementierungsplan
1. Neuen Labeler in `inference_oc.py` implementieren.
2. Flag in `parser_args.py` hinzufügen.
3. Branches in `utils_training.py` und `inference_oc.py` einbauen.
4. Kurzer Debug-Run mit `--train-batches` klein.
5. Vergleich mit Baseline-Labeler.

## Anschauliche Denkweise für neue Clusterer
- Clustering ist hier ein „Adapter“ zwischen Node-Level-Netz und Shower-Level-Regression.
- Solange dein Clusterer stabile Labels liefert und die Noise-Konvention hält, bleibt der Rest der Pipeline weitgehend gleich.

## Weiterführende Links
- Matching-Details: [[13_InferenceOC_Functions_DeepDive]]
- Args/RunModes: [[15_Args_and_RunModes_DeepDive]]
- Agent-Übergabeseite: [[17_Agent_Context_HandOff]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
