# Determination Step (Energy, Position, PID)

## Konzept
Nach dem Clustering arbeitet MLPF auf **Shower-Objekten** statt auf einzelnen Hits:
- Energiekorrektur (regressiv)
- optional Position/Richtungsgrößen
- PID-Klassifikation

Die Idee: erst robuste Objektbildung, dann Eigenschaftsbestimmung pro Objekt.

Direkter Sprung:
- Vorheriger Schritt: [[02_Clustering_Step]]
- Technische Kernimplementierung: [[11_EnergyCorrection_DeepDive]]

Anschaulich:
- Clustering baut die „Objektgrenzen“.
- Determination schreibt dann pro Objekt ein „Datenblatt“: Energie, Richtung, PID.

## Technische Kette
### Einstieg
- In `ExampleWrapper.forward(...)`:
  - falls `args.correction=True` -> `self.energy_correction.forward_correction(...)`

### Kernklasse
- Datei: `src/models/energy_correction_NN_v1.py`
- Klasse: `EnergyCorrection`

### Hauptablauf (`forward_correction`)
1. `clustering_and_global_features(...)`
2. Aufteilen in charged/neutral Showers (`num_tracks`-basiert)
3. Separate Modelle:
   - `model_charged.charged_prediction(...)`
   - `model_neutral.neutral_prediction(...)`
4. Preds zusammenführen:
   - `pred_energy_corr`
   - optional `pred_pos`, `pred_ref_pt`, `pred_PID`
5. Strukturierte Rückgabe für Training/Validation.

Direkter Sprung:
- Matching/Labeling im Detail: [[13_InferenceOC_Functions_DeepDive]]
- Feature-Bildung im Detail: [[12_PostClustering_Features_DeepDive]]

## Mini-Beispiel (Pred-Dict)
Im Positions/PID-Modus enthält die Rückgabe u.a.:
- `pred_energy_corr`: Energie pro Shower
- `pred_pos`: Richtung/Positionsvektor pro Shower
- `pred_ref_pt`: Referenzpunkt
- `pred_PID`: vorhergesagte Klasse

Anschaulich:
- Aus einer Shower-Wolke wird ein einzelner „Particle-Flow-Kandidat“ mit Attributen.

## Wichtige Teilidee: zwei Submodelle
- Charged und neutral werden getrennt behandelt.
- Motivation: unterschiedliche Messcharakteristika (Track-getrieben vs kalorimeter-getrieben).

## Loss-Seite
In `EnergyCorrection.get_loss(...)`:
- Neutral-Energy-L1-Loss (mit Filterschritten)
- PID-Loss für charged/neutral
- optional weiterer score-loss (derzeit deaktiviert)

Im `training_step` von `Gatr_pf_e_noise.py` wird bei `--correction` der finale Loss auf EC/PID gesetzt.

## Bedeutung zentraler Flags
- `--correction`: aktiviert diesen kompletten Step.
- `--freeze-clustering`: friert Clustering-Backbone ein.
- `--use-gt-clusters`: benutzt Truth-Cluster für Matching/EC-Eingang.
- `--regress-pos`, `--regress-unit-p`: erweitert Output um Richtung/Positionsinfos.
- `--restrict_PID_charge`: charged/neutral PID-Räume strikt trennen.

## Typischer Script-Kontext
- `train_properties.sh` ist ein Determination-Debug/Trainingsscript.

## Typischer Debug-Aufruf (kurz)
```bash
python -m src.train_lightning1 \
  --data-train /path/to/parquets/ \
  --data-config config_files/config_hits_track_v4.yaml \
  --network-config src/models/wrapper/example_mode_gatr_noise.py \
  --model-prefix /tmp/mlpf_debug_props/ \
  --gpus 0 --batch-size 8 --num-epochs 1 \
  --correction --freeze-clustering --use-gt-clusters \
  --regress-pos --regress-unit-p \
  --separate-PID-GATr --n-layers-PID-head 3 \
  --restrict_PID_charge --PID-4-class \
  --fetch-by-files --fetch-step 1 --train-batches 50
```

## Weiterführende Links
- Matching und GT-Cluster: [[06_Cluster_Matching_and_GT_Clusters]]
- Clustering davor: [[02_Clustering_Step]]
- GATr im EC-Head: [[04_GATr_in_MLPF]]
- EnergyCorrection Deep Dive: [[11_EnergyCorrection_DeepDive]]
- Feature-Details: [[12_PostClustering_Features_DeepDive]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
