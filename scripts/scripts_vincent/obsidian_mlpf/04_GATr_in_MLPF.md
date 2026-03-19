# GATr in MLPF

## Konzept: Was ist GATr?
GATr ist ein Transformer auf Basis Geometric Algebra Repräsentationen.

Im MLPF-Kontext:
- räumliche Struktur (Hit-Positionen) und skalare Zusatzinfos werden gemeinsam modelliert
- pro Node entsteht eine latente Repräsentation, aus der Clusterkoordinaten/Beta oder EC-Features abgeleitet werden

Direkter Sprung:
- Wo GATr im Training hängt: [[10_ExampleWrapper_DeepDive]]
- Wo GATr im Determination-Teil hängt: [[11_EnergyCorrection_DeepDive]]

Anschaulich:
- Ein normaler Transformer liest „Wörter“.
- GATr liest stattdessen geometrische Objekte (Punkte/Richtungen) plus Messwerte und lernt deren Beziehungen.

## Wo im Code
- Clustering-Backbone:
  - `src/models/GATr/Gatr_pf_e_noise.py`
  - `self.gatr = GATr(...)` in `ExampleWrapper.__init__`
- Energy/PID:
  - `src/models/energy_correction_NN_v1.py`
  - `EnergyCorrectionWrapper` nutzt ebenfalls `GATr(...)`

## Input/Output (Clustering-Backbone)
### Input
- Punkte: `g.ndata["pos_hits_xyz"]`
- Skalare: `hit_type`, `e_hits`, `p_hits`
- Eventmaskierung: `BlockDiagonalMask` je Event im Batch

### Output
- Node-latente Einbettungen
- daraus:
  - Clusterkoordinaten (`self.clustering(...)`)
  - Beta (`self.beta(...)`)

## Technische Umsetzungsschritte
1. `embed_point(...)` + `embed_scalar(...)`
2. `embedded_inputs.unsqueeze(-2)` passend zum erwarteten Tensorformat
3. `self.gatr(embedded_inputs, scalars=..., attention_mask=...)`
4. Extraktion:
   - `extract_point(...)`
   - `extract_scalar(...)`
5. Heads auf die extrahierten Features

Mini-Beispiel:
- Input: `N` Hits eines Events
- Output (Clustering-Teil): `N x 4` (`coord3 + beta`)
- Output (EC-Teil): Shower-Embedding -> Energie/PID-Logits

## Warum block-diagonal Mask?
Ein Batch enthält mehrere unabhängige Events.
Die Maske verhindert, dass Attention über Eventgrenzen hinweg läuft.

## EC/PID-Variante
Im Determination-Step wird ebenfalls ein GATr verwendet:
- Nodefeatures der Showergraphen rein
- Aggregation auf Shower-Ebene via `scatter_sum`
- danach MLP/PID-Head

Direkter Sprung:
- Featureseite dazu: [[12_PostClustering_Features_DeepDive]]

## Weiterführende Links
- Clustering-Kontext: [[02_Clustering_Step]]
- Determination-Kontext: [[03_Determination_Step_Energy_PID]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
