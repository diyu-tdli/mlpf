# Object Condensation Loss

## Konzept
Object Condensation (OC) optimiert zwei gekoppelte Ziele:
1. Hits derselben Wahrheitsshower im Clusterraum zusammenziehen.
2. Hits verschiedener Objekte trennen.

Zusätzlich steuert `beta`, welche Hits repräsentative Objektkerne werden.

Direkter Sprung:
- Wo `beta` erzeugt wird: [[10_ExampleWrapper_DeepDive]]
- Clustering-Gesamtablauf: [[02_Clustering_Step]]

## Wo im Code
- Aufruf: `src/models/GATr/Gatr_pf_e_noise.py` in `training_step`
  - `object_condensation_loss2(...)`
- Implementierungsdetails:
  - `src/layers/object_cond.py`
  - zentrale interne Routine: `calc_LV_Lbeta(...)`

## Technische Kernbausteine
- `cluster_index_per_event` / `batch` bestimmen Objektzuordnung pro Event.
- Attraktive Komponente (`V_attractive`): gleiche Objekte zusammenziehen.
- Repulsive Komponente (`V_repulsive`): unterschiedliche Objekte trennen.
- `q(beta)`-Skalierung mit `qmin` stabilisiert die Kondensationsdynamik.

## Relevante Hyperparameter
- `--qmin`
- `--L_attractive_weight`
- `--L_repulsive_weight`
- `--fill_loss_weight`
- `--use-average-cc-pos`
- `--frac_cluster_loss`

Direkter Sprung:
- Flag-Kontext: [[15_Args_and_RunModes_DeepDive]]

## Im Trainingskontext
- Clustering-only Run: OC ist der Hauptloss.
- Determination-Run mit `--correction`: OC wird noch berechnet, finaler Loss wird aber im aktuellen Code auf EC/PID-Loss gesetzt.

## Weiterführende Links
- Clustering-Step: [[02_Clustering_Step]]
- Determination-Step: [[03_Determination_Step_Energy_PID]]

## Obsidian Navigation
- Hub: [[MLPF_HighLevel_Map]]
- Index: [[99_Code_Index]]
- Agent Hand-Off: [[17_Agent_Context_HandOff]]
