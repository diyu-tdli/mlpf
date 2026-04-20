#!/bin/bash

python3 /afs/cern.ch/user/v/vriecher/condor_submit_mlpf_arc/submit_jobs_train_ARC.py \
  --sample gun \
  --cldgeo CLD_o2_v05 \
  --config config_spread_neutral.gun \
  --outdir /eos/experiment/fcc/ee/simulation/key4hep_2024_10_03/91GeV/CLD_ARC/1M_training/eval/neutral \
  --condordir /eos/experiment/fcc/ee/simulation/key4hep_2024_10_03/91GeV/CLD_ARC/1M_training/eval/neutral/condor \
  --njobs 500 \
  --nev 100 \
  --queue tomorrow \
  --cldconfig /eos/user/v/vriecher/mlpf_arc/CLDConfig_ARC/CLDConfig \
  --arc \
  --pandora
