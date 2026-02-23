#!/bin/bash

# gun
# python submit_jobs_train_ARC.py \
#   --sample gun \
#   --cldgeo CLD_o2_v05 \
#   --config config_spread_eCH.gun \
#   --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_perfect_tracking_CHe_3/ \
#   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_perfect_tracking_CHe_3/ \
#   --njobs 2 \
#   --nev 20 \
#   --queue tomorrow \
#   --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/


# pythia
python submit_jobs_train_ARC.py \
  --sample Zcard \
  --cldgeo CLD_o2_v05 \
  --config p8_ee_Zuds_ecm91 \
  --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_uds_clustering_dataset_2/ \
  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Z_uds_clustering_dataset_2/ \
  --njobs 2 \
  --nev 1 \
  --queue tomorrow \
  --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/ # \
#   --arc \
#   --gentracking