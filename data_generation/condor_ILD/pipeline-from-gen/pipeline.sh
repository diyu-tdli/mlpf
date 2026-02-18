#!/usr/bin/env bash
set -e


# Wrapper function avoids passing CLI arguments to the source command
setup_key4hep_environment() {
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
}


if [[ -z "$key4hep_stack_version" ]]; then
    setup_key4hep_environment
fi
echo "Running pipeline with $key4hep_stack_version key4hep stack"


echo "Processing input from HTCondor submit file ..."
pythia_path=$1
pythia_filename="$(basename ${pythia_path})"
file_number=$(echo ${pythia_path} | sed -E 's/.*events_([0-9]+)\.hepmc/\1/')
temp_job_dir="$(pwd -P)"

# these two need to be hardcoded for the time being...
project_root="/afs/cern.ch/work/b/bdudar/mlpf"
ild_config="/afs/cern.ch/work/b/bdudar/ILDConfig"


echo "Copying input pythia input file from eos to the job node ..."
xrdcp ${pythia_path} ${temp_job_dir}


# Simulation args
compact_file="${K4GEO}/ILD/compact/ILD_sl5_v02/ILD_l5_o1_v02.xml"
input_gen_file="${temp_job_dir}/${pythia_filename}"
output_sim_file="${temp_job_dir}/${file_number}_SIM.edm4hep.root"


# Reconstruction args
input_sim_file="${output_sim_file}"
output_rec_file_base="${temp_job_dir}/${file_number}"


# Preprocessing args
input_rec_file="${temp_job_dir}/${file_number}_REC.edm4hep.root"
output_path="${temp_job_dir}"


echo "Running simulation ... "
cd ${ild_config}/StandardConfig/production
ddsim --steeringFile ddsim_steer.py --compactFile ${compact_file} --inputFiles ${input_gen_file} --outputFile ${output_sim_file}


echo "Running reconstruction ... "
cd ${ild_config}/StandardConfig/production
k4run ILDReconstruction.py --inputFiles=${input_sim_file} --outputFileBase=${output_rec_file_base}


echo "Running preprocessing ... "
cd "${project_root}/data_generation"
python3 -m preprocessing.dataset_creation_ILD --input ${input_rec_file} --outpath ${output_path}


echo "Copying output file from the job node to the eos"
xrdcp ${temp_job_dir}/*.parquet /eos/home-b/bdudar/mlpf/data/ILD/processed/zuds_from_dolores/${file_number}.parquet


echo "Removing heavy files from the temp job dir"
rm -rf ${temp_job_dir}/*.root