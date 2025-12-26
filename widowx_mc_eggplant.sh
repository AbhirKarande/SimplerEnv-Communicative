#!/bin/bash

#SBATCH --account=jdeshmuk_1278
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=widowx_mc_eggplant

module purge
module load apptainer

apptainer exec --bind $PWD,/scratch1/$USER --nv --writable-tmpfs simplerenv-octo.sif /bin/bash -c "export LOG_ROOT=/scratch1/\$USER/simpler_env_mc_data && . /opt/miniconda/etc/profile.d/conda.sh && conda activate simpler_env && cd /opt/octo && git pull && cd /opt/SimplerEnv-Communicative && git fetch && git pull && git checkout final_experiment_pose && cd scripts && chmod +x octo_bridge_eggplant.sh && cd .. && /opt/SimplerEnv-Communicative/scripts/octo_bridge_eggplant.sh"
