#!/bin/bash
  
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00

module purge
module load apptainer

apptainer exec --bind $PWD,/scratch1/$USER --nv --writable-tmpfs simplerenv-octo.sif /bin/bash -c ". /opt/miniconda/etc/profile.d/conda.sh && conda activate simpler_env && cd /opt/octo && git checkout main && git pull && cd /opt/SimplerEnv-Communicative && git fetch && git pull && git checkout from-4ab7178 &&  cd scripts && chmod +x octo_drawer_visual_matching.sh && cd .. && /opt/SimplerEnv-Communicative/scripts/octo_drawer_visual_matching.sh"