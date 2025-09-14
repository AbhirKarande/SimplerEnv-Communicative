#!/bin/bash
  
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=9:00:00

module purge
module load apptainer

apptainer exec --nv --writable-tmpfs simplerenv.sif /bin/bash -c ". /opt/miniconda/etc/profile.d/conda.sh && conda activate simpler_env && cd /opt/SimplerEnv-Communicative && git pull && cd scripts && chmod +x octo_move_near_visual_matching.sh && cd .. && /opt/SimplerEnv-Communicative/scripts/octo_move_near_visual_matching.sh"

