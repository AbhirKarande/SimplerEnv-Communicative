#!/bin/bash
  
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=drawer_exp1_mc10
#SBATCH --output=logs/drawer_exp1_mc10.out
#SBATCH --error=logs/drawer_exp1_mc10.err

module purge
module load apptainer

apptainer exec --bind $PWD,/scratch1/$USER --nv --writable-tmpfs simplerenv-octo.sif /bin/bash -c ". /opt/miniconda/etc/profile.d/conda.sh && conda activate simpler_env && cd /opt/octo && git pull && cd /opt/SimplerEnv-Communicative && git fetch && git pull && git checkout experiment1_drawer_place_apple &&  cd scripts && chmod +x octo_drawer_exp1_10_visual_matching.sh && cd .. && /opt/SimplerEnv-Communicative/scripts/octo_drawer_exp1_10_visual_matching.sh"