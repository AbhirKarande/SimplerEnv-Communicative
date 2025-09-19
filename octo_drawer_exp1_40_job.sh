#!/bin/bash

#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=drawer_exp1_mc40
#SBATCH --output=logs/drawer_exp1_mc40.out
#SBATCH --error=logs/drawer_exp1_mc40.err

module purge
module load apptainer

apptainer exec \
  --bind $PWD,/scratch1/$USER \
  --nv --writable-tmpfs \
  simplerenv-octo.sif /bin/bash -c "
    . /opt/miniconda/etc/profile.d/conda.sh &&
    conda activate simpler_env &&
    cd /opt/octo &&
    git pull &&
    cd /scratch1/\$USER/SimplerEnv-Communicative &&
    chmod +x scripts/octo_drawer_exp1_40_visual_matching.sh &&
    ./scripts/octo_drawer_exp1_40_visual_matching.sh
"