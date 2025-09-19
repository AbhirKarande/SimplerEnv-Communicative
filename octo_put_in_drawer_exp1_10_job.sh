#!/bin/bash

#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=putindrawer_exp2_mc10
#SBATCH --output=logs/putindrawer_exp2_mc10.out
#SBATCH --error=logs/putindrawer_exp2_mc10.err

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
    chmod +x scripts/octo_put_in_drawer_exp2_10_visual_matching.sh &&
    ./scripts/octo_put_in_drawer_exp2_10_visual_matching.sh
"