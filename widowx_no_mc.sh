#!/bin/bash
  
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module purge
module load apptainer

apptainer exec --bind $PWD,/scratch1/$USER --nv --writable-tmpfs simplerenv-octo.sif /bin/bash -c ". /opt/miniconda/etc/profile.d/conda.sh && unset CONDA_PREFIX CONDA_DEFAULT_ENV && export PYTHONNOUSERSITE=1 && conda activate simpler_env && cd /opt/octo && git pull && cd /opt/SimplerEnv-Communicative && git fetch && git pull && git checkout final_experiment && export PYTHONPATH=/opt/SimplerEnv-Communicative:/opt/SimplerEnv-Communicative/ManiSkill2_real2sim:/opt/octo:\$PYTHONPATH && export MPLBACKEND=Agg && python -c 'import jax, jaxlib, flax; print("jax", jax.__version__, "jaxlib", jaxlib.__version__, "flax", flax.__version__)' || python -m pip install --no-cache-dir -q jax==0.4.26 jaxlib==0.4.26 flax==0.8.1 && cd scripts && chmod +x octo_bridge_no_mc.sh && cd .. && /opt/SimplerEnv-Communicative/scripts/octo_bridge_no_mc.sh"