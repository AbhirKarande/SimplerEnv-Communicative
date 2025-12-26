#!/bin/bash

gpu_id=0
declare -a policy_models=(
"octo-base"
)

ckpt_path=None
EXP_SETUP=${EXP_SETUP:-1}
MC_PASSES=${MC_PASSES:-40}
SAMPLES_PER_INFERENCE=${SAMPLES_PER_INFERENCE:-30}

# Prefer persistent scratch for logs if available
if [ -d "/scratch1/$USER" ] && [ -w "/scratch1/$USER" ]; then
LOG_ROOT="/scratch1/$USER"
elif [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH:-}" ] && [ -w "${SCRATCH:-}" ]; then
LOG_ROOT="${SCRATCH}"
else
LOG_ROOT=${LOG_ROOT:-${TMPDIR:-/tmp}}
fi
LOG_DIR="$LOG_ROOT/simpler_env_results"
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Resolve repo root and ensure imports work regardless of CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ManiSkill2_real2sim:/opt/octo:${PYTHONPATH}"
export MPLBACKEND=Agg
cd "${REPO_ROOT}"

# Try to activate container conda env if available and isolate from host
if [ -f "/opt/miniconda/etc/profile.d/conda.sh" ]; then
    . /opt/miniconda/etc/profile.d/conda.sh
    conda activate simpler_env || true
fi
export PYTHONNOUSERSITE=1

# Ensure JAX/Flax compatibility (KeyArray present); install if missing
python - <<'PY'
try:
    import jax, jaxlib, flax  # noqa: F401
    from jax.random import KeyArray  # noqa: F401
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
                          "jax==0.4.26", "jaxlib==0.4.26", "flax==0.8.1"])
PY

echo "=========================================="
echo "MC DATA COLLECTION: PutSpoon"
echo "MC Passes: ${MC_PASSES}"
echo "Samples per inference: ${SAMPLES_PER_INFERENCE}"
echo "Experimental setup: ${EXP_SETUP}"
echo "=========================================="

# Scene 1: bridge_table_1_v1
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

for init_rng in 0 2 4;

do for policy_model in "${policy_models[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python -m simpler_env.main_inference --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge --octo-init-rng ${init_rng} \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-save-tags octo_init_rng_${init_rng}_mc${MC_PASSES} \
  --use-octo-batched --batched-experimental-setup ${EXP_SETUP} --batched-num-mc-inferences ${MC_PASSES} --batched-num-samples-per-inference ${SAMPLES_PER_INFERENCE} --mc-logging --logging-dir "$LOG_DIR" \
  --min-success-episodes 30;

done

done

echo "=========================================="
echo "PutSpoon MC data collection COMPLETED"
echo "Check logs at: $LOG_DIR"
echo "=========================================="
