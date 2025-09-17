# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close / place-into-closed-drawer tasks as policies often rely on shadows to infer depth



gpu_id=0

# Set a logging directory (prefer persistent HPC scratch if available)
if [ -d "/scratch1/$USER" ] && [ -w "/scratch1/$USER" ]; then
LOG_ROOT="/scratch1/$USER"
elif [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH:-}" ] && [ -w "${SCRATCH:-}" ]; then
LOG_ROOT="${SCRATCH}"
else
LOG_ROOT=${LOG_ROOT:-${TMPDIR:-/tmp}}
fi
LOG_DIR="$LOG_ROOT/simpler_env_results"
mkdir -p "$LOG_DIR" 2>/dev/null || true

EXP_SETUP=${EXP_SETUP:-2}                # 1 or 2
MC_PASSES=${MC_PASSES:-40}               # e.g., 10, 20, 40
SAMPLES_PER_INFERENCE=${SAMPLES_PER_INFERENCE:-30}
declare -a policy_models=(
"octo-base"
# "octo-server"
)

declare -a env_names=(
PlaceIntoClosedTopDrawerCustomInScene-v0
# PlaceIntoClosedMiddleDrawerCustomInScene-v0
# PlaceIntoClosedBottomDrawerCustomInScene-v0
)

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

for urdf_version in "${urdf_version_arr[@]}"; do

EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} model_ids=baked_apple_v2"

EvalOverlay() {
# A0
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
  ${EXTRA_ARGS} --use-octo-batched --batched-experimental-setup ${EXP_SETUP} --batched-num-mc-inferences ${MC_PASSES} --batched-num-samples-per-inference ${SAMPLES_PER_INFERENCE} --mc-logging --logging-dir "$LOG_DIR"

# A1
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png \
  ${EXTRA_ARGS} --use-octo-batched --batched-experimental-setup ${EXP_SETUP} --batched-num-mc-inferences ${MC_PASSES} --batched-num-samples-per-inference ${SAMPLES_PER_INFERENCE} --mc-logging --logging-dir "$LOG_DIR"

# B0
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
  ${EXTRA_ARGS} --use-octo-batched --batched-experimental-setup ${EXP_SETUP} --batched-num-mc-inferences ${MC_PASSES} --batched-num-samples-per-inference ${SAMPLES_PER_INFERENCE} --mc-logging --logging-dir "$LOG_DIR"

# C0
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
  ${EXTRA_ARGS} --use-octo-batched --batched-experimental-setup ${EXP_SETUP} --batched-num-mc-inferences ${MC_PASSES} --batched-num-samples-per-inference ${SAMPLES_PER_INFERENCE} --mc-logging --logging-dir "$LOG_DIR"
}


for policy_model in "${policy_models[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalOverlay
  done
done


done


