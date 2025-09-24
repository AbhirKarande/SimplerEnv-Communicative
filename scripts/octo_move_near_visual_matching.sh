

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

# Unique run identifier to avoid overwriting previous results
run_id=$(date +"%Y%m%d_%H%M%S")

declare -a policy_models=(
  "octo-base"
  # "octo-server"
)

env_name=MoveNearGoogleBakedTexInScene-v0
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png
# rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_2.png

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

for policy_model in "${policy_models[@]}"; do echo "$policy_model"; done


for urdf_version in "${urdf_version_arr[@]}";

do for policy_model in "${policy_models[@]}";

do \
  # Derive hierarchical logging dir per combination
  logging_dir=${LOG_DIR}/${run_id}/octo_move_near/${policy_model}/urdf_${urdf_version}

  echo "Logging to ${logging_dir}"

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs urdf_version=${urdf_version} \
  --additional-env-save-tags baked_except_bpb_orange \
  --logging-dir ${logging_dir}; # google_move_near_real_eval_1.png

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --rgb-overlay-path ${rgb_overlay_path} \
#   --robot-init-x 0.36 0.36 1 --robot-init-y 0.22 0.22 1 --obj-variation-mode episode --obj-episode-range 0 60 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.028 -0.028 1 \
#   --additional-env-build-kwargs urdf_version=${urdf_version} \
#   --additional-env-save-tags baked_except_bpb_orange; # google_move_near_real_eval_2.png

done

done
