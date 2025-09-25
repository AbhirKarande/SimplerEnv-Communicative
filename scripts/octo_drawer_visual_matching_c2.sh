# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth



declare -a policy_models=(
"octo-base"
# "octo-server"
)

declare -a env_names=(
CloseTopDrawerCustomInScene-v0
CloseMiddleDrawerCustomInScene-v0
CloseBottomDrawerCustomInScene-v0
)

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

for urdf_version in "${urdf_version_arr[@]}"; do

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

# Unique run identifier to avoid overwriting previous results (declared once per script run)
if [ -z "${run_id:-}" ]; then
run_id=$(date +"%Y%m%d_%H%M%S")
fi

EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version}"

EvalOverlay() {
# # A0
#   # Derive hierarchical logging dir per combination
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # A1
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # A2
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.889 0.889 1 --robot-init-y -0.203 -0.203 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.06 -0.06 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # B0
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # B1
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.752 0.752 1 --robot-init-y 0.009 0.009 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # B2
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.851 0.851 1 --robot-init-y 0.035 0.035 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # C0
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# # C1
#   logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

# python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
#   --env-name ${env_name} --scene-name dummy_drawer \
#   --robot-init-x 0.765 0.765 1 --robot-init-y 0.222 0.222 1 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
#   --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
#   --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png \
#   ${EXTRA_ARGS} \
#   --logging-dir ${logging_dir}

# C2
  logging_dir=${LOG_DIR}/${run_id}/octo_drawer/${policy_model}/${env_name}/urdf_${urdf_version}

python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.865 0.865 1 --robot-init-y 0.222 0.222 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png \
  ${EXTRA_ARGS} \
  --logging-dir ${logging_dir}
}


for policy_model in "${policy_models[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalOverlay
  done
done


done
