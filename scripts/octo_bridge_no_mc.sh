

gpu_id=0
declare -a policy_models=(
"octo-base"
# "octo-server"
)

ckpt_path=None

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# Resolve repo root and ensure imports work regardless of CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ManiSkill2_real2sim:/opt/octo:${PYTHONPATH}"
export MPLBACKEND=Agg
cd "${REPO_ROOT}"

for init_rng in 0 2 4;

do for policy_model in "${policy_models[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python -m simpler_env.main_inference --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge --octo-init-rng ${init_rng} \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-save-tags octo_init_rng_${init_rng}_nomc;

CUDA_VISIBLE_DEVICES=${gpu_id} python -m simpler_env.main_inference --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge --octo-init-rng ${init_rng} \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-save-tags octo_init_rng_${init_rng}_nomc;

CUDA_VISIBLE_DEVICES=${gpu_id} python -m simpler_env.main_inference --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge --octo-init-rng ${init_rng} \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-save-tags octo_init_rng_${init_rng}_nomc;

done

done





scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for init_rng in 0 2 4;

do for policy_model in "${policy_models[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python -m simpler_env.main_inference --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge --octo-init-rng ${init_rng} \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-save-tags octo_init_rng_${init_rng}_nomc;

done

done


