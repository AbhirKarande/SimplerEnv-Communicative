"""
Evaluate a model on ManiSkill2 environment.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler, euler2axangle

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.evaluation.instruction_refinement import InstructionRefiner
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video


def _ensure_serializable(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.int32, _np.int64)):
            return int(obj)
        if isinstance(obj, (_np.float32, _np.float64)):
            return float(obj)
    if isinstance(obj, dict):
        return {k: _ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_serializable(v) for v in obj]
    try:
        return obj
    except Exception:
        return str(obj)


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    episode_idx=None,
    total_episodes=None,
    mc_logging=False,
    instruction_refine_procedure: int = 0,
    instruction_refine_task: str = "auto",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False
    trajectory = [] if mc_logging else None

    # Initialize model
    model.reset(task_description)

    # Initialize optional instruction refiner
    refiner = None
    if instruction_refine_procedure in (1, 2):
        # Decide task kind if set to auto
        if instruction_refine_task == "auto":
            # Simple heuristic: map by env_name
            if "pick" in env_name.lower() or "coke" in env_name.lower():
                task_kind = "pick_coke_can"
            elif "drawer" in env_name.lower():
                task_kind = "close_drawer"
            else:
                task_kind = "pick_coke_can"
        else:
            task_kind = instruction_refine_task

        # task_name string passed to LLM (human-readable)
        task_name_for_llm = env_name
        refiner = InstructionRefiner(
            task_kind=task_kind,
            procedure=instruction_refine_procedure,
            logging_dir=logging_dir,
            model_type=str(getattr(model, "policy_setup", "google_robot")),
            task_name=task_name_for_llm,
        )

    # Whether to honor env-provided instruction updates
    allow_env_updates = True

    # Initialize RNG for experimental setup 2 (uniform selection across MC passes)
    batched_choice_rng = None
    batched_experimental_setup = None
    if getattr(model, "batch_step", None) is not None:
        batched_experimental_setup = getattr(model, "_batched_experimental_setup", 1)
        batched_random_seed = getattr(model, "_batched_random_seed", 0)
        if batched_experimental_setup == 2:
            batched_choice_rng = np.random.RandomState(batched_random_seed)

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):
        use_batched = getattr(model, "batch_step", None) is not None and hasattr(env, "step") and hasattr(model, "action_scale")
        if use_batched:
            # Batched sampling for epistemic/aleatoric estimates
            # Access settings attached to the model by main_inference
            num_mc_inferences = getattr(model, "_batched_num_mc_inferences", 10)
            num_samples_per_inference = getattr(model, "_batched_num_samples_per_inference", 30)
            experimental_setup = batched_experimental_setup if batched_experimental_setup is not None else getattr(model, "_batched_experimental_setup", 1)

            all_results = []
            for i in range(num_mc_inferences):
                current_image = image if i == 0 else None
                results = model.batch_step(current_image, num_samples_per_inference, task_description)
                all_results.append(results)

            # For each MC pass, compute mean action and per-dimension entropy across 30 samples
            per_pass_mean_actions = []
            per_pass_mean_entropies = []
            per_pass_wv_all = []
            per_pass_rot_all = []
            per_pass_grip_all = []
            # Also collect forward-pass actions (per MC pass mean actions) for logging
            forward_pass_actions = []
            epsilon = 1e-8
            for one_inference_results in all_results:
                per_pass_wv = np.stack([res[0]["world_vector"] for res in one_inference_results])
                per_pass_rot = np.stack([res[0]["rotation_delta"] for res in one_inference_results])
                per_pass_grip = np.stack([res[0]["open_gripper"] for res in one_inference_results])

                per_pass_wv_all.append(per_pass_wv)
                per_pass_rot_all.append(per_pass_rot)
                per_pass_grip_all.append(per_pass_grip)

                mean_wv = np.mean(per_pass_wv, axis=0)
                mean_rot = np.mean(per_pass_rot, axis=0)
                mean_grip = np.mean(per_pass_grip, axis=0)

                var_wv = np.var(per_pass_wv, axis=0)
                var_rot = np.var(per_pass_rot, axis=0)
                var_grip = np.var(per_pass_grip, axis=0)

                entropy = {
                    "world_vector": 0.5 * np.log(2 * np.pi * np.e * (var_wv + epsilon)),
                    "rotation_delta": 0.5 * np.log(2 * np.pi * np.e * (var_rot + epsilon)),
                    "open_gripper": 0.5 * np.log(2 * np.pi * np.e * (var_grip + epsilon)),
                }

                mean_action_dict = {
                    "world_vector": mean_wv,
                    "rotation_delta": mean_rot,
                    "open_gripper": mean_grip,
                }
                per_pass_mean_actions.append(mean_action_dict)
                per_pass_mean_entropies.append(entropy)
                forward_pass_actions.append(mean_action_dict)

            # Compute total, aleatoric, epistemic entropies for logging
            all_raw_actions_world_vector = np.concatenate(per_pass_wv_all, axis=0)
            all_raw_rot_delta = np.concatenate(per_pass_rot_all, axis=0)
            all_raw_grip_open = np.concatenate(per_pass_grip_all, axis=0)

            total_var_wv = np.var(all_raw_actions_world_vector, axis=0)
            total_var_rot = np.var(all_raw_rot_delta, axis=0)
            total_var_grip = np.var(all_raw_grip_open, axis=0)
            total_entropy = {
                "world_vector": 0.5 * np.log(2 * np.pi * np.e * (total_var_wv + epsilon)),
                "rotation_delta": 0.5 * np.log(2 * np.pi * np.e * (total_var_rot + epsilon)),
                "open_gripper": 0.5 * np.log(2 * np.pi * np.e * (total_var_grip + epsilon)),
            }

            per_inference_means_wv = np.stack([a["world_vector"] for a in per_pass_mean_actions])
            per_inference_means_rot = np.stack([a["rotation_delta"] for a in per_pass_mean_actions])
            per_inference_means_grip = np.stack([a["open_gripper"] for a in per_pass_mean_actions])
            epistemic_vars_wv = np.var(per_inference_means_wv, axis=0)
            epistemic_vars_rot = np.var(per_inference_means_rot, axis=0)
            epistemic_vars_grip = np.var(per_inference_means_grip, axis=0)
            epistemic_entropy = {
                "world_vector": 0.5 * np.log(2 * np.pi * np.e * (epistemic_vars_wv + epsilon)),
                "rotation_delta": 0.5 * np.log(2 * np.pi * np.e * (epistemic_vars_rot + epsilon)),
                "open_gripper": 0.5 * np.log(2 * np.pi * np.e * (epistemic_vars_grip + epsilon)),
            }

            per_inference_vars_wv = np.mean([np.var(arr, axis=0) for arr in per_pass_wv_all], axis=0)
            per_inference_vars_rot = np.mean([np.var(arr, axis=0) for arr in per_pass_rot_all], axis=0)
            per_inference_vars_grip = np.mean([np.var(arr, axis=0) for arr in per_pass_grip_all], axis=0)
            aleatoric_entropy = {
                "world_vector": 0.5 * np.log(2 * np.pi * np.e * (per_inference_vars_wv + epsilon)),
                "rotation_delta": 0.5 * np.log(2 * np.pi * np.e * (per_inference_vars_rot + epsilon)),
                "open_gripper": 0.5 * np.log(2 * np.pi * np.e * (per_inference_vars_grip + epsilon)),
            }

            if experimental_setup == 1:
                # Setup 1: mean over all MC passes and samples
                mean_world_vector = np.mean([a["world_vector"] for a in per_pass_mean_actions], axis=0)
                mean_rotation_delta = np.mean([a["rotation_delta"] for a in per_pass_mean_actions], axis=0)
                mean_open_gripper = np.mean([a["open_gripper"] for a in per_pass_mean_actions], axis=0)

                raw_action = {
                    "world_vector": mean_world_vector,
                    "rotation_delta": mean_rotation_delta,
                    "open_gripper": mean_open_gripper,
                }
                selected_entropy = {
                    "world_vector": np.mean([e["world_vector"] for e in per_pass_mean_entropies], axis=0),
                    "rotation_delta": np.mean([e["rotation_delta"] for e in per_pass_mean_entropies], axis=0),
                    "open_gripper": np.mean([e["open_gripper"] for e in per_pass_mean_entropies], axis=0),
                }
            else:
                # Setup 2: mean per pass, then uniformly pick one pass's mean action
                rng = batched_choice_rng if batched_choice_rng is not None else np.random
                chosen_idx = rng.randint(0, len(per_pass_mean_actions))
                raw_action = per_pass_mean_actions[chosen_idx]
                selected_entropy = per_pass_mean_entropies[chosen_idx]

            # Convert to environment action (mirrors OctoInference.step)
            action = {}
            action["world_vector"] = raw_action["world_vector"] * model.action_scale
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            r, p, y = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(r, p, y)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle * model.action_scale

            if getattr(model, "policy_setup", "google_robot") == "google_robot":
                current_gripper_action = raw_action["open_gripper"]
                if model.previous_gripper_action is None:
                    relative_gripper_action = np.array([0])
                else:
                    relative_gripper_action = model.previous_gripper_action - current_gripper_action
                model.previous_gripper_action = current_gripper_action
                if np.abs(relative_gripper_action) > 0.5 and not model.sticky_action_is_on:
                    model.sticky_action_is_on = True
                    model.sticky_gripper_action = relative_gripper_action
                if model.sticky_action_is_on:
                    model.gripper_action_repeat += 1
                    relative_gripper_action = model.sticky_gripper_action
                if model.gripper_action_repeat == model.sticky_gripper_num_repeat:
                    model.sticky_action_is_on = False
                    model.gripper_action_repeat = 0
                    model.sticky_gripper_action = 0.0
                action["gripper"] = relative_gripper_action
            else:
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

            action["terminate_episode"] = np.array([0.0])
            # Store for visualization
            predicted_actions.append(raw_action)

            # Instruction refinement (batched path uses raw_action dict with rotation_delta/world_vector)
            if refiner is not None:
                try:
                    new_task_description, did_refine = refiner.maybe_refine(
                        raw_action=raw_action,
                        timestep=timestep,
                        current_instruction=task_description,
                        image_np=image,
                    )
                    if did_refine and isinstance(new_task_description, str) and new_task_description:
                        task_description = new_task_description
                        # Update model with refined instruction
                        model.reset(task_description)
                        print(f"[Refined Instruction @ t={timestep}] {task_description}")
                        allow_env_updates = False
                except Exception as e:
                    print(f"Refinement error: {e}")

            # Per-timestep JSON logging (only action and raw_action)
            if mc_logging:
                timestep_log = {
                    "timestep": timestep,
                    "action": _ensure_serializable({
                        "world_vector": action["world_vector"],
                        "rot_axangle": action["rot_axangle"],
                        "gripper": action["gripper"],
                    }),
                    "raw_action": _ensure_serializable({
                        "world_vector": raw_action["world_vector"],
                        "rotation_delta": raw_action["rotation_delta"],
                        "open_gripper": raw_action["open_gripper"],
                    }),
                    "current_instruction": task_description,
                    "info": _ensure_serializable(info) if 'info' in locals() else {},
                }
                trajectory.append(timestep_log)

            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
        else:
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            step_output = model.step(image, task_description)
            raw_action, action = step_output[0], step_output[1]
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            # Instruction refinement (non-batched path has same raw_action structure)
            if refiner is not None:
                try:
                    new_task_description, did_refine = refiner.maybe_refine(
                        raw_action=raw_action,
                        timestep=timestep,
                        current_instruction=task_description,
                        image_np=image,
                    )
                    if did_refine and isinstance(new_task_description, str) and new_task_description:
                        task_description = new_task_description
                        model.reset(task_description)
                        print(f"[Refined Instruction @ t={timestep}] {task_description}")
                        allow_env_updates = False
                except Exception as e:
                    print(f"Refinement error: {e}")

            # step the environment
            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
        
        success = "success" if done else "failure"
        if allow_env_updates:
            new_task_description = env.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description
                print(task_description)
        is_final_subtask = env.is_final_subtask()

        if episode_idx is not None:
            if total_episodes is not None:
                print(f"Episode {episode_idx}/{total_episodes} | timestep {timestep}", info)
            else:
                print(f"Episode {episode_idx} | timestep {timestep}", info)
        else:
            print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # mc-style JSON and video saving
    if mc_logging and total_episodes is not None:
        exp_tag = f"exp{getattr(model, '_batched_experimental_setup', 1)}"
        num_mc = getattr(model, "_batched_num_mc_inferences", 10)
        mc_root_base = f"mc_dropout_{env_name}_{total_episodes}_episodes_{num_mc}_forward_passes_{exp_tag}"
        # add subdirectories to avoid overwriting across URDF variants and overlay presets
        try:
            overlay_tag = os.path.splitext(os.path.basename(rgb_overlay_path))[0] if rgb_overlay_path is not None else "None"
        except Exception:
            overlay_tag = "None"
        try:
            urdf_version = additional_env_build_kwargs.get("urdf_version", "None") if isinstance(additional_env_build_kwargs, dict) else "None"
        except Exception:
            urdf_version = "None"
        mc_root = os.path.join(mc_root_base, f"urdf_{urdf_version}", f"overlay_{overlay_tag}")
        json_dir = os.path.join(logging_dir, mc_root, "json")
        video_dir = os.path.join(logging_dir, mc_root, "video")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        success_bool = (success == "success")
        episode_zero_based = (episode_idx - 1) if episode_idx is not None else 0
        json_path = os.path.join(json_dir, f"{str(success_bool)}_{episode_zero_based}.json")
        video_mc_path = os.path.join(video_dir, f"{str(success_bool)}_{episode_zero_based}.mp4")

        try:
            if trajectory is not None:
                import json as _json
                with open(json_path, "w") as f:
                    _json.dump(trajectory, f, indent=2)
            write_video(video_mc_path, images, fps=5)
        except Exception as e:
            print(f"Error saving MC-style episode data or video: {e}")

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # precompute total number of episodes and track episode index (1-based)
    n_rx = len(args.robot_init_xs)
    n_ry = len(args.robot_init_ys)
    n_rq = len(args.robot_init_quats)
    if args.obj_variation_mode == "xy":
        n_ox = len(args.obj_init_xs)
        n_oy = len(args.obj_init_ys)
        total_episodes = n_rx * n_ry * n_rq * n_ox * n_oy
    elif args.obj_variation_mode == "episode":
        total_episodes = n_rx * n_ry * n_rq * (args.obj_episode_range[1] - args.obj_episode_range[0])
    else:
        raise NotImplementedError()
    episode_idx = 0

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    mc_logging=getattr(args, "mc_logging", False) and getattr(model, "batch_step", None) is not None,
                    instruction_refine_procedure=getattr(args, "instruction_refine_procedure", 0),
                    instruction_refine_task=getattr(args, "instruction_refine_task", "auto"),
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            episode_idx += 1
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    episode_idx=episode_idx,
                                    total_episodes=total_episodes,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        episode_idx += 1
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id,
                                episode_idx=episode_idx,
                                total_episodes=total_episodes,
                                **kwargs,
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr
