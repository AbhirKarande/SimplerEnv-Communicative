import os

import numpy as np
try:
    import tensorflow as tf  # optional; used for GPU mem limiting
except Exception:
    tf = None

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.policies.rt1.rt1_model import RT1Inference

try:
    from simpler_env.policies.octo.octo_model import OctoInference, BatchedOctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if tf is not None:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:
            # prevent a single tf process from taking up all the GPU memory
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
            )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            if getattr(args, "use_octo_batched", False):
                model = BatchedOctoInference(
                    model_type=args.ckpt_path,
                    policy_setup=args.policy_setup,
                    init_rng=args.octo_init_rng,
                    action_scale=args.action_scale,
                )
                # Store batched settings on the model for downstream use
                setattr(model, "_batched_num_mc_inferences", getattr(args, "batched_num_mc_inferences", 10))
                setattr(model, "_batched_num_samples_per_inference", getattr(args, "batched_num_samples_per_inference", 30))
                setattr(model, "_batched_experimental_setup", getattr(args, "batched_experimental_setup", 1))
                setattr(model, "_batched_random_seed", getattr(args, "batched_random_seed", 0))
            else:
                model = OctoInference(
                    model_type=args.ckpt_path,
                    policy_setup=args.policy_setup,
                    init_rng=args.octo_init_rng,
                    action_scale=args.action_scale,
                )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
