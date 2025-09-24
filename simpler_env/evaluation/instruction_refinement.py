import csv
import os
from typing import List, Optional, Tuple

import numpy as np


TRY_INCLUDE_IMAGE = True


class InstructionRefiner:
    """
    Computes per-timestep distances to success/failure action distributions and
    decides when to trigger instruction refinement via llm_instruction.

    Supported tasks: "pick_coke_can", "close_drawer".
    Procedures:
      1) Rule: Df < Ds for all orientation dims; trigger after 3 consecutive steps
      2) Rule: (Df < Ds for all positional dims) OR (Df < Ds for all orientation dims);
         trigger after 5 consecutive steps
    """

    def __init__(
        self,
        task_kind: str,
        procedure: int,
        logging_dir: str,
        model_type: str,
        task_name: str,
    ) -> None:
        assert task_kind in ("pick_coke_can", "close_drawer")
        assert procedure in (1, 2)
        self.task_kind = task_kind
        self.procedure = procedure
        self.logging_dir = logging_dir
        self.model_type = model_type
        self.task_name = task_name

        self._eps = 1e-8
        self._consec_needed = 3 if procedure == 1 else 5
        self._consec_counter = 0

        # Preload distributions
        (
            self.succ_pos_mean,
            self.succ_pos_std,
            self.succ_rot_mean,
            self.succ_rot_std,
            self.fail_pos_mean,
            self.fail_pos_std,
            self.fail_rot_mean,
            self.fail_rot_std,
        ) = self._load_distributions(task_kind)

        # Names for reporting
        self.pos_dim_names = ["world_x", "world_y", "world_z"]
        self.rot_dim_names = ["rot_x", "rot_y", "rot_z"]

        # Prepare frame export directory
        self._frames_dir = os.path.join(self.logging_dir, "refine_frames")
        try:
            os.makedirs(self._frames_dir, exist_ok=True)
        except Exception:
            pass

    def _csv_path(self, task_kind: str) -> str:
        here = os.path.dirname(__file__)
        if task_kind == "pick_coke_can":
            return os.path.join(here, "pick_coke_can_distribution.csv")
        else:
            return os.path.join(here, "close_drawer_distribution.csv")

    def _segments(self, task_kind: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns ((fail_start, fail_len), (succ_start, succ_len)) for the CSV.
        Row indices refer to data rows (excluding header).
        """
        if task_kind == "pick_coke_can":
            return (0, 80), (80, 80)
        else:
            return (0, 113), (113, 113)

    def _load_distributions(self, task_kind: str):
        csv_path = self._csv_path(task_kind)
        (fail_start, fail_len), (succ_start, succ_len) = self._segments(task_kind)

        # Preallocate
        succ_pos_mean = np.zeros((succ_len, 3), dtype=np.float64)
        succ_pos_std = np.ones((succ_len, 3), dtype=np.float64)
        succ_rot_mean = np.zeros((succ_len, 3), dtype=np.float64)
        succ_rot_std = np.ones((succ_len, 3), dtype=np.float64)
        fail_pos_mean = np.zeros((fail_len, 3), dtype=np.float64)
        fail_pos_std = np.ones((fail_len, 3), dtype=np.float64)
        fail_rot_mean = np.zeros((fail_len, 3), dtype=np.float64)
        fail_rot_std = np.ones((fail_len, 3), dtype=np.float64)

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)

        # Helper to fetch floats handling missing/blank values
        def fget(row: dict, key: str, default: float = 0.0) -> float:
            try:
                v = row.get(key, "")
                return float(v) if v not in (None, "", "NaN", "nan") else float(default)
            except Exception:
                return float(default)

        # Fill success segment
        for i in range(succ_len):
            row = data_rows[succ_start + i]
            succ_pos_mean[i, 0] = fget(row, "world_x_mean", 0.0)
            succ_pos_mean[i, 1] = fget(row, "world_y_mean", 0.0)
            succ_pos_mean[i, 2] = fget(row, "world_z_mean", 0.0)
            succ_pos_std[i, 0] = max(fget(row, "world_x_std", 1.0), self._eps)
            succ_pos_std[i, 1] = max(fget(row, "world_y_std", 1.0), self._eps)
            succ_pos_std[i, 2] = max(fget(row, "world_z_std", 1.0), self._eps)

            succ_rot_mean[i, 0] = fget(row, "rot_x_mean", 0.0)
            succ_rot_mean[i, 1] = fget(row, "rot_y_mean", 0.0)
            succ_rot_mean[i, 2] = fget(row, "rot_z_mean", 0.0)
            succ_rot_std[i, 0] = max(fget(row, "rot_x_std", 1.0), self._eps)
            succ_rot_std[i, 1] = max(fget(row, "rot_y_std", 1.0), self._eps)
            succ_rot_std[i, 2] = max(fget(row, "rot_z_std", 1.0), self._eps)

        # Fill failure segment
        for i in range(fail_len):
            row = data_rows[fail_start + i]
            fail_pos_mean[i, 0] = fget(row, "world_x_mean", 0.0)
            fail_pos_mean[i, 1] = fget(row, "world_y_mean", 0.0)
            fail_pos_mean[i, 2] = fget(row, "world_z_mean", 0.0)
            fail_pos_std[i, 0] = max(fget(row, "world_x_std", 1.0), self._eps)
            fail_pos_std[i, 1] = max(fget(row, "world_y_std", 1.0), self._eps)
            fail_pos_std[i, 2] = max(fget(row, "world_z_std", 1.0), self._eps)

            fail_rot_mean[i, 0] = fget(row, "rot_x_mean", 0.0)
            fail_rot_mean[i, 1] = fget(row, "rot_y_mean", 0.0)
            fail_rot_mean[i, 2] = fget(row, "rot_z_mean", 0.0)
            fail_rot_std[i, 0] = max(fget(row, "rot_x_std", 1.0), self._eps)
            fail_rot_std[i, 1] = max(fget(row, "rot_y_std", 1.0), self._eps)
            fail_rot_std[i, 2] = max(fget(row, "rot_z_std", 1.0), self._eps)

        return (
            succ_pos_mean,
            succ_pos_std,
            succ_rot_mean,
            succ_rot_std,
            fail_pos_mean,
            fail_pos_std,
            fail_rot_mean,
            fail_rot_std,
        )

    def _clamp_idx(self, idx: int, length: int) -> int:
        return 0 if length <= 0 else min(max(idx, 0), length - 1)

    def _compute_distances(
        self, timestep: int, action_world_vec: np.ndarray, action_rot_euler: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_idx = self._clamp_idx(timestep, self.succ_pos_mean.shape[0])
        f_idx = self._clamp_idx(timestep, self.fail_pos_mean.shape[0])

        Ds_pos = np.abs(action_world_vec - self.succ_pos_mean[s_idx]) / (self.succ_pos_std[s_idx] + self._eps)
        Ds_rot = np.abs(action_rot_euler - self.succ_rot_mean[s_idx]) / (self.succ_rot_std[s_idx] + self._eps)

        Df_pos = np.abs(action_world_vec - self.fail_pos_mean[f_idx]) / (self.fail_pos_std[f_idx] + self._eps)
        Df_rot = np.abs(action_rot_euler - self.fail_rot_mean[f_idx]) / (self.fail_rot_std[f_idx] + self._eps)

        return Ds_pos, Ds_rot, Df_pos, Df_rot

    def _write_frame(self, image_np: np.ndarray, timestep: int) -> Optional[str]:
        if not TRY_INCLUDE_IMAGE:
            return None
        try:
            import imageio

            path = os.path.join(self._frames_dir, f"timestep_{timestep}.png")
            imageio.imwrite(path, image_np)
            return path
        except Exception:
            return None

    def maybe_refine(
        self,
        raw_action: dict,
        timestep: int,
        current_instruction: str,
        image_np: Optional[np.ndarray] = None,
    ) -> Tuple[str, bool]:
        """
        Returns (instruction, did_refine)
        """
        try:
            action_world_vec = np.asarray(raw_action["world_vector"], dtype=np.float64)
            action_rot_euler = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        except Exception:
            # If fields are missing, cannot compute
            return current_instruction, False

        Ds_pos, Ds_rot, Df_pos, Df_rot = self._compute_distances(timestep, action_world_vec, action_rot_euler)

        # Evaluate rules
        rot_rule = bool(np.all(Df_rot < Ds_rot))
        pos_rule = bool(np.all(Df_pos < Ds_pos))

        if self.procedure == 1:
            rule_ok = rot_rule
        else:
            rule_ok = pos_rule or rot_rule

        if rule_ok:
            self._consec_counter += 1
        else:
            self._consec_counter = 0

        if self._consec_counter >= self._consec_needed:
            # Trigger refinement
            uncertain_dims: List[str] = []
            if self.procedure == 1:
                uncertain_dims = self.rot_dim_names
            else:
                uncertain_dims = self.rot_dim_names if rot_rule else (self.pos_dim_names if pos_rule else [])

            frame_path: Optional[str] = None
            if image_np is not None:
                frame_path = self._write_frame(image_np, timestep)
            frame_list = [frame_path] if frame_path is not None else []

            try:
                from simpler_env.evaluation.llm_instruction import generate_instruction

                new_instruction = generate_instruction(
                    initial_instruction=current_instruction,
                    uncertain_action_dims=", ".join(uncertain_dims) if uncertain_dims else "",
                    frame_images=frame_list,
                    current_timestep=timestep,
                    model_type=self.model_type,
                    task_name=self.task_name,
                    task_context="a robotic manipulation task",
                    api_key=None,
                    procedure=self.procedure,
                )
                # Reset counter so that the condition must be satisfied again before next refinement
                self._consec_counter = 0
                return new_instruction, True
            except Exception as e:
                print(f"LLM instruction generation error: {e}")
                self._consec_counter = 0
                return current_instruction, False

        return current_instruction, False


