import base64
import os
import requests
import time

prompt_template_proc1 = """
You are a robotics expert providing refined instructions to an AI robot model ({model_type}).
The robot is performing a task called '{task_name}' which is about {task_context}.
The initial instruction was: "{initial_instruction}".
It is currently at timestep {current_timestep}.

We are using Procedure 1: orientation-focused refinement. The model's actions suggest uncertainty specifically in orientation (rotation) dimensions: {action_dim}.
Provide a refined, orientation-specific command that helps adjust the gripper's orientation to make progress. Keep it short and immediately actionable.

Here is the last frame from the robot's camera:

If available, a goal/overlay image for visual matching is also included below.

Constraints:
- One concise imperative sentence.
- The sentence MUST explicitly state the high-level task action appropriate for '{task_name}':
  - If the task is a move-near task, begin with "Move ... near ..." specifying the two relevant objects.
  - If the task is to pick the coke can, begin with "Pick up the coke can" (include approach/orientation detail as needed).
  - If the task is to close a drawer, begin with "Close the drawer" (include approach/orientation detail as needed).
- Use object names from the initial instruction when available.
- Focus on orientation cues (roll/pitch/yaw) relative to visible objects.
- Do not ask questions. Output only the instruction.

Refined Instruction:
"""

prompt_template_proc2 = """
You are a robotics expert providing refined instructions to an AI robot model ({model_type}).
The robot is performing a task called '{task_name}' which is about {task_context}.
The initial instruction was: "{initial_instruction}".
It is currently at timestep {current_timestep}.

We are using Procedure 2: position-orientation refinement. The model's actions suggest uncertainty in these dimensions: {action_dim}.
Provide a refined command targeting position (x,y,z), orientation (roll,pitch,yaw), or both, choosing whichever is most immediately helpful. Keep it short and directly actionable.

Here is the last frame from the robot's camera:

If available, a goal/overlay image for visual matching is also included below.

Constraints:
- One concise imperative sentence.
- The sentence MUST explicitly state the high-level task action appropriate for '{task_name}':
  - If the task is a move-near task, begin with "Move ... near ..." specifying the two relevant objects.
  - If the task is to pick the coke can, begin with "Pick up the coke can" (and any necessary approach/position/orientation detail).
  - If the task is to close a drawer, begin with "Close the drawer" (and any necessary approach/position/orientation detail).
- Use object names from the initial instruction when available.
- Do not ask questions. Output only the instruction.

Refined Instruction:
"""


def encode_image(image_path):
    """
    Encodes an image as base64
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_images(frame_images):
    """
    Returns last image in frame_images encoded as base64
    """
    latest_image_path = frame_images[-1]
    base64_image = encode_image(latest_image_path)
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]


def send_request(
    prompt_text,
    image_content=[],
    model="gemini-2.0-flash",
    api_endpoint="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    api_key=None,
    delay=1,
):
    """
    Sends an API request with the given prompt.
    Returns None if the request fails.
    """
    if not api_key:
        # Get API key from env var if it isn't passed to this function
        api_key = os.environ.get("API_KEY", "")

    if api_key == "":
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_content}],
        "max_tokens": 50,
    }

    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None
    finally:
        # Sleep to stay under API rate limit
        time.sleep(delay)

    return response


def generate_instruction(
    initial_instruction,
    uncertain_action_dims,
    frame_images,
    current_timestep,
    model_type="octo-base",
    robot_type="google_robot",
    task_name="google_robot_move_near",
    task_context="a robotic manipulation task",
    api_key=None,
    procedure: int = 1,
    goal_image_path: str | None = None,
):
    """
    Generate new instruction using LLM
    """
    tmpl = prompt_template_proc1 if int(procedure) == 1 else prompt_template_proc2
    prompt = tmpl.format(
        model_type=model_type,
        task_name=task_name,
        task_context=task_context,
        initial_instruction=initial_instruction,
        current_timestep=current_timestep,
        action_dim=uncertain_action_dims,
    )

    image_content = []
    if frame_images:
        image_content = process_images(frame_images)
    # Optionally include a goal/overlay image (e.g., from rgb_overlay_path)
    if goal_image_path is not None:
        try:
            base64_goal = encode_image(goal_image_path)
            image_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal}"}})
        except Exception:
            pass

    response = send_request(prompt, image_content, api_key=api_key)
    if response is None:
        return initial_instruction

    print(response)
    instruction = response.json()["choices"][0]["message"]["content"].strip()
    return instruction
