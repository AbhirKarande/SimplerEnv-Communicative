import base64
import os
import requests
import time

prompt_template = """
You are a robotics expert providing refined instructions to an AI robot model ({model_type}).
The robot is performing a task called '{task_name}' which is about {task_context}.
The initial instruction was: "{initial_instruction}".
It is currently at timestep {current_timestep}.

The policy model is exhibiting low confidence, indicated by low entropy in the following action dimensions: {action_dim}.
This might mean the robot is "stuck" or "hesitant".

Here is the last frame from the robot's camera:

Your task is to provide a new, more specific, and actionable instruction to help the robot proceed with the task.
The instruction should be a short command, guiding the next immediate action.
Focus on what the robot should do right now based on the visual evidence.
Do not ask questions. Provide only the refined instruction.

Example of a good refined instruction: "push the red block slightly to the left."
Example of a bad refined instruction: "Can you see the red block? If so, push it."

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
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_endpoint="https://api.groq.com/openai/v1/chat/completions",
    api_key=None,
    delay=15,
):
    """
    Sends an API request with the given prompt.
    Returns None if the request fails.
    """
    if not api_key:
        # Get API key from env var if it isn't passed to this function
        api_key = os.environ.get("GROQ_API_KEY", "")

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
):
    """
    Generate new instruction using LLM
    """
    prompt = prompt_template.format(
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

    response = send_request(prompt, image_content, api_key=api_key)
    if response is None:
        return initial_instruction

    print(response)
    instruction = response.json()["choices"][0]["message"]["content"].strip()
    return instruction
