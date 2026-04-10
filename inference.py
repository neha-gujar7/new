import json
import os
import re

import openai

try:
    from openai import OpenAIError
except ImportError:  # pragma: no cover
    OpenAIError = Exception

from ecommerce_env import EcommerceCatalogManagerEnv, Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

openai.api_base = API_BASE_URL
openai.api_key = HF_TOKEN


def sanitize_line(value: str) -> str:
    return re.sub(r"[\n\r]+", " ", value).strip()


def build_message(task_name: str, observation: dict[str, any]) -> str:
    if task_name == "categorize_product":
        return (
            "Task: categorize_product\n"
            f"Title: {observation['title']}\n"
            f"Choices: {', '.join(observation['choices'])}\n"
            "Respond with valid JSON containing a single field 'category'."
        )

    if task_name == "extract_attributes":
        return (
            "Task: extract_attributes\n"
            f"Description: {observation['description']}\n"
            "Extract the product Color and Size into JSON with fields 'attributes'."
        )

    if task_name == "flag_and_fix":
        items = observation["items"]
        return (
            "Task: flag_and_fix\n"
            f"Items: {items}\n"
            "Flag the prohibited item and return JSON with 'flagged_item' and 'title_fixes'. "
            "Standardize the two remaining titles using consistent casing and punctuation."
        )

    return ""


def parse_action(text: str) -> dict[str, any]:
    text = text.strip()
    json_text = text

    if not json_text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON output: {exc} | response: {text}")


def call_model(system_prompt: str, user_prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an assistant that responds with valid JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message["content"].strip()


def main() -> int:
    env = EcommerceCatalogManagerEnv()

    print(f"[START] task=ecommerce_catalog env=ecommerce_catalog model={MODEL_NAME}")
    total_reward = 0.0
    step_index = 0
    rewards: list[str] = []
    success = True

    for task_name in ["categorize_product", "extract_attributes", "flag_and_fix"]:
        try:
            observation = env.reset(task_name)
            prompt = build_message(task_name, observation.payload)
            model_output = call_model("", prompt)
            action_data = parse_action(model_output)
            action = Action(**action_data)
            result = env.step(action)
            step_index += 1
            total_reward += result.reward
            rewards.append(f"{result.reward:.2f}")
            print(
                f"[STEP] step={step_index} action={sanitize_line(json.dumps(action_data, ensure_ascii=False))} "
                f"reward={result.reward:.2f} done={str(result.done).lower()} error=null"
            )
        except (OpenAIError, ValueError, TypeError) as exc:
            step_index += 1
            success = False
            error_message = sanitize_line(str(exc))
            rewards.append("0.00")
            print(
                f"[STEP] step={step_index} action=null reward=0.00 done=false error={error_message}"
            )

    print(
        f"[END] success={str(success).lower()} steps={step_index} score={total_reward:.2f} rewards={','.join(rewards)}"
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
