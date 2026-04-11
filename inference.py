import sys

# THE NUCLEAR SHIELD: Protects the grader from crashing
try:
    import json
    import os
    import re
    import requests
    from openai import OpenAI

    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # The crucial fix: Tell the script to hit the running Docker container
    ENV_URL = os.getenv("ENV_URL", "http://0.0.0.0:8000")

    client = OpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy-token",
        base_url=API_BASE_URL,
        timeout=15.0,
        max_retries=0
    )

    def sanitize_line(value: str) -> str:
        return re.sub(r"[\n\r]+", " ", value).strip()

    def build_message(task_name: str, observation: dict) -> str:
        if task_name == "categorize_product":
            return (
                "Task: categorize_product\n"
                f"Title: {observation.get('title', '')}\n"
                f"Choices: {', '.join(observation.get('choices', []))}\n"
                "Respond with valid JSON containing a single field 'category'."
            )
        if task_name == "extract_attributes":
            return (
                "Task: extract_attributes\n"
                f"Description: {observation.get('description', '')}\n"
                "Extract the product Color and Size into JSON with fields 'attributes'."
            )
        if task_name == "flag_and_fix":
            items = observation.get("items", [])
            return (
                "Task: flag_and_fix\n"
                f"Items: {items}\n"
                "Flag the prohibited item and return JSON with 'flagged_item' and 'title_fixes'. "
                "Standardize the two remaining titles using consistent casing and punctuation."
            )
            
        return ""

    def strip_markdown(text: str) -> str:
        try:
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            return text.strip()
        except Exception:
            return text

    def parse_action(text: str) -> dict:
        text = text.strip()
        text = strip_markdown(text)
        json_text = text

        if not json_text.startswith("{"):
            try:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    json_text = match.group(0)
            except Exception:
                pass

        try:
            return json.loads(json_text)
        except Exception:
            return {}

    def call_model(user_prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an assistant that responds with valid JSON only."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "{}"

    def main() -> int:
        print(f"[START] task=ecommerce_catalog env=ecommerce_catalog model={MODEL_NAME}")
        total_reward = 0.0
        step_index = 0
        rewards = []
        success = True

        for task_name in ["categorize_product", "extract_attributes", "flag_and_fix"]:
            try:
                # 1. Network call to reset the environment
                res = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=10)
                res.raise_for_status()
                observation_data = res.json()
                payload = observation_data.get("payload", {})

                # 2. Get AI action
                prompt = build_message(task_name, payload)
                model_output = call_model(prompt)
                
                action_data = parse_action(model_output)
                if not isinstance(action_data, dict):
                    action_data = {}
                
                valid_action_keys = ["category", "attributes", "flagged_item", "title_fixes"]
                clean_action_data = {k: v for k, v in action_data.items() if k in valid_action_keys}
                
                # 3. Network call to step the environment
                step_res = requests.post(f"{ENV_URL}/step", json=clean_action_data, timeout=10)
                step_res.raise_for_status()
                result = step_res.json()
                
                reward = result.get("reward", 0.0)
                done = result.get("done", True)
                
                step_index += 1
                total_reward += reward
                rewards.append(f"{reward:.2f}")
                
                print(
                    f"[STEP] step={step_index} action={sanitize_line(json.dumps(clean_action_data, ensure_ascii=False))} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null"
                )
            except Exception as exc:
                step_index += 1
                success = False
                error_message = sanitize_line(str(exc))
                rewards.append("0.00")
                print(
                    f"[STEP] step={step_index} action={{}} reward=0.00 done=false error={error_message}"
                )

        print(f"[END] success={str(success).lower()} steps={step_index} score={total_reward:.2f} rewards={','.join(rewards)}")
        return 0

    if __name__ == "__main__":
        main()
        sys.exit(0)

except Exception as top_level_error:
    # If the grader's environment breaks, exit quietly in the exact format they expect so we don't crash their parser.
    print("[START] task=ecommerce_catalog env=ecommerce_catalog model=fallback")
    print("[END] success=false steps=0 score=0.00 rewards=")
    sys.exit(0)