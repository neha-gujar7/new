import sys
import threading
import time

try:
    import json
    import os
    import re
    import requests
    import uvicorn
    from openai import OpenAI

    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")
    ENV_PORT = 8000
    ENV_URL = f"http://127.0.0.1:{ENV_PORT}"

    client = OpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy-token",
        base_url=API_BASE_URL,
        timeout=20.0,
        max_retries=0
    )

    def start_server():
        try:
            from ecommerce_env import app
            uvicorn.run(app, host="127.0.0.1", port=ENV_PORT, log_level="error")
        except Exception as e:
            print(f"[DEBUG] Server thread error: {e}", flush=True)

    def sanitize_line(value: str) -> str:
        return re.sub(r"[\n\r]+", " ", str(value)).strip()

    def build_message(task_name: str, observation: dict) -> str:
        if task_name == "categorize_product":
            return (
                f"Task: categorize_product\n"
                f"Title: {observation.get('title', '')}\n"
                f"Choices: {', '.join(observation.get('choices', []))}\n"
                f"Respond with valid JSON: {{\"category\": \"<one of the choices>\"}}"
            )
        if task_name == "extract_attributes":
            return (
                f"Task: extract_attributes\n"
                f"Description: {observation.get('description', '')}\n"
                f"Respond with valid JSON: {{\"attributes\": {{\"Color\": \"<value>\", \"Size\": \"<value>\"}}}}"
            )
        if task_name == "flag_and_fix":
            items = observation.get("items", [])
            return (
                f"Task: flag_and_fix\n"
                f"Items: {items}\n"
                f"Respond with valid JSON: {{\"flagged_item\": \"<prohibited item>\", "
                f"\"title_fixes\": {{\"<original>\": \"<fixed>\", \"<original2>\": \"<fixed2>\"}}}}"
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
        try:
            text = strip_markdown(text.strip())
            if not text.startswith("{"):
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    text = match.group(0)
            return json.loads(text)
        except Exception:
            return {}

    def call_model(user_prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Respond with valid JSON only. No explanation, no markdown."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "{}"

    def main():
        # Start the env server in background
        t = threading.Thread(target=start_server, daemon=True)
        t.start()

        # Wait for server to be ready (max 15 seconds)
        ready = False
        for _ in range(15):
            try:
                r = requests.get(f"{ENV_URL}/", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            print(f"[START] task=ecommerce_catalog env=ecommerce_catalog model={MODEL_NAME}", flush=True)
            print("[STEP] step=1 action={} reward=0.00 done=true error=server_not_ready", flush=True)
            print("[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
            sys.exit(0)

        print(f"[START] task=ecommerce_catalog env=ecommerce_catalog model={MODEL_NAME}", flush=True)
        total_reward = 0.0
        step_index = 0
        rewards = []
        success = True

        for task_name in ["categorize_product", "extract_attributes", "flag_and_fix"]:
            try:
                res = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=10)
                res.raise_for_status()
                payload = res.json().get("payload", {})

                prompt = build_message(task_name, payload)
                model_output = call_model(prompt)
                action_data = parse_action(model_output)
                if not isinstance(action_data, dict):
                    action_data = {}

                valid_keys = ["category", "attributes", "flagged_item", "title_fixes"]
                clean_data = {k: v for k, v in action_data.items() if k in valid_keys}

                step_res = requests.post(f"{ENV_URL}/step", json=clean_data, timeout=10)
                step_res.raise_for_status()
                result = step_res.json()

                reward = float(result.get("reward", 0.0))
                done = result.get("done", True)
                step_index += 1
                total_reward += reward
                rewards.append(f"{reward:.2f}")

                print(
                    f"[STEP] step={step_index} "
                    f"action={sanitize_line(json.dumps(clean_data, ensure_ascii=False))} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True
                )
            except Exception as exc:
                step_index += 1
                success = False
                rewards.append("0.00")
                print(
                    f"[STEP] step={step_index} action={{}} reward=0.00 done=true "
                    f"error={sanitize_line(str(exc))}",
                    flush=True
                )

        score = total_reward / 3.0
        print(
            f"[END] success={str(success).lower()} steps={step_index} "
            f"score={score:.2f} rewards={','.join(rewards)}",
            flush=True
        )
        sys.exit(0)

    if __name__ == "__main__":
        main()

except Exception as e:
    print(f"[START] task=ecommerce_catalog env=ecommerce_catalog model=fallback", flush=True)
    print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
    sys.exit(0)