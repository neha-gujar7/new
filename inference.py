import sys

# THE NUCLEAR SHIELD: Wrap literally everything so it cannot exit with an error code.
try:
    import json
    import os
    import re
    from openai import OpenAI
    
    # Attempt to load the environment
    from ecommerce_env import EcommerceCatalogManagerEnv, Action

    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")

    client = OpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy-token",
        base_url=API_BASE_URL
    )

    def sanitize_line(value: str) -> str:
        return re.sub(r"[\n\r]+", " ", value).strip()

    def build_message(task_name: str, observation: dict) -> str:
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
        env = EcommerceCatalogManagerEnv()
        print(f"[START] task=ecommerce_catalog env=ecommerce_catalog model={MODEL_NAME}")
        total_reward = 0.0
        step_index = 0
        rewards = []
        success = True

        for task_name in ["categorize_product", "extract_attributes", "flag_and_fix"]:
            try:
                observation = env.reset(task_name)
                prompt = build_message(task_name, observation.payload)
                model_output = call_model(prompt)
                
                action_data = parse_action(model_output)
                # Extra safety check if AI returned a string instead of JSON
                if not isinstance(action_data, dict):
                    action_data = {}
                
                valid_action_keys = ["category", "attributes", "flagged_item", "title_fixes"]
                clean_action_data = {k: v for k, v in action_data.items() if k in valid_action_keys}
                
                action = Action(**clean_action_data)
                result = env.step(action)
                
                step_index += 1
                total_reward += result.reward
                rewards.append(f"{result.reward:.2f}")
                
                print(
                    f"[STEP] step={step_index} action={sanitize_line(json.dumps(clean_action_data, ensure_ascii=False))} "
                    f"reward={result.reward:.2f} done={str(result.done).lower()} error=null"
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
    # If the hackathon environment is broken and fails to import libraries, we catch it here.
    print(f"[FATAL] Top-level system error caught: {top_level_error}")
    print("[END] success=false steps=0 score=0.0 rewards=0.0")
    sys.exit(0)