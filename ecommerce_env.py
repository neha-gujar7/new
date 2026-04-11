from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    task: str
    payload: Dict[str, Any]


class Action(BaseModel):
    category: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    flagged_item: Optional[str] = None
    title_fixes: Optional[Dict[str, str]] = None


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task: str = "categorize_product"


class EnvironmentState(BaseModel):
    task: Optional[str] = None
    step: int = 0
    done: bool = True


class EcommerceCatalogManagerEnv:
    categories = [
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Beauty",
        "Toys",
    ]

    task_definitions = {
        "categorize_product": {
            "prompt": {
                "title": "UltraNoise Wireless Headphones",
                "choices": categories,
                "instructions": "Pick the single best category for this product title.",
            },
            "answer": "Electronics",
        },
        "extract_attributes": {
            "prompt": {
                "description": (
                    "Vintage denim jacket in mostly blue with small red accents, "
                    "size L fits a 42-inch chest."
                ),
                "attributes": ["Color", "Size"],
                "instructions": "Extract the Color and Size values into JSON.",
            },
            "answer": {"Color": "Blue", "Size": "L"},
        },
        "flag_and_fix": {
            "prompt": {
                "items": [
                    "Women's Handbag",
                    "2-Pack Batteries (AAA)",
                    "Toy Sword",
                ],
                "instructions": (
                    "Flag the prohibited item and standardize the remaining two titles "
                    "with consistent casing and punctuation."
                ),
            },
            "prohibited": "Toy Sword",
            "canonical_fixes": {
                "Women's Handbag": "Womens Handbag",
                "2-Pack Batteries (AAA)": "2 Pack Batteries AAA",
            },
        },
    }

    def __init__(self) -> None:
        self.current_task: Optional[str] = None
        self.current_observation: Optional[Observation] = None
        self.step_count: int = 0
        self.done: bool = True

    def reset(self, task: str) -> Observation:
        if task not in self.task_definitions:
            raise ValueError(f"Unknown task: {task}")
        self.current_task = task
        self.step_count = 0
        self.done = False
        self.current_observation = Observation(
            task=task, payload=self.task_definitions[task]["prompt"]
        )
        return self.current_observation

    def step(self, action: Action) -> StepResponse:
        if self.current_task is None:
            raise ValueError("Environment must be reset before calling step().")
        if self.done:
            raise ValueError("Episode is complete. Call reset() before taking another step.")

        self.step_count += 1
        task_name = self.current_task
        task_data = self.task_definitions[task_name]
        reward = 0.0
        info: Dict[str, Any] = {}

        if task_name == "categorize_product":
            expected = task_data["answer"]
            selected = action.category or ""
            correct = selected.strip().lower() == expected.lower()
            reward = 1.0 if correct else 0.0
            info["expected_category"] = expected
            info["selected_category"] = selected
            info["correct"] = correct

        elif task_name == "extract_attributes":
            expected = task_data["answer"]
            submitted = action.attributes or {}
            correct_count = 0
            attribute_info: Dict[str, Any] = {}
            for key, target in expected.items():
                actual = submitted.get(key, "").strip()
                is_correct = actual.lower() == target.lower()
                attribute_info[key] = {
                    "submitted": actual,
                    "expected": target,
                    "correct": is_correct,
                }
                if is_correct:
                    correct_count += 1
            reward = round(0.5 * correct_count, 2)
            info["attributes"] = attribute_info

        elif task_name == "flag_and_fix":
            prohibited = task_data["prohibited"]
            expected_fixes = task_data["canonical_fixes"]
            flagged = (action.flagged_item or "").strip()
            fixes = action.title_fixes or {}
            flag_correct = flagged == prohibited
            reward += 0.4 if flag_correct else 0.0
            info["flagged_item"] = flagged
            info["expected_flagged_item"] = prohibited
            info["flag_correct"] = flag_correct
            fix_info: Dict[str, Any] = {}
            for original, canonical in expected_fixes.items():
                submitted_fix = fixes.get(original, "").strip()
                is_correct = submitted_fix == canonical
                fix_info[original] = {
                    "submitted": submitted_fix,
                    "expected": canonical,
                    "correct": is_correct,
                }
                if is_correct:
                    reward += 0.3
            info["title_fixes"] = fix_info
            reward = min(round(reward, 2), 1.0)

        self.done = True
        response_observation = Observation(
            task=task_name,
            payload={
                "result": "completed",
                "task": task_name,
                "step": self.step_count,
            },
        )
        return StepResponse(
            observation=response_observation,
            reward=reward,
            done=self.done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task=self.current_task, step=self.step_count, done=self.done
        )


app = FastAPI(title="E-Commerce Catalog Manager OpenEnv")
env = EcommerceCatalogManagerEnv()


@app.post("/reset", response_model=Observation)
def reset_environment(request: Optional[ResetRequest] = None) -> Observation:
    try:
        task_name = request.task if request else "categorize_product"
        return env.reset(task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def step_environment(action: Action) -> StepResponse:
    try:
        return env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=EnvironmentState)
def get_state() -> EnvironmentState:
    return env.state()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": "categorize_product",
                "difficulty": "easy",
                "description": "Assign a product title to the correct category",
                "grader": {"type": "exact_match", "field": "category"},
                "score_range": [0.0, 1.0],
            },
            {
                "name": "extract_attributes",
                "difficulty": "medium",
                "description": "Extract Color and Size attributes from a product description",
                "grader": {"type": "partial_match", "field": "attributes"},
                "score_range": [0.0, 1.0],
            },
            {
                "name": "flag_and_fix",
                "difficulty": "hard",
                "description": "Flag prohibited item and standardize remaining titles",
                "grader": {"type": "partial_match", "field": "flagged_item"},
                "score_range": [0.0, 1.0],
            },
        ]
    }


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "E-Commerce Catalog Manager OpenEnv is running.",
        "tasks": ", ".join(list(EcommerceCatalogManagerEnv.task_definitions.keys())),
    }