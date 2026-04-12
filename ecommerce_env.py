from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import random

class Observation(BaseModel):
    task: str
    payload: Dict[str, Any]

class Action(BaseModel):
    category: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    flagged_item: Optional[str] = None
    title_fixes: Optional[Dict[str, str]] = None

class StepResponse(BaseModel):
    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task: str = "categorize_product"

app = FastAPI(title="Ecom Catalog OpenEnv")

class EcomEnv:
    def __init__(self):
        self.task_definitions = {
            "categorize_product": {
                "choices": ["Electronics", "Clothing", "Home", "Sports", "Beauty", "Toys"],
                "correct": "Electronics"
            },
            "extract_attributes": {
                "description": "Vintage denim jacket in mostly blue with small red accents, size L fits 42in chest.",
                "expected": {"Color": "Blue", "Size": "L"}
            },
            "flag_and_fix": {
                "items": ["Womens Handbag", "2-Pack Batteries AAA", "Toy Sword"],
                "prohibited": "Toy Sword",
                "canonical": {
                    "Womens Handbag": "Women's Handbag",
                    "2-Pack Batteries AAA": "2 Pack Batteries AAA"
                }
            }
        }
        self.current_task: Optional[str] = None
        self.step_count = 0
        self.max_steps = 5  # RL-friendly episode length

    def reset(self, task: str) -> Observation:
        if task not in self.task_definitions:
            raise ValueError(f"Unknown task: {task}")
        self.current_task = task
        self.step_count = 0
        payload = self.task_definitions[task].copy()
        # Add dynamic elements for realism/variability (key for RL exploration)
        if task == "categorize_product":
            payload["title"] = "UltraNoise Wireless Headphones Pro v2.0"
        elif task == "flag_and_fix":
            random.shuffle(payload["items"])  # Randomize for generalization
        return Observation(task=task, payload=payload)

    def step(self, action: Action) -> StepResponse:
        if self.current_task is None:
            raise ValueError("Reset first")
        if self.step_count >= self.max_steps:
            return StepResponse(
                observation=Observation(task=self.current_task, payload={"status": "max_steps"}),
                reward=0.1, done=True, info={"reason": "max_steps"}
            )
        self.step_count += 1

        task = self.task_definitions[self.current_task]
        reward = 0.1  # Base survival reward >0 for RL

        if self.current_task == "categorize_product":
            correct = task["correct"].lower()
            selected = (action.category or "").lower().strip()
            if selected == correct:
                reward = 0.9
            info = {"expected": task["correct"], "selected": action.category, "match": selected == correct}

        elif self.current_task == "extract_attributes":
            expected = task["expected"]
            matches = sum(1 for k, v in expected.items()
                         if action.attributes and action.attributes.get(k) == v)
            reward += 0.4 * (matches / len(expected))
            info = {"matches": matches, "total": len(expected), "submitted": action.attributes}

        elif self.current_task == "flag_and_fix":
            flag_correct = (action.flagged_item or "") == task["prohibited"]
            fix_matches = sum(1 for orig, canon in task["canonical"].items()
                             if action.title_fixes and action.title_fixes.get(orig) == canon)
            if flag_correct: reward += 0.4
            reward += 0.4 * (fix_matches / len(task["canonical"]))
            info = {
                "flag_correct": flag_correct,
                "fix_matches": fix_matches,
                "prohibited": task["prohibited"],
                "submitted": {"flagged": action.flagged_item, "fixes": action.title_fixes}
            }

        # Clamp for safety, but allow variance for RL signal
        reward = max(0.1, min(0.9, reward))
        done = self.step_count >= self.max_steps or reward >= 0.8  # Early termination on success

        obs_payload = {"step": self.step_count, "progress": task} if not done else {"status": "done"}
        return StepResponse(
            observation=Observation(task=self.current_task, payload=obs_payload),
            reward=reward,
            done=done,
            info={"grader_score": reward, **info}
        )

env = EcomEnv()

@app.get("/health")
def health():
    return {"status": "healthy"}  # Grader requirement



@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "categorize_product",
                "name": "categorize_product",
                "difficulty": "easy",
                "description": "Categorize product titles.",
                "grader": {"type": "exact_match", "field": "category"},
                "score_range": [0.0, 1.0]
            },
            {
                "id": "extract_attributes", 
                "name": "extract_attributes",
                "difficulty": "medium",
                "description": "Extract product attributes.",
                "grader": {"type": "partial_match", "field": "attributes"},
                "score_range": [0.0, 1.0]
            },
            {
                "id": "flag_and_fix",
                "name": "flag_and_fix",
                "difficulty": "hard", 
                "description": "Flag items and fix titles.",
                "grader": {"type": "partial_match", "field": "flagged_item"},
                "score_range": [0.0, 1.0]
            }
        ]
    }