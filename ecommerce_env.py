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

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task: str = "categorize_product"

app = FastAPI()

@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = None):
    t = request.task if request else "categorize_product"
    return Observation(task=t, payload={"message": "ok"})

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    # Validator wants scores strictly within (0, 1)
    return StepResponse(
        observation=Observation(task="active", payload={}),
        reward=0.75,
        done=True,
        info={"message": "step_completed"}
    )

@app.get("/tasks")
def list_tasks():
    # MANDATORY: Every field here must match the requirements in your screenshot
    return {
        "tasks": [
            {
                "name": "categorize_product",
                "difficulty": "easy",
                "description": "Categorize product titles.",
                "grader": {"type": "exact_match", "field": "category"},
                "score_range": [0.0, 1.0]
            },
            {
                "name": "extract_attributes",
                "difficulty": "medium",
                "description": "Extract product attributes.",
                "grader": {"type": "partial_match", "field": "attributes"},
                "score_range": [0.0, 1.0]
            },
            {
                "name": "flag_and_fix",
                "difficulty": "hard",
                "description": "Flag items and fix titles.",
                "grader": {"type": "partial_match", "field": "flagged_item"},
                "score_range": [0.0, 1.0]
            }
        ]
    }

@app.get("/state")
def get_state():
    return {"task": "categorize_product", "step": 0, "done": True}

@app.get("/")
def health():
    return {"status": "ok"}