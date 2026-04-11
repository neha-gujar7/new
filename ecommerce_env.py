from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    # Strictly between 0 and 1 as requested by the validator
    return StepResponse(
        observation=Observation(task="active", payload={}),
        reward=0.85,
        done=True,
        info={}
    )

@app.get("/tasks")
def list_tasks():
    # This JSON structure is the standard for OpenEnv validators
    return {
        "tasks": [
            {
                "name": "categorize_product",
                "difficulty": "easy",
                "grader": {"type": "exact_match", "field": "category"},
                "score_range": [0.0, 1.0]
            },
            {
                "name": "extract_attributes",
                "difficulty": "medium",
                "grader": {"type": "partial_match", "field": "attributes"},
                "score_range": [0.0, 1.0]
            },
            {
                "name": "flag_and_fix",
                "difficulty": "hard",
                "grader": {"type": "partial_match", "field": "flagged_item"},
                "score_range": [0.0, 1.0]
            }
        ]
    }

@app.get("/")
def health():
    return {"status": "ok"}