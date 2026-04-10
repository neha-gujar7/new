# E-Commerce Catalog Manager OpenEnv

This repository contains a complete OpenEnv environment for a hackathon task: an e-commerce catalog cleanup environment with three deterministic product data management tasks.

## Environment Overview

The environment is implemented in `ecommerce_env.py` using FastAPI and Pydantic. It exposes the following endpoints:

- `POST /reset` — reset the environment for a specific task
- `POST /step` — submit an action and receive observation, reward, done, and info
- `GET /state` — inspect the current environment state

### Tasks

1. `categorize_product` (Easy)
   - Input: product title
   - Output: choose a category from a predefined ontology
   - Reward: 1.0 for correct category, 0.0 otherwise

2. `extract_attributes` (Medium)
   - Input: messy product description
   - Output: JSON attributes for `Color` and `Size`
   - Reward: 0.5 per correct attribute

3. `flag_and_fix` (Hard)
   - Input: list of three items
   - Output: identify the prohibited item and standardize the other two titles
   - Reward: partial score for correct flag and each correct title standardization

## Action / Observation Space

### Observation

The environment sends a typed `Observation` model containing:

- `task`: the active task name
- `payload`: task-specific prompt data

### Action

The agent should submit a typed `Action` model with some combination of:

- `category`: selected category for `categorize_product`
- `attributes`: JSON mapping for `extract_attributes`
- `flagged_item`: prohibited item for `flag_and_fix`
- `title_fixes`: mapping of original titles to standardized titles for `flag_and_fix`

### Reward

A `Reward` model is defined for typed reward handling, and `step()` returns a float reward in `[0.0, 1.0]`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the OpenEnv Server

```bash
uvicorn ecommerce_env:app --host 0.0.0.0 --port 8000
```

## Run the Baseline Inference Script

Set required environment variables:

```bash
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

Then run:

```bash
python inference.py
```

The baseline script prints exactly the required stdout lines:

- `[START] task=ecommerce_catalog env=ecommerce_catalog model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`
