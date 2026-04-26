# PR Review Environment

A reinforcement learning environment for training language models to perform automated pull request code review triage. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

**Live Environment:** [hitanshjain1812/meta_final on Hugging Face Spaces](https://huggingface.co/spaces/hitanshjain1812/meta_final)

**GitHub:** [mithilesh11705/Meta_3_1](https://github.com/mithilesh11705/Meta_3_1)

---

## Problem

Code review is one of the biggest bottlenecks in modern software engineering. Senior engineers spend 6-10 hours per week reviewing pull requests, yet studies show that 60% of review comments are about surface-level issues — wrong labels, missing priority flags, or boilerplate summaries that don't cite the actual code.

Current LLM-based review tools generate generic feedback without structured decision-making. They can't reliably:

- Classify the **type** of change (bug, security, enhancement, breaking-change)
- Assign an appropriate **priority** level
- Make a **triage decision** (approve, request changes, close)
- Write an **evidence-grounded summary** that cites the diff

This environment frames PR triage as a **verifiable RL problem** where every dimension of a review is scored against a deterministic gold standard, enabling GRPO training with dense, multi-component rewards.

---

## Architecture

![PR Review Environment Architecture](https://raw.githubusercontent.com/HitanshGithub/meta_hack/main/assets/architecture.png)

`PRReviewEnv` inherits from `openenv.core.env_server.interfaces.Environment` — the official OpenEnv base class — and implements the standard Gymnasium-style API:

```python
from openenv.core.env_server.interfaces import Environment

class PRReviewEnv(Environment):
    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation
    def step(self, action, timeout_s=None, **kwargs) -> Observation
    @property
    def state(self) -> State
```

### Key Components

| Component | Description |
|---|---|
| **PRReviewEnv** | Core environment inheriting from OpenEnv `Environment` ABC |
| **100 PR Scenarios** | 30 easy, 35 medium, 35 hard — covering typo fixes to distributed deadlocks |
| **Multi-Stage Review** | 3-stage pipeline: Identify Risk, Assess Impact, Final Triage |
| **Reward Engine** | 4-component composite score: Decision + Label F1 + Priority + Summary |
| **FastAPI Server** | HTTP endpoints (`/reset`, `/step`, `/state`, `/health`) on port 7860 |

---

## How the Environment Works

### Observation Space

Each episode presents the agent with a PR containing:

```
pr_id, title, description, diff, comments, files_changed,
author, base_branch, additions, deletions,
current_step, max_steps, review_stage, stage_prompt
```

### Action Space

The agent must return a structured JSON decision:

```json
{
  "decision": "approve | request_changes | close",
  "labels": ["bug", "security", "enhancement", ...],
  "priority": "low | medium | high | critical",
  "review_summary": "Evidence-grounded summary citing the diff..."
}
```

### Reward Function

The reward is a composite score in (0, 1) computed from four independently verifiable components:

| Component | Weight | How It's Scored |
|---|---|---|
| **Decision** | Exact match against gold standard | 1.0 if correct, scaled otherwise |
| **Label F1** | Set-level F1 score across 8 valid labels | Precision + Recall based |
| **Priority** | Distance on ordinal scale (low < medium < high < critical) | 1.0 - normalized distance |
| **Summary** | Keyword overlap with gold evidence terms | Coverage of key terms |

A step penalty of -0.02 per extra step encourages efficient triage.

### Task Difficulties

| Difficulty | Count | Examples | Max Steps |
|---|---|---|---|
| **Easy** | 30 | Typo fixes, dead code removal, import ordering | 4 |
| **Medium** | 35 | Auth refactors, SQL injection, missing validation | 6 |
| **Hard** | 35 | TOCTOU races, cache stampede, distributed deadlocks | 8 |

---

## Training Pipeline

The environment is paired with a GRPO (Group Relative Policy Optimization) training pipeline that fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` using LoRA:

```
Agent generates N completions per prompt
    -> Each completion is sent to the environment via HTTP POST /step
    -> Environment returns (observation, reward, done)
    -> GRPO computes group-relative advantages: A = (r - mean) / std
    -> Policy is updated to favor high-reward completions
```

### Training Results

Training was run on Kaggle (Tesla T4) with the following configuration:

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-0.5B-Instruct |
| Fine-tuning | LoRA (rank=16, alpha=32) |
| Optimizer | AdamW (lr=2e-5, cosine schedule) |
| Batch Size | 4 prompts x 4 completions |
| Training Steps | 60 |
| Framework | TRL GRPOTrainer |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Environment Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Interact with the Environment

```python
import requests

BASE = "http://localhost:7860"

# Reset to an easy task
obs = requests.post(f"{BASE}/reset", json={"task": "easy"}).json()
print(obs["title"], obs["review_stage"])

# Submit a review action
action = {
    "decision": "request_changes",
    "labels": ["bug"],
    "priority": "high",
    "review_summary": "The diff introduces a race condition in the auth handler."
}
result = requests.post(f"{BASE}/step", json=action).json()
print(f"Reward: {result['reward']:.3f}, Done: {result['done']}")
```

### 4. Run GRPO Training

```bash
python train_grpo.py --env-base-url http://localhost:7860
```

---

## Deployment

### Docker (Hugging Face Spaces)

```bash
docker build -t pr-review-env .
docker run -p 7860:7860 pr-review-env
```

The Dockerfile is pre-configured for Hugging Face Spaces deployment.

---

## Project Structure

```
Meta_3_1/
├── pr_review_env/          # Core environment package
│   ├── env.py              # PRReviewEnv(Environment) — OpenEnv base class
│   ├── models.py           # Action, Observation, Reward, StepResult schemas
│   ├── reward.py           # 4-component composite reward engine
│   └── tasks/              # 100 PR scenarios (easy/, medium/, hard/)
├── server/
│   └── app.py              # FastAPI HTTP server
├── train_grpo.py           # GRPO training script (TRL + LoRA)
├── inference.py            # Single-PR inference script
├── openenv.yaml            # OpenEnv environment manifest
├── Dockerfile              # HF Spaces deployment
├── demo/
│   └── PR_Review_GRPO_Kaggle.ipynb  # Kaggle training notebook
└── requirements.txt
```

---

## OpenEnv Integration

This environment is built on top of **OpenEnv** (`openenv-core >= 0.2.3`):

- `PRReviewEnv` inherits from `openenv.core.env_server.interfaces.Environment`
- Implements the abstract `reset()`, `step()`, and `state` contract
- Uses OpenEnv canonical types: `Action`, `Observation`, `State`
- Ships with `openenv.yaml` manifest for environment discovery
- Deployed as a Docker-based Hugging Face Space

---

## References

- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- [TRL Library (GRPO)](https://huggingface.co/docs/trl)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

---

## License

This project is released under the MIT License.
