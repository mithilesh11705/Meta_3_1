---
title: PR Review Environment
emoji: 🏆
colorFrom: green
colorTo: indigo
sdk: docker
app_file: server/app.py
pinned: false
---

# pr-review-env

Deterministic OpenEnv environment for pull request triage and review quality assessment with **100 real-world PR scenarios** across three difficulty levels.

## Theme Fit
Primary fit: **Theme #3.1 - World Modeling (Professional Tasks)**.

Why:
- Professional workflow simulation (realistic PR review and risk triage).
- Verifiable outcomes via environment-based reward.
- Multi-step interaction through staged review (`identify_risk`, `assess_impact`, `final_triage`).

## What This Environment Trains
- Decision quality: `approve` vs `request_changes` vs `close`
- Risk labeling: security, bug, urgency, and test coverage cues
- Priority judgment: low/medium/high/critical
- Evidence-grounded communication in review summaries

## Observation Space
`pr_id`, `title`, `description`, `diff`, `comments`, `files_changed`, `author`, `base_branch`, `additions`, `deletions`, `current_step`, `max_steps`, `task_name`, `review_stage`, `stage_prompt`

## Action Space
- `decision`: `approve | request_changes | close`
- `labels`: list from `bug, security, enhancement, documentation, breaking-change, needs-tests, trivial, urgent`
- `priority`: `low | medium | high | critical`
- `review_summary`: free-form text, constrained by validator

## Reward Design
Stage-aware weighted score:
- Decision correctness
- Label F1
- Priority distance
- Summary quality + evidence cues
- Consistency penalties (anti-gaming)
- Step penalty (`0.02 * (current_step - 1)`)

Total is clamped to strict `(0, 1)`.

### Latency-Accuracy Tradeoff Scoring
To better reflect CI/CD deployability, we now compute a latency-aware benchmark score:

- `latency_adjusted_score = raw_reward * latency_discount`
- `raw_reward` is the existing deterministic environment score
- `latency_discount` is:
  - `1.0` when latency is at or below task budget
  - exponential decay above budget: `exp(-0.35 * max(0, latency_seconds - budget_seconds))`
  - floored at `0.1` so slow-but-correct agents are penalized but not zeroed

This keeps reward semantics deterministic while adding runtime-aware benchmarking in inference/evaluation pipelines.
Use both `raw_reward` and `latency_adjusted_score` for analysis to plot an accuracy-speed Pareto frontier.

## Tasks

**100 tasks** across 3 difficulty levels:

| Difficulty | Count | Max Steps | Latency Budget (s) | Example Scenarios |
|---|---:|---:|---:|---|
| Easy | 30 | 4 | 5.0 | Typo fixes, import ordering, dead code removal, dependency pinning |
| Medium | 35 | 6 | 8.0 | SQL injection, hardcoded credentials, cache invalidation, breaking API changes |
| Hard | 35 | 8 | 10.0 | TOCTOU races, distributed locks, saga patterns, connection pool exhaustion |

Backward-compatible task IDs: `easy`, `medium`, `hard` (first fixture per difficulty).
All 100 tasks are accessible via `GET /tasks` or by ID (e.g. `easy_1012`, `medium_2081`).

## Quick Start
```bash
docker build -t pr-review-env .
docker run --rm -p 7860:7860 pr-review-env
curl http://localhost:7860/health
```

## Baseline Inference
```bash
set HF_TOKEN=hf_xxx
python inference.py
```

Optional environment variables:
- `ENV_BASE_URL` (default: `http://127.0.0.1:7860`)
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)

## RL Training (TRL + Unsloth, Colab-First)

This repo now includes a minimal GRPO training pipeline that talks directly to the environment verifier and writes judge-friendly artifacts.

### Colab Notebook (Recommended)
- [`colab/PR_Review_GRPO_Training.ipynb`](colab/PR_Review_GRPO_Training.ipynb)

### Local Script
```bash
pip install -r requirements-train.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

In a second terminal:
```bash
python train_grpo.py ^
  --env-base-url http://127.0.0.1:7860 ^
  --model-name Qwen/Qwen2.5-0.5B-Instruct ^
  --num-samples 24 ^
  --num-train-epochs 1 ^
  --num-generations 2 ^
  --max-completion-length 220 ^
  --max-new-tokens 220 ^
  --output-dir artifacts/grpo_run
```

Training outputs:
- `artifacts/grpo_run/logs/reward_history.csv`
- `artifacts/grpo_run/logs/reward_components.csv` (if available)
- `artifacts/grpo_run/logs/evaluation_baseline.csv` (raw + latency metrics)
- `artifacts/grpo_run/logs/evaluation_after_training.csv` (raw + latency metrics)
- `artifacts/grpo_run/logs/training_summary.json`
- `artifacts/grpo_run/logs/before_after.md`
- `artifacts/grpo_run/plots/reward_curve.png`
- `artifacts/grpo_run/checkpoints/final/`

## Hugging Face Space Deployment
```bash
git remote add space https://huggingface.co/spaces/<username>/pr-review-env
git push space main
```

The Docker image launches with:
`uvicorn server.app:app --host 0.0.0.0 --port 7860`

## Judge Reproduction Flow
```bash
git clone <repo-url>
cd pr-review-env
docker build -t pr-review-env .
docker run --rm -p 7860:7860 pr-review-env
```

In a second terminal:
```bash
python inference.py
python train_grpo.py --env-base-url http://127.0.0.1:7860 --model-name Qwen/Qwen2.5-0.5B-Instruct --num-generations 2 --num-train-epochs 1 --output-dir artifacts/grpo_judge_run
```

Then inspect:
- `artifacts/grpo_judge_run/plots/reward_curve.png`
- `artifacts/grpo_judge_run/logs/before_after.md`

## Required Submission Links
Add these in your final Hackathon submission:
- Hugging Face Space URL: `TODO`
- Mini-blog URL (HF article) or `<2 min` YouTube URL: `TODO`
- Optional slide deck URL: `TODO`

## Additional Docs
- [JUDGES_GUIDE.md](JUDGES_GUIDE.md)
- [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md)
- [SUBMISSION_READY.md](SUBMISSION_READY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [SCORING_ANALYSIS.md](SCORING_ANALYSIS.md)
