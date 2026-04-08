---
title: PR Review Environment
emoji: 🏆
colorFrom: green
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# pr-review-env

## Why This Environment Exists
Modern engineering teams triage pull requests continuously, and quality depends on correctly interpreting diffs, reviewer signals, and risk, not just PR descriptions. `pr-review-env` is a deterministic OpenEnv benchmark that simulates real PR review workflows across bugfix, security-sensitive, and contested infrastructure changes so agents can be measured on decision quality, prioritization, and review communication under realistic constraints.

Key properties:
- Realistic pull request diffs and reviewer comments across bugfix, security, and concurrency scenarios
- Deterministic scoring with no LLM calls in the grader
- Stage-aware review workflow instead of a single one-shot answer
- Containerized FastAPI deployment with OpenEnv-compatible metadata

## Observation Space

| Field | Type | Description |
|---|---|---|
| `pr_id` | `int` | Unique pull request identifier |
| `title` | `str` | PR title |
| `description` | `str` | Author-provided PR description |
| `diff` | `str` | Unified diff text (realistic code changes) |
| `comments` | `list[str]` | Inline reviewer comments |
| `files_changed` | `list[str]` | Changed files in the PR |
| `author` | `str` | PR author username |
| `base_branch` | `str` | Target base branch |
| `additions` | `int` | Number of added lines |
| `deletions` | `int` | Number of deleted lines |
| `current_step` | `int` | Current step in episode |
| `max_steps` | `int` | Maximum allowed steps |
| `task_name` | `str` | Task id (`easy`, `medium`, `hard`) |
| `review_stage` | `str` | Current workflow stage (`identify_risk`, `assess_impact`, `final_triage`) |
| `stage_prompt` | `str` | Stage-specific instruction for what the agent should focus on this step |

## Action Space

| Field | Type | Description |
|---|---|---|
| `decision` | `Literal["approve", "request_changes", "close"]` | Final triage decision |
| `labels` | `list[str]` | Multi-label tags from allowed set |
| `priority` | `Literal["low", "medium", "high", "critical"]` | Incident-style urgency |
| `review_summary` | `str` | 1-3 sentence review summary |

Allowed labels: `bug`, `security`, `enhancement`, `documentation`, `breaking-change`, `needs-tests`, `trivial`, `urgent`.

## Reward Function
`reward = stage_weighted_mean(decision_score, label_score, priority_score, summary_score) - contradiction_penalty - step_penalty`, clamped to `(0, 1)`.

1. `decision_score` (0.25 weight): `1.0` if decision matches gold, else `0.0`.
2. `label_score` (0.25 weight): F1 score between predicted and gold labels.
3. `priority_score` (0.25 weight): exact=`1.0`, off-by-1=`0.5`, off-by-2=`0.25`, else=`0.0`.
4. `summary_score` (0.25 weight): keyword hit ratio against `gold_keywords`, with hard penalty to `0.0` when summary length `<20` or `>500` chars.

Additional shaping:
- Stage-aware weighting: early steps emphasize evidence and impact discovery, final steps emphasize coherent triage.
- Contradiction penalties: inconsistent outputs like `approve` + `security`/`urgent` are penalized.
- Step penalty: subtract `0.02 * (current_step - 1)` to reward fast, correct triage.

## Review Workflow

Each episode now follows a realistic review progression instead of repeating the same final answer:

1. `identify_risk`: identify the core bug or vulnerability with code evidence
2. `assess_impact`: explain user/system impact and align labels + priority
3. `final_triage`: provide final decision with concise remediation guidance

## Tasks

| Name | Difficulty | Scenario | What Makes It Hard |
|---|---|---|---|
| `easy` | easy | Off-by-one bugfix in Python slicing helper with clear reviewer consensus | Mostly a correctness check with low risk |
| `medium` | medium | Auth middleware refactor silently removes token expiry enforcement | Security issue hidden behind "cleanup" framing |
| `hard` | hard | Redis rate limiter introduces TOCTOU race under concurrency with reviewer disagreement | Misleading description and contested comments require deep diff reasoning |

## Quick Start

### Build and run with Docker
```bash
docker build -t pr-review-env .
docker run --rm -p 7860:7860 pr-review-env
```

### Health check
```bash
curl http://localhost:7860/health
```

Expected response:
```json
{
  "status": "ok",
  "environment": "pr-review-env",
  "version": "1.0.0",
  "active_sessions": 0,
  "available_tasks": ["easy", "medium", "hard"],
  "timestamp": 1672531200.0
}
```

### Explore the API
```bash
# List available tasks
curl http://localhost:7860/tasks

# Get example actions
curl http://localhost:7860/examples

# Check system metrics
curl http://localhost:7860/metrics

# Get OpenAPI docs
open http://localhost:7860/docs
```

## Running Inference
Set your Hugging Face token, then run:

```bash
export HF_TOKEN=hf_xxx
python inference.py
```

Optional env vars:
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `ENV_BASE_URL` (default: `http://127.0.0.1:7860`)

### Expected Output Format
```
[START] task=easy env=pr-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"decision":"approve","labels":["bug"],"priority":"low","review_summary":"LGTM - fixes the off-by-one error correctly."} reward=0.95 done=true error=null
[END] success=true steps=1 score=0.95 rewards=0.95
[START] task=medium env=pr-review-env model=Qwen/Qwen2.5-72B-Instruct
...
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_reward.py -v

# Run with coverage
pytest tests/ --cov=pr_review_env --cov-report=html
```

## Baseline Scores

| Task | Model | Score | Steps | Notes |
|---|---|---:|---:|---|
| `easy` | `Qwen/Qwen2.5-72B-Instruct` | `0.95` | `1` | Perfect decision and labels |
| `medium` | `Qwen/Qwen2.5-72B-Instruct` | `0.72` | `2` | Catches security issue |
| `hard` | `Qwen/Qwen2.5-72B-Instruct` | `0.48` | `3` | Identifies race condition |

**Expected score ranges for competitive models:**
- **Easy**: 0.80-1.00 (should be near-perfect)
- **Medium**: 0.50-0.75 (requires security analysis)
- **Hard**: 0.20-0.55 (challenging concurrency reasoning)

## Troubleshooting

### Common Issues

**1. Docker build fails**
```bash
# Clear Docker cache and rebuild
docker system prune -f
docker build --no-cache -t pr-review-env .
```

**2. Port already in use**
```bash
# Find and kill process on port 7860
lsof -ti:7860 | xargs kill -9
# Or use different port
docker run -p 8080:7860 pr-review-env
```

**3. Inference fails with connection error**
```bash
# Check if server is running
curl http://localhost:7860/health
# If not running, start the server first
# Then check ENV_BASE_URL in inference
export ENV_BASE_URL=http://127.0.0.1:7860
```

**4. LLM response parsing errors**
- Check the debug output in inference logs
- Verify model supports JSON output format
- Try reducing `max_tokens` in inference.py

**5. Low scores on easy task**
- Ensure labels exactly match gold labels
- Check review summary contains keywords
- Verify priority matches gold priority

**6. Session state issues**
- Use `/reset` to start fresh session
- Check session_id is passed in headers
- Use `/metrics` to debug session state

### Debugging Tools

**Reward Breakdown Analysis**
```bash
# Use validation endpoint to debug scoring
curl -X POST http://localhost:7860/validate \
  -H "Content-Type: application/json" \
  -d '{"task":"easy","action":{"decision":"approve","labels":["bug"],"priority":"low","review_summary":"test"}}' | jq
```

**Environment State Inspection**
```bash
# Get current state
curl http://localhost:7860/state -H "session_id:your-id" | jq

# View all active sessions
curl http://localhost:7860/metrics | jq '.sessions'
```

**Test Individual Components**
```bash
# Test reward function directly
python -c "
from pr_review_env.reward import compute_reward_breakdown
from pr_review_env.models import Action, Observation
# ... test code here
"
```

### Performance Optimization

**Reduce LLM latency:**
- Use smaller models for debugging
- Cache responses during development
- Set lower temperature for faster convergence

**Improve reward scores:**
- Study gold keywords in each task
- Use the `/examples` endpoint for reference
- Analyze reward breakdown to identify weak areas

**Handle concurrency:**
- Each session is independent
- Use different session IDs for parallel evaluation
- Monitor `/metrics` for session leaks

## Hugging Face Space Deployment
1. Create a Space (SDK: Docker) and set secret `HF_TOKEN` in Space settings.
2. Push this repo contents to the Space remote:
```bash
git remote add space https://huggingface.co/spaces/<username>/pr-review-env
git push space main
```
3. Space starts the app with `uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1` from `Dockerfile`.
4. Validate deployment by opening `/health`, `/tasks`, and then running local `inference.py` against the Space URL via `ENV_BASE_URL`.

## Advanced Usage

### Session Management
The environment supports concurrent sessions via UUID-based session IDs:
```bash
# Start a session
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"medium"}' \
  -i  # Capture session_id header

# Use the session for subsequent steps
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -H "session_id: your-uuid-here" \
  -d '{"decision":"approve","labels":["bug"],"priority":"low","review_summary":"LGTM"}'
```

### Validation Endpoint
Test actions without affecting session state:
```bash
curl -X POST http://localhost:7860/validate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "easy",
    "action": {
      "decision": "approve",
      "labels": ["bug"],
      "priority": "low",
      "review_summary": "LGTM - fixes the issue correctly."
    }
  }'
```

### Custom Evaluation
```python
from pr_review_env.env import PRReviewEnv
from pr_review_env.models import Action

# Create environment
env = PRReviewEnv()
obs = env.reset("hard")

# Test custom action
action = Action(
    decision="request_changes",
    labels=["bug", "needs-tests"],
    priority="high",
    review_summary="Has race condition - use atomic operations."
)

result = env.step(action)
print(f"Reward: {result.reward:.3f}, Done: {result.done}")
print(f"Breakdown: {result.info['reward_breakdown']}")
```

## Environment Architecture

### Component Overview
`
pr-review-env/
  app.py              # FastAPI server with session management
  inference.py        # Baseline agent using the OpenAI client
  pr_review_env/
    env.py            # Core environment logic and staged workflow
    models.py         # Pydantic models with validation
    reward.py         # Deterministic stage-aware reward function
    tasks/            # Task definitions and fixtures
  tests/              # Targeted model, env, and reward tests
  fixtures/           # Realistic PR data
`

### Design Notes
- The environment stays deterministic during grading.
- Observations evolve by review stage so multi-step episodes are meaningful.
- Reward shaping combines decision quality with evidence-grounded summaries and consistency checks.
- The API is designed to be easy to validate locally and on Hugging Face Spaces.

## Additional Documentation

- **[JUDGES_GUIDE.md](JUDGES_GUIDE.md)** - Evaluation criteria breakdown
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and technical architecture
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment across platforms
- **[SCORING_ANALYSIS.md](SCORING_ANALYSIS.md)** - Reward function design and mathematics
- **[COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md)** - Positioning and comparison notes

## Quick Judge Validation

`ash
# Clone and build
git clone <repository-url>
cd pr-review-env
docker build -t pr-review-env .

# Run and verify
docker run --rm -p 7860:7860 pr-review-env &
sleep 3
curl http://localhost:7860/health

# Test evaluation
export OPENAI_API_KEY=your_token
python inference.py
`

Expected outcome: the API responds, the baseline runs, and the staged benchmark produces deterministic scores across all tasks.
