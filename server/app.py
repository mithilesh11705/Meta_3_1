from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from pr_review_env.env import PRReviewEnv, TASK_CONFIGS, _serialize_reward_breakdown
from pr_review_env.models import Action, Observation, StepResult

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pr-review-env")

app = FastAPI(
    title="pr-review-env",
    version="2.0.0",
    description="""Pull Request Code Review Triage Environment

A deterministic OpenEnv benchmark that simulates real PR review workflows across
100 realistic scenarios spanning bugfix, security-sensitive, and contested
infrastructure changes. Agents are evaluated on decision quality, prioritization,
and review communication under realistic constraints.

## Features
- 100 PR scenarios across three difficulty levels (30 easy, 35 medium, 35 hard)
- Dense reward function with partial credit
- Deterministic grading without LLM calls
- Session-based concurrent support with TTL eviction
- Comprehensive logging and error handling
""",
    contact={
        "name": "Meta Infra Tooling",
        "url": "https://github.com/meta"
    },
    license={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

SESSION_TTL_SECONDS = 1800  # 30 minutes
SESSION_STORE: dict[str, PRReviewEnv] = {}
_SESSION_LAST_ACTIVE: dict[str, float] = {}


def _evict_expired_sessions() -> None:
    """Remove sessions idle longer than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [sid for sid, ts in _SESSION_LAST_ACTIVE.items() if now - ts > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSION_STORE.pop(sid, None)
        _SESSION_LAST_ACTIVE.pop(sid, None)
    if expired:
        logger.info(f"Evicted {len(expired)} expired sessions")


def _touch_session(session_id: str) -> None:
    """Update the last-active timestamp for a session."""
    _SESSION_LAST_ACTIVE[session_id] = time.time()


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = "easy"


def resolve_session_id(session_id: str | None = Header(default=None, alias="session_id")) -> str:
    return session_id or "default"


def get_env(session_id: str = Depends(resolve_session_id)) -> PRReviewEnv:
    _evict_expired_sessions()
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = PRReviewEnv()
    _touch_session(session_id)
    return SESSION_STORE[session_id]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception - path: {request.url.path}, error: {exc}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "unhandled_exception",
            "message": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )


@app.post("/reset", response_model=Observation)
def reset(
    body: ResetRequest | None = None,
    response: Response = None,
    session_id: str | None = Header(default=None, alias="session_id"),
) -> Observation:
    start_time = time.time()
    resolved_session_id = session_id or str(uuid4())
    
    task = body.task if body else "easy"
    
    logger.info(f"Reset request - task: {task}, session_id: {resolved_session_id}")
    
    if task not in TASK_CONFIGS:
        logger.error(f"Invalid task requested: {task}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "invalid_task",
                "message": f"Unknown task '{task}'",
                "available_tasks": list(TASK_CONFIGS.keys())
            }
        )

    try:
        env = SESSION_STORE.get(resolved_session_id)
        if env is None:
            _evict_expired_sessions()
            env = PRReviewEnv()
            SESSION_STORE[resolved_session_id] = env
            logger.info(f"Created new environment for session: {resolved_session_id}")
        _touch_session(resolved_session_id)
        
        observation = env.reset(task)
        
        if response:
            response.headers["session_id"] = resolved_session_id
        
        duration = time.time() - start_time
        logger.info(f"Reset completed - session: {resolved_session_id}, task: {task}, duration: {duration:.3f}s")
        
        return observation
        
    except Exception as exc:
        logger.error(f"Reset failed - session: {resolved_session_id}, error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "reset_failed",
                "message": "Failed to reset environment",
                "session_id": resolved_session_id
            }
        ) from exc


@app.post("/step", response_model=StepResult)
def step(action: Action, request: Request, env: PRReviewEnv = Depends(get_env)) -> StepResult:
    start_time = time.time()
    session_id = request.headers.get("session_id", "default")
    
    logger.info(f"Step request - session: {session_id}, action: {action.decision}, labels: {action.labels}")
    
    try:
        result = env.step(action)
        
        duration = time.time() - start_time
        logger.info(
            f"Step completed - session: {session_id}, reward: {result.reward:.3f}, "
            f"done: {result.done}, duration: {duration:.3f}s"
        )
        
        return result
        
    except RuntimeError as exc:
        logger.error(f"Step failed - session: {session_id}, error: {exc}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "step_failed",
                "message": str(exc),
                "session_id": session_id
            }
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected step error - session: {session_id}, error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Internal server error during step",
                "session_id": session_id
            }
        ) from exc


@app.get("/state")
def state(request: Request, env: PRReviewEnv = Depends(get_env)) -> dict[str, object]:
    session_id = request.headers.get("session_id", "default")
    
    try:
        state_data = env.get_state()
        logger.info(f"State retrieved - session: {session_id}, step: {state_data.get('current_step')}")
        return state_data
    except RuntimeError as exc:
        logger.error(f"State retrieval failed - session: {session_id}, error: {exc}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "state_failed",
                "message": str(exc),
                "session_id": session_id
            }
        ) from exc


@app.get("/tasks")
def tasks() -> dict[str, object]:
    logger.info("Tasks metadata requested")
    return {
        "tasks": PRReviewEnv.tasks(),
        "total_tasks": len(TASK_CONFIGS),
        "environment": "pr-review-env",
        "version": "1.0.0"
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Health check endpoint with system status"""
    return {
        "status": "ok",
        "environment": "pr-review-env",
        "version": "2.0.0",
        "total_tasks": len(TASK_CONFIGS),
        "active_sessions": len(SESSION_STORE),
        "available_tasks": list(TASK_CONFIGS.keys()),
        "timestamp": time.time()
    }


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    action: Action
    task: str


class ValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    valid: bool
    error: str | None = None
    reward_breakdown: dict[str, float] | None = None


@app.post("/validate", response_model=ValidationResponse)
def validate_action(body: ValidationRequest) -> ValidationResponse:
    """Validate an action against a task without affecting session state"""
    logger.info(f"Validation request - task: {body.task}, decision: {body.action.decision}")
    
    if body.task not in TASK_CONFIGS:
        return ValidationResponse(
            valid=False,
            error=f"Unknown task '{body.task}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    
    try:
        # Create temporary environment for validation
        temp_env = PRReviewEnv()
        temp_env.reset(body.task)
        
        # Get the reward breakdown
        from pr_review_env.reward import compute_reward_breakdown
        
        observation = temp_env._build_observation(body.task)
        gold = TASK_CONFIGS[body.task].gold
        
        breakdown = compute_reward_breakdown(
            observation=observation,
            action=body.action,
            gold=gold
        )
        
        logger.info(f"Validation completed - task: {body.task}, reward: {breakdown.total:.3f}")
        
        return ValidationResponse(
            valid=True,
            reward_breakdown=_serialize_reward_breakdown(breakdown)
        )
        
    except Exception as exc:
        logger.error(f"Validation failed - task: {body.task}, error: {exc}")
        return ValidationResponse(
            valid=False,
            error=f"Validation error: {exc}"
        )


class ActionExamplesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    examples: dict[str, dict[str, object]]


@app.get("/examples", response_model=ActionExamplesResponse)
def get_action_examples() -> ActionExamplesResponse:
    """Get example actions for each task difficulty level"""
    return ActionExamplesResponse(
        examples={
            "easy": {
                "decision": "approve",
                "labels": ["bug"],
                "priority": "low",
                "review_summary": "LGTM - fixes the off-by-one error in window_slice function. Good catch on the slice bounds."
            },
            "medium": {
                "decision": "request_changes",
                "labels": ["security", "breaking-change"],
                "priority": "critical",
                "review_summary": "This removes token expiry enforcement which creates a security vulnerability. Please restore expiry checks and add regression tests."
            },
            "hard": {
                "decision": "request_changes",
                "labels": ["bug", "needs-tests", "urgent"],
                "priority": "high",
                "review_summary": "The Redis rate limiter has a TOCTOU race condition. Use atomic operations or Lua script to fix the concurrency issue."
            }
        }
    )


@app.get("/metrics")
def get_metrics() -> dict[str, object]:
    """Get environment metrics and statistics"""
    total_sessions = len(SESSION_STORE)
    session_stats = []
    
    for session_id, env in SESSION_STORE.items():
        try:
            state = env.get_state()
            session_stats.append({
                "session_id": session_id,
                "task": state.get("task"),
                "current_step": state.get("current_step"),
                "max_steps": state.get("max_steps"),
                "done": state.get("done"),
                "last_reward": state.get("last_reward"),
                "history_length": len(state.get("history", []))
            })
        except Exception:
            # Skip sessions that are in an invalid state
            continue
    
    return {
        "total_sessions": total_sessions,
        "active_sessions": len([s for s in session_stats if not s.get("done", True)]),
        "completed_sessions": len([s for s in session_stats if s.get("done", False)]),
        "sessions": session_stats,
        "available_tasks": list(TASK_CONFIGS.keys()),
        "timestamp": time.time()
    }

def main():
    import uvicorn
    import sys
    import os
    # Ensure root is in path so pr_review_env can be imported
    sys.path.append(os.getcwd())
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
