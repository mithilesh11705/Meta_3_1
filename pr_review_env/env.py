from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Action, Observation, StepResult
from .reward import _MIN_SCORE, compute_reward_breakdown
from .tasks.easy import FIXTURE as EASY_FIXTURE, GOLD as EASY_GOLD
from .tasks.hard import FIXTURE as HARD_FIXTURE, GOLD as HARD_GOLD
from .tasks.medium import FIXTURE as MEDIUM_FIXTURE, GOLD as MEDIUM_GOLD


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    description: str
    difficulty: str
    fixture: dict[str, Any]
    gold: dict[str, Any]
    max_steps: int
    expected_score_range: tuple[float, float]


TASK_CONFIGS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        description="Trivial off-by-one bugfix PR with straightforward approval path.",
        difficulty="easy",
        fixture=EASY_FIXTURE,
        gold=EASY_GOLD,
        max_steps=4,
        expected_score_range=(0.01, 0.99),
    ),
    "medium": TaskConfig(
        task_id="medium",
        description="Authentication middleware refactor hiding removal of token expiry checks.",
        difficulty="medium",
        fixture=MEDIUM_FIXTURE,
        gold=MEDIUM_GOLD,
        max_steps=6,
        expected_score_range=(0.01, 0.99),
    ),
    "hard": TaskConfig(
        task_id="hard",
        description="Redis rate limiter PR with TOCTOU race and conflicting reviewer sentiment.",
        difficulty="hard",
        fixture=HARD_FIXTURE,
        gold=HARD_GOLD,
        max_steps=8,
        expected_score_range=(0.01, 0.99),
    ),
}


def _serialize_reward_breakdown(breakdown: Any) -> dict[str, float]:
    """Expose only strict (0, 1) score fields in API payloads."""
    return breakdown.model_dump(exclude={"step_penalty"})


class PRReviewEnv:
    def __init__(self) -> None:
        self._task_name: str = "easy"
        self._current_step: int = 1
        self._last_reward: float = _MIN_SCORE
        self._done: bool = False
        self._history: list[dict[str, Any]] = []
        self._observation: Observation | None = None
        self._gold: dict[str, Any] = {}
        self.reset("easy")

    def _build_observation(self, task_name: str) -> Observation:
        task = TASK_CONFIGS[task_name]
        fixture = task.fixture
        return Observation(
            pr_id=int(fixture["pr_id"]),
            title=str(fixture["title"]),
            description=str(fixture["description"]),
            diff=str(fixture["diff"]),
            comments=[str(c) for c in fixture["comments"]],
            files_changed=[str(path) for path in fixture["files_changed"]],
            author=str(fixture["author"]),
            base_branch=str(fixture["base_branch"]),
            additions=int(fixture["additions"]),
            deletions=int(fixture["deletions"]),
            current_step=self._current_step,
            max_steps=task.max_steps,
            task_name=task_name,
        )

    def reset(self, task_name: str) -> Observation:
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task: {task_name}")

        self._task_name = task_name
        self._current_step = 1
        self._last_reward = _MIN_SCORE
        self._done = False
        self._history = []
        self._gold = dict(TASK_CONFIGS[task_name].gold)
        self._observation = self._build_observation(task_name)
        return self._observation

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is complete. Call reset() before step().")
        if self._observation is None:
            raise RuntimeError("Environment not initialized")

        breakdown = compute_reward_breakdown(observation=self._observation, action=action, gold=self._gold)
        self._last_reward = breakdown.total
        reward_breakdown = _serialize_reward_breakdown(breakdown)

        self._done = breakdown.total >= 0.95 or self._current_step >= self._observation.max_steps
        self._history.append(
            {
                "step": self._current_step,
                "action": action.model_dump(),
                "reward": breakdown.total,
                "reward_breakdown": reward_breakdown,
                "done": self._done,
            }
        )

        if not self._done:
            self._current_step += 1

        self._observation = self._build_observation(self._task_name)

        return StepResult(
            observation=self._observation,
            reward=breakdown.total,
            done=self._done,
            info={
                "task": self._task_name,
                "step": self._history[-1]["step"],
                "reward_breakdown": reward_breakdown,
            },
        )

    def get_state(self) -> dict[str, Any]:
        if self._observation is None:
            raise RuntimeError("Environment not initialized")

        return {
            "task": self._task_name,
            "current_step": self._observation.current_step,
            "max_steps": self._observation.max_steps,
            "done": self._done,
            "last_reward": self._last_reward,
            "history": self._history,
            "observation": self._observation.model_dump(),
        }

    @staticmethod
    def tasks() -> list[dict[str, Any]]:
        return [
            {
                "id": cfg.task_id,
                "description": cfg.description,
                "difficulty": cfg.difficulty,
                "max_steps": cfg.max_steps,
                "expected_score_range": [cfg.expected_score_range[0], cfg.expected_score_range[1]],
            }
            for cfg in TASK_CONFIGS.values()
        ]
