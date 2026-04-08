from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ALLOWED_LABELS: set[str] = {
    "bug",
    "security",
    "enhancement",
    "documentation",
    "breaking-change",
    "needs-tests",
    "trivial",
    "urgent",
}


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pr_id: int
    title: str
    description: str
    diff: str
    comments: list[str]
    files_changed: list[str]
    author: str
    base_branch: str
    additions: int
    deletions: int
    current_step: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    task_name: str
    review_stage: str = Field(default="identify_risk")
    stage_prompt: str = Field(default="")


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "request_changes", "close"]
    labels: list[str]
    priority: Literal["low", "medium", "high", "critical"]
    review_summary: str

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        if len(labels) != len(set(labels)):
            raise ValueError("labels must not contain duplicates")

        invalid = [label for label in labels if label not in ALLOWED_LABELS]
        if invalid:
            raise ValueError(f"labels contain invalid values: {invalid}")

        return labels

    @field_validator("review_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("review_summary must not be empty")
        return stripped


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_score: float = Field(gt=0.0, lt=1.0)
    label_score: float = Field(gt=0.0, lt=1.0)
    priority_score: float = Field(gt=0.0, lt=1.0)
    summary_score: float = Field(gt=0.0, lt=1.0)
    step_penalty: float = Field(ge=0.0)
    total: float = Field(gt=0.0, lt=1.0)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float = Field(gt=0.0, lt=1.0)
    done: bool
    info: dict[str, Any]
