from __future__ import annotations

import json
import os
import re
import sys
from typing import Any
from urllib import error, request

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


ENV_NAME = "pr-review-env"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MIN_SCORE = 0.01
MAX_SCORE = 0.99
TASKS = ("easy", "medium", "hard")


SYSTEM_PROMPT = """You are a senior code review triage agent.

Return only valid JSON with this exact schema:
{
  "decision": "approve|request_changes|close",
  "labels": ["bug|security|enhancement|documentation|breaking-change|needs-tests|trivial|urgent"],
  "priority": "low|medium|high|critical",
  "review_summary": "1-3 sentence review summary"
}
"""


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _bounded_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, value))


def _format_score(value: float) -> str:
    return f"{_bounded_score(value):.2f}"


def _format_action(action: dict[str, Any] | None) -> str:
    if action is None:
        return "null"
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def _http_post(path: str, payload: dict[str, Any], session_id: str | None = None) -> tuple[dict[str, Any], str | None]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["session_id"] = session_id

    req = request.Request(
        f"{ENV_BASE_URL}{path}",
        method="POST",
        data=body,
        headers=headers,
    )
    with request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data, resp.headers.get("session_id")


def _observation_prompt(observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_name": observation.get("task_name"),
            "title": observation.get("title"),
            "description": observation.get("description"),
            "diff": observation.get("diff"),
            "comments": observation.get("comments"),
            "files_changed": observation.get("files_changed"),
            "author": observation.get("author"),
            "base_branch": observation.get("base_branch"),
            "additions": observation.get("additions"),
            "deletions": observation.get("deletions"),
            "current_step": observation.get("current_step"),
            "max_steps": observation.get("max_steps"),
            "review_stage": observation.get("review_stage"),
            "stage_prompt": observation.get("stage_prompt"),
        },
        ensure_ascii=True,
    )


def _llm_action(client: OpenAI, observation: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    stage = observation.get("review_stage", "identify_risk")
    stage_guidance = observation.get("stage_prompt", "")
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current review stage: {stage}\n"
                    f"{stage_guidance}\n"
                    "Respond with the required JSON only. Keep the summary aligned to this stage while staying consistent overall.\n"
                    + _observation_prompt(observation)
                ),
            },
        ]
        last_error: str | None = None
        for _ in range(2):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""
            cleaned = _strip_code_fences(content)
            if not cleaned:
                last_error = "json_decode_error:empty_response"
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Return valid JSON only with exactly the required keys."})
                continue
            try:
                parsed = json.loads(cleaned)
                break
            except json.JSONDecodeError as exc:
                last_error = f"json_decode_error:{exc.msg}"
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Your last response was invalid. Return valid JSON only."})
        else:
            return None, last_error or "json_decode_error:unknown"
    except Exception as exc:
        return None, f"llm_error:{exc}"

    required_keys = {"decision", "labels", "priority", "review_summary"}
    if set(parsed.keys()) != required_keys:
        return None, "schema_error:invalid_keys"

    valid_decisions = {"approve", "request_changes", "close"}
    valid_priorities = {"low", "medium", "high", "critical"}
    valid_labels = {
        "bug",
        "security",
        "enhancement",
        "documentation",
        "breaking-change",
        "needs-tests",
        "trivial",
        "urgent",
    }

    if parsed["decision"] not in valid_decisions:
        return None, f"schema_error:invalid_decision:{parsed['decision']}"
    if parsed["priority"] not in valid_priorities:
        return None, f"schema_error:invalid_priority:{parsed['priority']}"
    if not isinstance(parsed["labels"], list):
        return None, "schema_error:labels_not_array"
    if any(label not in valid_labels for label in parsed["labels"]):
        return None, "schema_error:invalid_labels"
    if not isinstance(parsed["review_summary"], str) or not parsed["review_summary"].strip():
        return None, "schema_error:empty_summary"

    return parsed, None


def run_task(client: OpenAI, task: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

    rewards: list[float] = []
    done = False
    success = False
    steps = 0

    try:
        observation, session_id = _http_post("/reset", {"task": task})
    except error.URLError as exc:
        print(f"[END] success=false steps=0 score={_format_score(MIN_SCORE)} rewards=")
        return

    for step in range(1, MAX_STEPS + 1):
        action, action_error = _llm_action(client, observation)

        if action is None:
            reward_value = MIN_SCORE
            rewards.append(reward_value)
            steps = step
            print(
                f"[STEP] step={step} action=null reward={_format_score(reward_value)} "
                f"done=false error={action_error or 'null'}"
            )
            continue

        try:
            step_result, _ = _http_post("/step", action, session_id=session_id)
            reward_value = _bounded_score(float(step_result.get("reward", MIN_SCORE)))
            done = bool(step_result.get("done", False))
            observation = step_result.get("observation", observation)
            step_error = "null"
        except error.URLError as exc:
            reward_value = MIN_SCORE
            done = False
            step_error = f"env_error:{exc}"

        rewards.append(reward_value)
        steps = step
        print(
            f"[STEP] step={step} action={_format_action(action)} reward={_format_score(reward_value)} "
            f"done={str(done).lower()} error={step_error}"
        )

        if done:
            success = True
            break

    score = _bounded_score(sum(rewards) / len(rewards)) if rewards else MIN_SCORE
    rewards_serialized = ",".join(_format_score(value) for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_format_score(score)} "
        f"rewards={rewards_serialized}"
    )


def main() -> int:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(client, task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
