from __future__ import annotations

import json
import os
import re
import sys
from typing import Any
from urllib import error, request

from openai import OpenAI


ENV_NAME = "pr-review-env"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_STEPS = 8
_MIN_SCORE = 0.001
_MAX_SCORE = 0.999


SYSTEM_PROMPT = """You are a senior code review triage agent with 10+ years of experience in infrastructure and security.

Your task is to analyze pull requests and make triage decisions. Consider:
1. Code quality and correctness
2. Security implications
3. Breaking changes and compatibility
4. Testing coverage
5. Performance impact
6. Reviewer consensus and concerns

Return ONLY valid JSON with this exact schema:
{
  "decision": "approve|request_changes|close",
  "labels": ["bug|security|enhancement|documentation|breaking-change|needs-tests|trivial|urgent"],
  "priority": "low|medium|high|critical",
  "review_summary": "1-3 sentence review summary"
}

Decision guidelines:
- "approve": Code is correct, well-tested, no security issues, reviewers agree
- "request_changes": Has bugs, security issues, breaking changes, or insufficient testing
- "close": Spam, duplicate, or completely inappropriate PR

Priority guidelines:
- "critical": Security vulnerabilities, data loss risk, production breakage
- "high": Security concerns, breaking changes, significant bugs
- "medium": Moderate bugs, performance issues, missing tests
- "low": Minor issues, documentation, trivial fixes

Few-shot examples:

Example 1 - Security issue:
{"decision":"request_changes","labels":["security","breaking-change"],"priority":"critical","review_summary":"This removes token expiry enforcement creating a security vulnerability. Please restore expiry checks and add regression tests."}

Example 2 - Simple bugfix:
{"decision":"approve","labels":["bug"],"priority":"low","review_summary":"LGTM - fixes the off-by-one error correctly. Good catch on the slice bounds."}

Example 3 - Race condition:
{"decision":"request_changes","labels":["bug","needs-tests","urgent"],"priority":"high","review_summary":"The Redis rate limiter has a TOCTOU race condition. Use atomic operations or Lua script to fix concurrency."}

No markdown, no extra keys, no explanations - ONLY JSON."""


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_\-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _http_post(path: str, payload: dict[str, Any], session_id: str | None = None) -> tuple[dict[str, Any], str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ENV_BASE_URL}{path}",
        method="POST",
        data=body,
        headers={"Content-Type": "application/json", **({"session_id": session_id} if session_id else {})},
    )
    with request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        sid = resp.headers.get("session_id")
    return data, sid


def _format_observation(observation: dict[str, Any]) -> str:
    comments = "\n".join(f"  • {c}" for c in observation.get("comments", []))
    files_changed = "\n".join(f"  • {f}" for f in observation.get("files_changed", []))
    
    # Highlight key metrics
    net_change = observation.get('additions', 0) - observation.get('deletions', 0)
    
    return (
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║ PULL REQUEST REVIEW TRIAGE - Task: {observation.get('task_name').upper()}     ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n\n"
        f"📋 PR ID: {observation.get('pr_id')}\n"
        f"📝 Title: {observation.get('title')}\n"
        f"👤 Author: {observation.get('author')}  │  🌿 Base: {observation.get('base_branch')}\n"
        f"📊 Stats: +{observation.get('additions')} -{observation.get('deletions')} (net: {net_change:+d})\n"
        f"🔄 Step: {observation.get('current_step')} / {observation.get('max_steps')}\n\n"
        f"📄 Description:\n{observation.get('description')}\n\n"
        f"📁 Files Changed:\n{files_changed}\n\n"
        f"💬 Reviewer Comments:\n{comments}\n\n"
        f"⚡ Code Diff:\n{observation.get('diff')}\n\n"
        f"═══════════════════════════════════════════════════════════════"
    )


def _llm_action(client: OpenAI, observation: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    prompt = (
        "Analyze this pull request and provide your triage decision in JSON format.\n\n"
        + _format_observation(observation)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,  # Limit response length
        )
        content = response.choices[0].message.content or ""
        
        # Log raw response for debugging
        if len(content) > 1000:
            print(f"[DEBUG] LLM response truncated: {content[:200]}...")
        else:
            print(f"[DEBUG] LLM response: {content}")
            
    except Exception as exc:
        return None, f"llm_error:{exc}"

    cleaned = _strip_code_fences(content)
    
    # Additional validation for common issues
    if not cleaned.strip():
        return None, "json_decode_error:empty_response"
    
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed = cleaned.replace(", \n}", "\n}").replace(", }", "}")
            parsed = json.loads(fixed)
        except:
            return None, f"json_decode_error:{exc.msg}"

    # Validate schema
    required_keys = {"decision", "labels", "priority", "review_summary"}
    missing_keys = required_keys - set(parsed.keys())
    extra_keys = set(parsed.keys()) - required_keys
    
    if missing_keys:
        return None, f"schema_error:missing_keys:{','.join(missing_keys)}"
    if extra_keys:
        return None, f"schema_error:extra_keys:{','.join(extra_keys)}"
    
    # Validate values
    valid_decisions = {"approve", "request_changes", "close"}
    valid_priorities = {"low", "medium", "high", "critical"}
    valid_labels = {"bug", "security", "enhancement", "documentation", "breaking-change", "needs-tests", "trivial", "urgent"}
    
    if parsed["decision"] not in valid_decisions:
        return None, f"schema_error:invalid_decision:{parsed['decision']}"
    if parsed["priority"] not in valid_priorities:
        return None, f"schema_error:invalid_priority:{parsed['priority']}"
    
    if not isinstance(parsed["labels"], list):
        return None, "schema_error:labels_not_array"
    
    invalid_labels = [label for label in parsed["labels"] if label not in valid_labels]
    if invalid_labels:
        return None, f"schema_error:invalid_labels:{','.join(invalid_labels)}"
    
    if not isinstance(parsed["review_summary"], str) or not parsed["review_summary"].strip():
        return None, "schema_error:empty_summary"

    return parsed, ""


def _format_action(action: dict[str, Any] | None) -> str:
    if action is None:
        return "null"
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def _bounded_score(value: float) -> float:
    return max(_MIN_SCORE, min(_MAX_SCORE, value))


def run_task(client: OpenAI, task: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")
    try:
        observation, session_id = _http_post("/reset", {"task": task})
    except error.URLError as exc:
        print(f"[END] success=false steps=0 score=<{_MIN_SCORE:.3f}> rewards=error:{exc}")
        return

    rewards: list[float] = []
    done = False
    executed_steps = 0

    for step in range(1, MAX_STEPS + 1):
        action, parse_error = _llm_action(client, observation)

        if action is None:
            print(f"[STEP] step={step} action=\"null\" reward=<{_MIN_SCORE:.3f}> done=false error={parse_error}")
            executed_steps += 1
            rewards.append(_MIN_SCORE)
            continue

        try:
            step_result, _ = _http_post("/step", action, session_id=session_id)
        except error.URLError as exc:
            print(
                f"[STEP] step={step} action={_format_action(action)} reward=<{_MIN_SCORE:.3f}> "
                f"done=false error=env_error:{exc}"
            )
            executed_steps += 1
            rewards.append(_MIN_SCORE)
            continue

        reward_value = _bounded_score(float(step_result.get("reward", 0.0)))
        done = bool(step_result.get("done", False))
        err = ""
        print(
            f"[STEP] step={step} action={_format_action(action)} reward=<{reward_value:.3f}> "
            f"done={str(done).lower()} error={err}"
        )

        executed_steps += 1
        rewards.append(reward_value)
        observation = step_result.get("observation", observation)

        if done:
            break

    score = _bounded_score(sum(rewards) / len(rewards)) if rewards else _MIN_SCORE
    rewards_serialized = ",".join(f"{_bounded_score(value):.3f}" for value in rewards)
    print(
        f"[END] success={str(done).lower()} steps={executed_steps} "
        f"score=<{score:.3f}> rewards={rewards_serialized}"
    )


def main() -> int:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task in ("easy", "medium", "hard"):
        run_task(client, task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
