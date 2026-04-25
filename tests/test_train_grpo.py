"""Unit tests for train_grpo.py pure utility functions."""

from __future__ import annotations

import json
import pytest

# train_grpo imports torch/datasets/transformers at module level,
# so skip this entire module when those are not installed.
pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from train_grpo import (
    strip_code_fences,
    _extract_first_json_object,
    _normalize_action,
    safe_json_loads,
    bootstrap_action,
    clamp_reward,
    heuristic_action_from_text,
)


class TestStripCodeFences:
    def test_plain_json(self):
        assert strip_code_fences('{"a":1}') == '{"a":1}'

    def test_json_fences(self):
        raw = '```json\n{"a":1}\n```'
        assert strip_code_fences(raw) == '{"a":1}'

    def test_plain_fences(self):
        raw = '```\n{"a":1}\n```'
        assert strip_code_fences(raw) == '{"a":1}'

    def test_whitespace_only(self):
        assert strip_code_fences("   ") == ""


class TestExtractFirstJsonObject:
    def test_bare_json(self):
        assert _extract_first_json_object('{"a":1}') == '{"a":1}'

    def test_json_embedded_in_text(self):
        raw = 'Here is the result: {"decision":"approve"} end'
        result = _extract_first_json_object(raw)
        assert result is not None
        assert json.loads(result)["decision"] == "approve"

    def test_no_json(self):
        assert _extract_first_json_object("hello world") is None

    def test_empty(self):
        assert _extract_first_json_object("") is None

    def test_nested_json(self):
        raw = '{"a": {"b": 1}}'
        result = _extract_first_json_object(raw)
        assert json.loads(result) == {"a": {"b": 1}}


class TestNormalizeAction:
    def test_valid_action(self):
        parsed = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "review_summary": "Looks good.",
        }
        result = _normalize_action(parsed)
        assert result is not None
        assert result["decision"] == "approve"

    def test_missing_keys(self):
        assert _normalize_action({"decision": "approve"}) is None

    def test_invalid_decision(self):
        parsed = {
            "decision": "yolo",
            "labels": ["bug"],
            "priority": "low",
            "review_summary": "LGTM",
        }
        assert _normalize_action(parsed) is None

    def test_invalid_priority(self):
        parsed = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "ultra",
            "review_summary": "LGTM",
        }
        assert _normalize_action(parsed) is None

    def test_string_labels_coerced(self):
        parsed = {
            "decision": "approve",
            "labels": "bug, security",
            "priority": "low",
            "review_summary": "test",
        }
        result = _normalize_action(parsed)
        assert result is not None
        assert "bug" in result["labels"]
        assert "security" in result["labels"]

    def test_invalid_labels_fallback_to_bug(self):
        parsed = {
            "decision": "approve",
            "labels": ["invalid_label"],
            "priority": "low",
            "review_summary": "test",
        }
        result = _normalize_action(parsed)
        assert result is not None
        assert result["labels"] == ["bug"]

    def test_deduplication(self):
        parsed = {
            "decision": "approve",
            "labels": ["bug", "bug", "security"],
            "priority": "low",
            "review_summary": "test",
        }
        result = _normalize_action(parsed)
        assert result["labels"] == ["bug", "security"]

    def test_summary_truncated(self):
        parsed = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "review_summary": "x" * 600,
        }
        result = _normalize_action(parsed)
        assert result is not None
        assert len(result["review_summary"]) == 500


class TestSafeJsonLoads:
    def test_valid(self):
        raw = json.dumps({
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "review_summary": "Fine.",
        })
        result = safe_json_loads(raw)
        assert result is not None
        assert result["decision"] == "approve"

    def test_fenced_json(self):
        raw = '```json\n' + json.dumps({
            "decision": "request_changes",
            "labels": ["security"],
            "priority": "high",
            "review_summary": "Not safe.",
        }) + '\n```'
        result = safe_json_loads(raw)
        assert result is not None
        assert result["decision"] == "request_changes"

    def test_garbage(self):
        assert safe_json_loads("not json at all") is None

    def test_non_dict(self):
        assert safe_json_loads("[1,2,3]") is None


class TestClampReward:
    def test_within_bounds(self):
        assert clamp_reward(0.5) == 0.5

    def test_below_min(self):
        assert clamp_reward(-1.0) == 0.01

    def test_above_max(self):
        assert clamp_reward(2.0) == 0.99


class TestBootstrapAction:
    def test_easy(self):
        action = bootstrap_action("easy")
        assert action["decision"] in {"approve", "request_changes", "close"}
        assert isinstance(action["labels"], list) and action["labels"]
        assert action["priority"] in {"low", "medium", "high", "critical"}
        assert action["review_summary"]

    def test_medium(self):
        action = bootstrap_action("medium")
        assert action["decision"] in {"approve", "request_changes", "close"}

    def test_hard(self):
        action = bootstrap_action("hard")
        assert action["decision"] in {"approve", "request_changes", "close"}

    def test_dynamic_task_id(self):
        """Dynamic task IDs like easy_1012 should still return valid fallbacks."""
        action = bootstrap_action("easy_1012")
        assert action["decision"] in {"approve", "request_changes", "close"}
        assert isinstance(action["labels"], list)

    def test_unknown_task_fallback(self):
        """Completely unknown task should still return a valid action."""
        action = bootstrap_action("nonexistent_xyz")
        assert action["decision"] in {"approve", "request_changes", "close"}
        assert isinstance(action["review_summary"], str)


class TestHeuristicAction:
    def test_approve_keyword(self):
        action = heuristic_action_from_text("This looks good, LGTM!", "easy")
        assert action["decision"] == "approve"

    def test_close_keyword(self):
        action = heuristic_action_from_text("We should close this PR", "easy")
        assert action["decision"] == "close"

    def test_security_keyword_extracts_label(self):
        action = heuristic_action_from_text(
            "This has a critical security vulnerability in the token handling", "medium"
        )
        assert "security" in action["labels"]
        assert action["priority"] == "critical"

    def test_fallback_on_empty(self):
        action = heuristic_action_from_text("", "easy")
        assert action["decision"] in {"approve", "request_changes", "close"}
        assert isinstance(action["labels"], list)
