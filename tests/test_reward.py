"""Unit tests for reward function"""

from __future__ import annotations

import pytest

from pr_review_env.models import Action, Observation
from pr_review_env.reward import compute_reward, compute_reward_breakdown


class TestRewardFunction:
    """Test suite for reward computation"""

    def test_perfect_score_easy_task(self):
        """Test perfect score on easy task"""
        obs = Observation(
            pr_id=1011,
            title="Fix off-by-one in window_slice",
            description="Fixes off-by-one bug",
            diff="+ return list(items[start : end + 1])",
            comments=["alice: LGTM"],
            files_changed=["utils/list_helpers.py"],
            author="nina",
            base_branch="main",
            additions=4,
            deletions=2,
            current_step=1,
            max_steps=4,
            task_name="easy",
        )
        
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="LGTM - fixes the off-by-one error correctly. Good catch on the slice bounds."
        )
        
        gold = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "gold_keywords": ["off-by-one", "slice", "fix"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        assert breakdown.decision_score == pytest.approx(0.9)
        assert breakdown.label_score == pytest.approx(0.95)
        assert breakdown.priority_score == pytest.approx(0.95)
        assert breakdown.summary_score == pytest.approx(0.95)
        assert breakdown.total == pytest.approx(0.9375)
        assert breakdown.step_penalty == 0.0

    def test_security_task_high_priority(self):
        """Test security task gets high priority weighting"""
        obs = Observation(
            pr_id=2044,
            title="Auth middleware cleanup",
            description="Refactors middleware",
            diff="- expires_at = payload.get('exp')",
            comments=["maya: This removes token expiry"],
            files_changed=["src/auth/middleware.py"],
            author="kevin",
            base_branch="release/2026.04",
            additions=22,
            deletions=13,
            current_step=1,
            max_steps=6,
            task_name="medium",
        )
        
        action = Action(
            decision="request_changes",
            labels=["security", "breaking-change"],
            priority="critical",
            review_summary="This removes token expiry enforcement creating a security vulnerability. Please restore expiry checks."
        )
        
        gold = {
            "decision": "request_changes",
            "labels": ["security", "breaking-change"],
            "priority": "critical",
            "gold_keywords": ["token expiry", "session", "security risk"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        assert breakdown.decision_score == pytest.approx(0.9)
        assert breakdown.label_score == pytest.approx(0.95)
        assert breakdown.priority_score == pytest.approx(0.95)
        assert breakdown.summary_score == pytest.approx(0.5)
        assert breakdown.total == pytest.approx(0.825)

    def test_partial_credit_decision(self):
        """Test partial credit for close decisions"""
        obs = Observation(
            pr_id=1011,
            title="Fix off-by-one",
            description="Simple fix",
            diff="+ return list(items[start : end + 1])",
            comments=["alice: LGTM"],
            files_changed=["utils/list_helpers.py"],
            author="nina",
            base_branch="main",
            additions=4,
            deletions=2,
            current_step=1,
            max_steps=4,
            task_name="easy",
        )
        
        # Wrong decision but same category
        action = Action(
            decision="request_changes",  # Should be approve
            labels=["bug"],
            priority="low",
            review_summary="This looks good but needs more tests."
        )
        
        gold = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "gold_keywords": ["off-by-one", "slice", "fix"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        assert breakdown.decision_score == pytest.approx(0.1)  # Partial credit for same category
        assert breakdown.label_score == pytest.approx(0.95)
        assert breakdown.priority_score == pytest.approx(0.95)

    def test_zero_credit_close_vs_review(self):
        """Test zero credit for close vs review decisions"""
        obs = Observation(
            pr_id=1011,
            title="Fix off-by-one",
            description="Simple fix",
            diff="+ return list(items[start : end + 1])",
            comments=["alice: LGTM"],
            files_changed=["utils/list_helpers.py"],
            author="nina",
            base_branch="main",
            additions=4,
            deletions=2,
            current_step=1,
            max_steps=4,
            task_name="easy",
        )
        
        # Close vs approve - completely different category
        action = Action(
            decision="close",  # Should be approve
            labels=["bug"],
            priority="low",
            review_summary="This is spam."
        )
        
        gold = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "gold_keywords": ["off-by-one", "slice", "fix"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        assert breakdown.decision_score == pytest.approx(0.1)  # No credit for close vs approve

    def test_priority_distance_scoring(self):
        """Test priority distance scoring"""
        obs = Observation(
            pr_id=1011,
            title="Fix off-by-one",
            description="Simple fix",
            diff="+ return list(items[start : end + 1])",
            comments=["alice: LGTM"],
            files_changed=["utils/list_helpers.py"],
            author="nina",
            base_branch="main",
            additions=4,
            deletions=2,
            current_step=1,
            max_steps=4,
            task_name="easy",
        )
        
        gold = {"priority": "critical", "decision": "approve", "labels": [], "gold_keywords": []}
        
        # Test distance scoring
        test_cases = [
            ("critical", 0.95),  # Exact match
            ("high", 0.6),      # Off by 1
            ("medium", 0.3),    # Off by 2
            ("low", 0.05),      # Off by 3
        ]
        
        for priority, expected_score in test_cases:
            action = Action(
                decision="approve",
                labels=[],
                priority=priority,
                review_summary="Test summary"
            )
            
            breakdown = compute_reward_breakdown(obs, action, gold)
            assert breakdown.priority_score == expected_score

    def test_summary_length_penalties(self):
        """Test summary length penalties"""
        obs = Observation(
            pr_id=1011,
            title="Fix off-by-one",
            description="Simple fix",
            diff="+ return list(items[start : end + 1])",
            comments=["alice: LGTM"],
            files_changed=["utils/list_helpers.py"],
            author="nina",
            base_branch="main",
            additions=4,
            deletions=2,
            current_step=1,
            max_steps=4,
            task_name="easy",
        )
        
        gold = {"decision": "approve", "labels": [], "priority": "low", "gold_keywords": []}
        
        # Test length penalties
        test_cases = [
            ("Too short", 0.05),  # < 20 chars
            ("A" * 600, 0.05),   # > 500 chars
            ("Perfect length summary with good content", 0.3),  # Good length with no keywords
        ]
        
        for summary, expected_min_score in test_cases:
            action = Action(
                decision="approve",
                labels=[],
                priority="low",
                review_summary=summary
            )
            
            breakdown = compute_reward_breakdown(obs, action, gold)
            assert breakdown.summary_score >= expected_min_score

    def test_step_penalty(self):
        """Test step penalty accumulation"""
        obs = Observation(
            pr_id=1011,
            title="Fix off-by-one",
            description="Simple fix",
            diff="+ return list(items[start : end + 1])",
            comments=["alice: LGTM"],
            files_changed=["utils/list_helpers.py"],
            author="nina",
            base_branch="main",
            additions=4,
            deletions=2,
            current_step=3,  # Step 3
            max_steps=4,
            task_name="easy",
        )
        
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="LGTM - fixes the off-by-one error correctly."
        )
        
        gold = {
            "decision": "approve",
            "labels": ["bug"],
            "priority": "low",
            "gold_keywords": ["off-by-one", "slice", "fix"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        # Step 3 should have penalty of 0.02 * (3-1) = 0.04
        assert breakdown.step_penalty == 0.04
        assert breakdown.total < 1.0  # Should be reduced by penalty

    def test_critical_label_bonus(self):
        """Test bonus for getting critical labels right"""
        obs = Observation(
            pr_id=2044,
            title="Auth middleware cleanup",
            description="Refactors middleware",
            diff="- expires_at = payload.get('exp')",
            comments=["maya: This removes token expiry"],
            files_changed=["src/auth/middleware.py"],
            author="kevin",
            base_branch="release/2026.04",
            additions=22,
            deletions=13,
            current_step=1,
            max_steps=6,
            task_name="medium",
        )
        
        action = Action(
            decision="request_changes",
            labels=["security", "breaking-change"],
            priority="critical",
            review_summary="Security issue with token expiry."
        )
        
        gold = {
            "decision": "request_changes",
            "labels": ["security", "breaking-change"],
            "priority": "critical",
            "gold_keywords": ["token expiry", "session", "security risk"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        # Should get bonus for critical labels
        assert breakdown.label_score >= 0.9

    def test_race_condition_keywords(self):
        """Test race condition keyword matching"""
        obs = Observation(
            pr_id=3099,
            title="Rate limiter improvement",
            description="Adds Redis rate limiter",
            diff="current = int(current_raw) if current_raw else 0",
            comments=["sre-olivia: This is a TOCTOU race"],
            files_changed=["api/gateway/rate_limiter.py"],
            author="max",
            base_branch="main",
            additions=94,
            deletions=4,
            current_step=1,
            max_steps=8,
            task_name="hard",
        )
        
        action = Action(
            decision="request_changes",
            labels=["bug", "needs-tests"],
            priority="high",
            review_summary="The Redis rate limiter has a TOCTOU race condition under high concurrency."
        )
        
        gold = {
            "decision": "request_changes",
            "labels": ["bug", "needs-tests", "urgent"],
            "priority": "high",
            "gold_keywords": ["race condition", "TOCTOU", "Redis", "concurrency", "atomic"]
        }
        
        breakdown = compute_reward_breakdown(obs, action, gold)
        
        # Should match multiple keywords
        assert breakdown.summary_score >= 0.6
