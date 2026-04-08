"""Unit tests for Pydantic models"""

from __future__ import annotations

import pytest

from pr_review_env.models import Action, Observation, Reward, StepResult


class TestActionModel:
    """Test suite for Action model validation"""

    def test_valid_action(self):
        """Test valid action creation"""
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="LGTM - looks good to me."
        )
        
        assert action.decision == "approve"
        assert action.labels == ["bug"]
        assert action.priority == "low"
        assert action.review_summary == "LGTM - looks good to me."

    def test_duplicate_labels_rejected(self):
        """Test that duplicate labels are rejected"""
        with pytest.raises(ValueError, match="labels must not contain duplicates"):
            Action(
                decision="approve",
                labels=["bug", "bug"],  # Duplicate
                priority="low",
                review_summary="Test summary"
            )

    def test_invalid_labels_rejected(self):
        """Test that invalid labels are rejected"""
        with pytest.raises(ValueError, match="labels contain invalid values"):
            Action(
                decision="approve",
                labels=["invalid_label"],  # Not in allowed set
                priority="low",
                review_summary="Test summary"
            )

    def test_empty_summary_rejected(self):
        """Test that empty summary is rejected"""
        with pytest.raises(ValueError, match="review_summary must not be empty"):
            Action(
                decision="approve",
                labels=["bug"],
                priority="low",
                review_summary="   "  # Only whitespace
            )

    def test_invalid_decision_rejected(self):
        """Test that invalid decisions are rejected by Pydantic"""
        with pytest.raises(ValueError):
            Action(
                decision="invalid_decision",  # Not in enum
                labels=["bug"],
                priority="low",
                review_summary="Test summary"
            )

    def test_invalid_priority_rejected(self):
        """Test that invalid priorities are rejected by Pydantic"""
        with pytest.raises(ValueError):
            Action(
                decision="approve",
                labels=["bug"],
                priority="invalid_priority",  # Not in enum
                review_summary="Test summary"
            )

    def test_multiple_labels(self):
        """Test multiple valid labels"""
        action = Action(
            decision="request_changes",
            labels=["security", "breaking-change", "needs-tests"],
            priority="critical",
            review_summary="Security issue with breaking changes."
        )
        
        assert len(action.labels) == 3
        assert "security" in action.labels
        assert "breaking-change" in action.labels
        assert "needs-tests" in action.labels

    def test_all_allowed_labels(self):
        """Test all allowed labels are accepted"""
        allowed_labels = [
            "bug", "security", "enhancement", "documentation",
            "breaking-change", "needs-tests", "trivial", "urgent"
        ]
        
        action = Action(
            decision="approve",
            labels=allowed_labels,
            priority="medium",
            review_summary="Test with all labels"
        )
        
        assert set(action.labels) == set(allowed_labels)


class TestObservationModel:
    """Test suite for Observation model"""

    def test_valid_observation(self):
        """Test valid observation creation"""
        obs = Observation(
            pr_id=123,
            title="Test PR",
            description="Test description",
            diff="+ print('hello')",
            comments=["alice: LGTM"],
            files_changed=["test.py"],
            author="test_user",
            base_branch="main",
            additions=5,
            deletions=2,
            current_step=1,
            max_steps=4,
            task_name="easy"
        )
        
        assert obs.pr_id == 123
        assert obs.title == "Test PR"
        assert obs.current_step == 1
        assert obs.max_steps == 4

    def test_step_constraints(self):
        """Test step field constraints"""
        # Valid steps
        obs1 = Observation(
            pr_id=123, title="Test", description="Test", diff="test",
            comments=[], files_changed=[], author="test", base_branch="main",
            additions=0, deletions=0, current_step=1, max_steps=1, task_name="easy"
        )
        assert obs1.current_step == 1
        assert obs1.max_steps == 1

        # Invalid steps (should be rejected by Pydantic)
        with pytest.raises(ValueError):
            Observation(
                pr_id=123, title="Test", description="Test", diff="test",
                comments=[], files_changed=[], author="test", base_branch="main",
                additions=0, deletions=0, current_step=0, max_steps=4, task_name="easy"
            )

        with pytest.raises(ValueError):
            Observation(
                pr_id=123, title="Test", description="Test", diff="test",
                comments=[], files_changed=[], author="test", base_branch="main",
                additions=0, deletions=0, current_step=1, max_steps=0, task_name="easy"
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValueError):
            Observation(
                pr_id=123,
                title="Test",
                description="Test",
                diff="test",
                comments=[],
                files_changed=[],
                author="test",
                base_branch="main",
                additions=0,
                deletions=0,
                current_step=1,
                max_steps=4,
                task_name="easy",
                extra_field="not_allowed"  # Extra field should be rejected
            )


class TestRewardModel:
    """Test suite for Reward model"""

    def test_valid_reward(self):
        """Test valid reward creation"""
        reward = Reward(
            decision_score=0.999,
            label_score=0.8,
            priority_score=0.9,
            summary_score=0.7,
            step_penalty=0.02,
            total=0.87
        )
        
        assert reward.decision_score == 0.999
        assert reward.label_score == 0.8
        assert reward.priority_score == 0.9
        assert reward.summary_score == 0.7
        assert reward.step_penalty == 0.02
        assert reward.total == 0.87

    def test_score_bounds(self):
        """Test score bounds are enforced"""
        # Valid scores
        reward = Reward(
            decision_score=0.001,
            label_score=0.5,
            priority_score=0.999,
            summary_score=0.001,
            step_penalty=0.0,
            total=0.375
        )
        assert reward.decision_score == 0.001
        assert reward.priority_score == 0.999

        # Invalid scores (should be rejected by Pydantic)
        with pytest.raises(ValueError):
            Reward(
                decision_score=-0.1,  # Below 0.0
                label_score=0.5,
                priority_score=0.9,
                summary_score=0.7,
                step_penalty=0.0,
                total=0.5
            )

        with pytest.raises(ValueError):
            Reward(
                decision_score=1.1,  # Above 1.0
                label_score=0.5,
                priority_score=0.9,
                summary_score=0.7,
                step_penalty=0.0,
                total=0.5
            )


class TestStepResultModel:
    """Test suite for StepResult model"""

    def test_valid_step_result(self):
        """Test valid step result creation"""
        obs = Observation(
            pr_id=123, title="Test", description="Test", diff="test",
            comments=[], files_changed=[], author="test", base_branch="main",
            additions=0, deletions=0, current_step=1, max_steps=4, task_name="easy"
        )
        
        result = StepResult(
            observation=obs,
            reward=0.85,
            done=False,
            info={"step": 1, "task": "easy"}
        )
        
        assert result.observation == obs
        assert result.reward == 0.85
        assert result.done is False
        assert result.info["step"] == 1
        assert result.info["task"] == "easy"

    def test_reward_bounds(self):
        """Test reward bounds are enforced"""
        obs = Observation(
            pr_id=123, title="Test", description="Test", diff="test",
            comments=[], files_changed=[], author="test", base_branch="main",
            additions=0, deletions=0, current_step=1, max_steps=4, task_name="easy"
        )
        
        # Valid reward
        result = StepResult(
            observation=obs,
            reward=0.001,  # Minimum strictly greater than 0
            done=False,
            info={}
        )
        assert result.reward == 0.001

        result = StepResult(
            observation=obs,
            reward=0.999,  # Maximum strictly less than 1
            done=False,
            info={}
        )
        assert result.reward == 0.999

        # Invalid reward (should be rejected by Pydantic)
        with pytest.raises(ValueError):
            StepResult(
                observation=obs,
                reward=-0.1,  # Below 0.0
                done=False,
                info={}
            )

        with pytest.raises(ValueError):
            StepResult(
                observation=obs,
                reward=1.1,  # Above 1.0
                done=False,
                info={}
            )
