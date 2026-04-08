"""Unit tests for environment"""

from __future__ import annotations

import pytest

from pr_review_env.env import PRReviewEnv, TASK_CONFIGS
from pr_review_env.models import Action


class TestPRReviewEnv:
    """Test suite for PRReviewEnv"""

    def test_environment_initialization(self):
        """Test environment initialization"""
        env = PRReviewEnv()
        
        # Should start with easy task by default
        assert env._task_name == "easy"
        assert env._current_step == 1
        assert env._done is False
        assert env._last_reward == 0.001
        assert len(env._history) == 0

    def test_reset_valid_task(self):
        """Test resetting to valid task"""
        env = PRReviewEnv()
        
        # Reset to medium task
        obs = env.reset("medium")
        
        assert env._task_name == "medium"
        assert env._current_step == 1
        assert env._done is False
        assert obs.task_name == "medium"
        assert obs.max_steps == TASK_CONFIGS["medium"].max_steps

    def test_reset_invalid_task(self):
        """Test resetting to invalid task raises error"""
        env = PRReviewEnv()
        
        with pytest.raises(ValueError, match="Unsupported task"):
            env.reset("invalid_task")

    def test_step_when_done_raises_error(self):
        """Test that stepping when done raises RuntimeError"""
        env = PRReviewEnv()
        
        # Force done state
        env._done = True
        
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="Test summary"
        )
        
        with pytest.raises(RuntimeError, match="Episode is complete"):
            env.step(action)

    def test_successful_step(self):
        """Test successful step execution"""
        env = PRReviewEnv()
        env.reset("easy")
        
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="LGTM - fixes the off-by-one error correctly."
        )
        
        result = env.step(action)
        
        assert result.observation is not None
        assert 0.0 < result.reward < 1.0
        assert isinstance(result.done, bool)
        assert "task" in result.info
        assert "step" in result.info
        assert "reward_breakdown" in result.info

    def test_step_advances_current_step(self):
        """Test that step advances current step when not done"""
        env = PRReviewEnv()
        env.reset("easy")
        
        initial_step = env._current_step
        
        action = Action(
            decision="request_changes",  # Not perfect, so probably not done
            labels=["bug"],
            priority="low",
            review_summary="Needs more testing."
        )
        
        result = env.step(action)
        
        if not result.done:
            assert env._current_step == initial_step + 1
        else:
            assert env._current_step == initial_step  # Done, no advance

    def test_step_records_history(self):
        """Test that step records history"""
        env = PRReviewEnv()
        env.reset("easy")
        
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="LGTM"
        )
        
        initial_history_len = len(env._history)
        env.step(action)
        
        assert len(env._history) == initial_history_len + 1
        
        # Check history entry structure
        entry = env._history[-1]
        assert "step" in entry
        assert "action" in entry
        assert "reward" in entry
        assert "reward_breakdown" in entry
        assert "done" in entry

    def test_get_state(self):
        """Test getting environment state"""
        env = PRReviewEnv()
        env.reset("medium")
        
        # Take a step to create some history
        action = Action(
            decision="approve",
            labels=["enhancement"],
            priority="medium",
            review_summary="Looks good"
        )
        env.step(action)
        
        state = env.get_state()
        
        assert state["task"] == "medium"
        assert state["current_step"] == env._current_step
        assert state["max_steps"] == env._observation.max_steps
        assert state["done"] == env._done
        assert state["last_reward"] == env._last_reward
        assert len(state["history"]) == 1
        assert "observation" in state

    def test_tasks_static_method(self):
        """Test tasks static method"""
        tasks = PRReviewEnv.tasks()
        
        assert isinstance(tasks, list)
        assert len(tasks) == 3  # easy, medium, hard
        
        task_ids = {task["id"] for task in tasks}
        assert task_ids == {"easy", "medium", "hard"}
        
        # Check task structure
        for task in tasks:
            assert "id" in task
            assert "description" in task
            assert "difficulty" in task
            assert "max_steps" in task
            assert "expected_score_range" in task
            assert isinstance(task["expected_score_range"], list)
            assert len(task["expected_score_range"]) == 2

    def test_high_reward_triggers_done(self):
        """Test that high reward triggers done state"""
        env = PRReviewEnv()
        env.reset("easy")
        
        # Create a near-perfect action
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="LGTM - fixes the off-by-one error in window_slice function. Good catch on the slice bounds and the fix is correct."
        )
        
        result = env.step(action)
        
        # High reward should trigger done
        if result.reward >= 0.95:
            assert result.done is True
            assert env._done is True

    def test_max_steps_triggers_done(self):
        """Test that reaching max steps triggers done"""
        env = PRReviewEnv()
        env.reset("easy")
        
        # Force to max steps
        env._current_step = env._observation.max_steps
        
        action = Action(
            decision="approve",
            labels=["bug"],
            priority="low",
            review_summary="Test"
        )
        
        result = env.step(action)
        
        assert result.done is True
        assert env._done is True

    def test_observation_updates_after_step(self):
        """Test that observation updates after step when not done"""
        env = PRReviewEnv()
        env.reset("easy")
        
        initial_obs_step = env._observation.current_step
        
        action = Action(
            decision="request_changes",  # Not perfect, so probably not done
            labels=["bug"],
            priority="low",
            review_summary="Needs work"
        )
        
        result = env.step(action)
        
        if not result.done:
            assert result.observation.current_step == initial_obs_step + 1
            assert env._observation.current_step == initial_obs_step + 1

    def test_task_configs_structure(self):
        """Test task configs have correct structure"""
        for task_id, config in TASK_CONFIGS.items():
            assert task_id in ["easy", "medium", "hard"]
            assert hasattr(config, 'task_id')
            assert hasattr(config, 'description')
            assert hasattr(config, 'difficulty')
            assert hasattr(config, 'fixture')
            assert hasattr(config, 'gold')
            assert hasattr(config, 'max_steps')
            assert hasattr(config, 'expected_score_range')
            
            # Check fixture has required keys
            fixture = config.fixture
            required_fixture_keys = {
                'pr_id', 'title', 'description', 'diff', 'comments',
                'files_changed', 'author', 'base_branch', 'additions',
                'deletions', 'gold'
            }
            assert set(fixture.keys()) == required_fixture_keys
            
            # Check gold has required keys
            gold = config.gold
            required_gold_keys = {'decision', 'labels', 'priority', 'gold_keywords'}
            assert set(gold.keys()) == required_gold_keys

    def test_environment_state_persistence(self):
        """Test environment state persists across operations"""
        env = PRReviewEnv()
        
        # Reset to hard task
        obs1 = env.reset("hard")
        assert obs1.task_name == "hard"
        assert env._task_name == "hard"
        
        # Take multiple steps
        for i in range(3):
            action = Action(
                decision="request_changes",
                labels=["bug"],
                priority="high",
                review_summary=f"Review step {i+1}"
            )
            env.step(action)
        
        # State should persist
        assert env._task_name == "hard"
        assert env._current_step == 4  # Started at 1, took 3 steps
        assert len(env._history) == 3
        
        # Reset to different task
        obs2 = env.reset("easy")
        assert obs2.task_name == "easy"
        assert env._task_name == "easy"
        assert env._current_step == 1  # Reset to step 1
        assert len(env._history) == 0  # History cleared
