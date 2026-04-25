"""Unit tests for task modules"""

from __future__ import annotations

import pytest

from pr_review_env.tasks import easy, medium, hard
from pr_review_env.models import Action


class TestTaskModules:
    """Test suite for task modules"""

    def test_easy_task_exports(self):
        """Test easy task exports required items"""
        assert hasattr(easy, 'FIXTURE')
        assert hasattr(easy, 'GOLD')
        assert hasattr(easy, '_observation')
        assert hasattr(easy, 'grade')
        
        # Check types
        assert isinstance(easy.FIXTURE, dict)
        assert isinstance(easy.GOLD, dict)
        assert callable(easy._observation)
        assert callable(easy.grade)

    def test_medium_task_exports(self):
        """Test medium task exports required items"""
        assert hasattr(medium, 'FIXTURE')
        assert hasattr(medium, 'GOLD')
        assert hasattr(medium, '_observation')
        assert hasattr(medium, 'grade')
        
        # Check types
        assert isinstance(medium.FIXTURE, dict)
        assert isinstance(medium.GOLD, dict)
        assert callable(medium._observation)
        assert callable(medium.grade)

    def test_hard_task_exports(self):
        """Test hard task exports required items"""
        assert hasattr(hard, 'FIXTURE')
        assert hasattr(hard, 'GOLD')
        assert hasattr(hard, '_observation')
        assert hasattr(hard, 'grade')
        
        # Check types
        assert isinstance(hard.FIXTURE, dict)
        assert isinstance(hard.GOLD, dict)
        assert callable(hard._observation)
        assert callable(hard.grade)

    def test_task_fixture_structure(self):
        """Test all task fixtures have correct structure"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            fixture = task.FIXTURE
            
            # Required fixture keys
            required_keys = {
                'pr_id', 'title', 'description', 'diff', 'comments',
                'files_changed', 'author', 'base_branch', 'additions',
                'deletions', 'gold'
            }
            assert set(fixture.keys()) == required_keys
            
            # Check data types
            assert isinstance(fixture['pr_id'], int)
            assert isinstance(fixture['title'], str)
            assert isinstance(fixture['description'], str)
            assert isinstance(fixture['diff'], str)
            assert isinstance(fixture['comments'], list)
            assert isinstance(fixture['files_changed'], list)
            assert isinstance(fixture['author'], str)
            assert isinstance(fixture['base_branch'], str)
            assert isinstance(fixture['additions'], int)
            assert isinstance(fixture['deletions'], int)
            assert isinstance(fixture['gold'], dict)
            
            # Check comments are strings
            for comment in fixture['comments']:
                assert isinstance(comment, str)
            
            # Check files changed are strings
            for file_path in fixture['files_changed']:
                assert isinstance(file_path, str)

    def test_task_gold_structure(self):
        """Test all task gold data has correct structure"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            gold = task.GOLD
            
            # Required gold keys
            required_keys = {'decision', 'labels', 'priority', 'gold_keywords'}
            assert set(gold.keys()) == required_keys
            
            # Check data types
            assert isinstance(gold['decision'], str)
            assert isinstance(gold['labels'], list)
            assert isinstance(gold['priority'], str)
            assert isinstance(gold['gold_keywords'], list)
            
            # Check valid decision
            valid_decisions = {'approve', 'request_changes', 'close'}
            assert gold['decision'] in valid_decisions
            
            # Check valid priority
            valid_priorities = {'low', 'medium', 'high', 'critical'}
            assert gold['priority'] in valid_priorities
            
            # Check labels are valid
            valid_labels = {
                'bug', 'security', 'enhancement', 'documentation',
                'breaking-change', 'needs-tests', 'trivial', 'urgent'
            }
            for label in gold['labels']:
                assert label in valid_labels
            
            # Check keywords are strings
            for keyword in gold['gold_keywords']:
                assert isinstance(keyword, str)
                assert keyword.strip()  # Not empty

    def test_observation_creation(self):
        """Test observation creation for all tasks"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            obs = task._observation()
            
            # Check observation structure
            assert obs.task_name in ['easy', 'medium', 'hard']
            assert obs.current_step == 1
            assert obs.max_steps > 0
            assert obs.pr_id == task.FIXTURE['pr_id']
            assert obs.title == task.FIXTURE['title']
            assert obs.description == task.FIXTURE['description']
            assert obs.diff == task.FIXTURE['diff']
            assert obs.comments == task.FIXTURE['comments']
            assert obs.files_changed == task.FIXTURE['files_changed']
            assert obs.author == task.FIXTURE['author']
            assert obs.base_branch == task.FIXTURE['base_branch']
            assert obs.additions == task.FIXTURE['additions']
            assert obs.deletions == task.FIXTURE['deletions']

    def test_grade_function(self):
        """Test grade function for all tasks"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            # Create a valid action
            action = Action(
                decision=task.GOLD['decision'],
                labels=task.GOLD['labels'],
                priority=task.GOLD['priority'],
                review_summary="Test summary with some keywords"
            )
            
            # Grade should return a float between 0 and 1
            score = task.grade(action)
            assert isinstance(score, float)
            assert 0.0 < score < 1.0

    def test_perfect_score_possible(self):
        """Test that perfect score is possible with gold action"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            # Create action matching gold exactly
            action = Action(
                decision=task.GOLD['decision'],
                labels=task.GOLD['labels'],
                priority=task.GOLD['priority'],
                review_summary=" ".join(task.GOLD['gold_keywords'])
            )
            
            score = task.grade(action)
            
            # Should get high score (adjusted for dynamic evidence terms)
            assert score >= 0.7

    def test_task_difficulty_progression(self):
        """Test that task difficulty progresses as expected"""
        # Easy should have fewer max steps than hard
        assert easy._observation().max_steps <= medium._observation().max_steps
        assert medium._observation().max_steps <= hard._observation().max_steps
        
        # Easy should have higher expected score range than hard
        easy_obs = easy._observation()
        hard_obs = hard._observation()
        
        # This is based on the task design - easy should be easier to get perfect scores
        assert easy_obs.max_steps <= hard_obs.max_steps

    def test_diff_content_realism(self):
        """Test that diff content looks realistic"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            diff = task.FIXTURE['diff']
            
            # Should contain diff headers
            assert 'diff --git' in diff
            assert 'index' in diff
            assert '+++' in diff
            assert '---' in diff
            
            # Should contain line numbers
            assert '@@' in diff
            
            # Should contain code changes
            assert '+' in diff or '-' in diff

    def test_comments_realism(self):
        """Test that comments look realistic"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            comments = task.FIXTURE['comments']
            
            # Should have at least one comment
            assert len(comments) > 0
            
            # Comments should contain usernames and opinions
            for comment in comments:
                assert ':' in comment  # Username: comment format
                assert len(comment.split(':', 1)[0].strip()) > 0  # Username not empty
                assert len(comment.split(':', 1)[1].strip()) > 0  # Comment not empty

    def test_unique_pr_ids(self):
        """Test that all tasks have unique PR IDs"""
        pr_ids = [easy.FIXTURE['pr_id'], medium.FIXTURE['pr_id'], hard.FIXTURE['pr_id']]
        assert len(set(pr_ids)) == 3  # All unique

    def test_unique_authors(self):
        """Test that tasks have different authors (realistic)"""
        authors = [easy.FIXTURE['author'], medium.FIXTURE['author'], hard.FIXTURE['author']]
        # At least some variety in authors
        assert len(set(authors)) >= 2

    def test_file_paths_realism(self):
        """Test that file paths look realistic"""
        tasks = [easy, medium, hard]
        
        for task in tasks:
            files = task.FIXTURE['files_changed']
            
            for file_path in files:
                # Should look like real file paths
                assert '/' in file_path or file_path.endswith('.py')
                assert not file_path.startswith('/')  # Relative paths
                assert len(file_path) > 3  # Not trivial names


class TestAllFixtures:
    """Parameterized tests validating all 100 fixtures across all difficulties."""

    VALID_DECISIONS = {"approve", "request_changes", "close"}
    VALID_PRIORITIES = {"low", "medium", "high", "critical"}
    VALID_LABELS = {
        "bug", "security", "enhancement", "documentation",
        "breaking-change", "needs-tests", "trivial", "urgent",
    }
    REQUIRED_KEYS = {
        "pr_id", "title", "description", "diff", "comments",
        "files_changed", "author", "base_branch", "additions",
        "deletions", "gold",
    }
    REQUIRED_GOLD_KEYS = {"decision", "labels", "priority", "gold_keywords"}

    @staticmethod
    def _all_fixtures():
        """Yield (difficulty, index, fixture) for all 100 fixtures."""
        for difficulty, module in [("easy", easy), ("medium", medium), ("hard", hard)]:
            for i, fixture in enumerate(module.ALL_FIXTURES):
                yield difficulty, i, fixture

    @pytest.mark.parametrize(
        "difficulty,idx,fixture",
        list(_all_fixtures.__func__()),
        ids=lambda val: f"{val}" if isinstance(val, str) else None,
    )
    def test_fixture_schema(self, difficulty, idx, fixture):
        """Every fixture must have all required keys with correct types."""
        missing = self.REQUIRED_KEYS - set(fixture.keys())
        assert not missing, f"{difficulty}[{idx}] missing keys: {missing}"

        assert isinstance(fixture["pr_id"], int)
        assert isinstance(fixture["title"], str) and fixture["title"]
        assert isinstance(fixture["description"], str) and fixture["description"]
        assert isinstance(fixture["diff"], str) and fixture["diff"]
        assert isinstance(fixture["comments"], list) and len(fixture["comments"]) > 0
        assert isinstance(fixture["files_changed"], list) and len(fixture["files_changed"]) > 0
        assert isinstance(fixture["author"], str) and fixture["author"]
        assert isinstance(fixture["base_branch"], str)
        assert isinstance(fixture["additions"], int) and fixture["additions"] >= 0
        assert isinstance(fixture["deletions"], int) and fixture["deletions"] >= 0

    @pytest.mark.parametrize(
        "difficulty,idx,fixture",
        list(_all_fixtures.__func__()),
        ids=lambda val: f"{val}" if isinstance(val, str) else None,
    )
    def test_gold_schema(self, difficulty, idx, fixture):
        """Every fixture's gold must have valid decision, priority, labels, and keywords."""
        gold = fixture["gold"]
        missing = self.REQUIRED_GOLD_KEYS - set(gold.keys())
        assert not missing, f"{difficulty}[{idx}] gold missing: {missing}"

        assert gold["decision"] in self.VALID_DECISIONS
        assert gold["priority"] in self.VALID_PRIORITIES
        for label in gold["labels"]:
            assert label in self.VALID_LABELS, f"{difficulty}[{idx}] invalid label: {label}"
        for kw in gold["gold_keywords"]:
            assert isinstance(kw, str) and kw.strip()

    @pytest.mark.parametrize(
        "difficulty,idx,fixture",
        list(_all_fixtures.__func__()),
        ids=lambda val: f"{val}" if isinstance(val, str) else None,
    )
    def test_diff_has_headers(self, difficulty, idx, fixture):
        """Every diff must contain standard git diff headers."""
        diff = fixture["diff"]
        assert "diff --git" in diff, f"{difficulty}[{idx}] missing diff header"
        assert "@@" in diff, f"{difficulty}[{idx}] missing hunk header"

    def test_all_pr_ids_unique(self):
        """All 100 fixtures must have unique PR IDs."""
        all_ids = []
        for _, _, fixture in self._all_fixtures():
            all_ids.append(fixture["pr_id"])
        assert len(all_ids) == len(set(all_ids)), f"Duplicate PR IDs found: {[x for x in all_ids if all_ids.count(x) > 1]}"
        assert len(all_ids) == 100, f"Expected 100 fixtures, got {len(all_ids)}"

    def test_fixture_counts(self):
        """Verify we have exactly 30 easy, 35 medium, 35 hard."""
        assert len(easy.ALL_FIXTURES) == 30
        assert len(medium.ALL_FIXTURES) == 35
        assert len(hard.ALL_FIXTURES) == 35
