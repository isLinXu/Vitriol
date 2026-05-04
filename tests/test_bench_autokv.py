"""Tests for vitriol.bench.autokv module."""
from vitriol.bench.autokv import Case, default_prompt_suite, prefix_match_tokens


class TestCase:
    def test_case_creation(self):
        case = Case(name="test", prompt="Hello")
        assert case.name == "test"
        assert case.prompt == "Hello"

    def test_case_immutable(self):
        case = Case(name="test", prompt="Hello")
        # frozen dataclass should raise
        try:
            case.name = "changed"
            assert False, "Should have raised"
        except Exception:
            pass


class TestDefaultPromptSuite:
    def test_returns_list(self):
        suite = default_prompt_suite()
        assert isinstance(suite, list)
        assert len(suite) == 4

    def test_suite_items_are_tuples(self):
        suite = default_prompt_suite()
        for item in suite:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_suite_names(self):
        suite = default_prompt_suite()
        names = [name for name, _ in suite]
        assert "code" in names
        assert "math" in names
        assert "zh" in names
        assert "reasoning" in names

    def test_suite_prompts_non_empty(self):
        suite = default_prompt_suite()
        for _, prompt in suite:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestPrefixMatchTokens:
    def test_empty_lists(self):
        assert prefix_match_tokens([], []) == 0

    def test_full_match(self):
        assert prefix_match_tokens([1, 2, 3], [1, 2, 3]) == 3

    def test_partial_match(self):
        assert prefix_match_tokens([1, 2, 3], [1, 2, 4]) == 2

    def test_no_match(self):
        assert prefix_match_tokens([1, 2, 3], [4, 5, 6]) == 0

    def test_first_element_match_only(self):
        assert prefix_match_tokens([1, 2, 3], [1, 5, 6]) == 1

    def test_different_lengths_shorter_first(self):
        assert prefix_match_tokens([1, 2], [1, 2, 3]) == 2

    def test_different_lengths_longer_first(self):
        assert prefix_match_tokens([1, 2, 3, 4], [1, 2, 3]) == 3

    def test_single_element_match(self):
        assert prefix_match_tokens([42], [42]) == 1

    def test_single_element_no_match(self):
        assert prefix_match_tokens([42], [99]) == 0
