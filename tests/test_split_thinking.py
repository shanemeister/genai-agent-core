"""Tests for _split_thinking — DeepSeek-R1 reasoning extraction."""

from core.api.routes.chat import _split_thinking


class TestSplitThinking:
    """DeepSeek-R1 emits reasoning in <think> blocks. _split_thinking separates
    the visible answer from the chain-of-thought reasoning."""

    def test_no_thinking(self):
        answer, reasoning = _split_thinking("Hello, world!")
        assert answer == "Hello, world!"
        assert reasoning is None

    def test_full_think_tags(self):
        raw = "<think>Let me reason about this.</think>The answer is 42."
        answer, reasoning = _split_thinking(raw)
        assert answer == "The answer is 42."
        assert reasoning == "Let me reason about this."

    def test_vllm_stripped_opening_tag(self):
        """vLLM chat template strips the opening <think> tag, leaving just
        reasoning_text...</think>\\n\\nactual_answer."""
        raw = "I need to think carefully about this.</think>\n\nThe capital is Paris."
        answer, reasoning = _split_thinking(raw)
        assert answer == "The capital is Paris."
        assert reasoning == "I need to think carefully about this."

    def test_multiple_think_blocks(self):
        raw = "<think>First thought.</think>Middle text.<think>Second thought.</think>Final answer."
        answer, reasoning = _split_thinking(raw)
        assert "Final answer" in answer
        assert "First thought" in reasoning
        assert "Second thought" in reasoning

    def test_empty_think_block(self):
        raw = "<think></think>Just the answer."
        answer, reasoning = _split_thinking(raw)
        assert answer == "Just the answer."
        assert reasoning is None

    def test_multiline_reasoning(self):
        raw = "<think>Step 1: analyze\nStep 2: consider\nStep 3: conclude</think>The result."
        answer, reasoning = _split_thinking(raw)
        assert answer == "The result."
        assert "Step 1" in reasoning
        assert "Step 3" in reasoning

    def test_whitespace_handling(self):
        raw = "  <think> some reasoning </think>  answer with spaces  "
        answer, reasoning = _split_thinking(raw)
        assert answer == "answer with spaces"
        assert reasoning == "some reasoning"

    def test_plain_text_stripped(self):
        answer, reasoning = _split_thinking("   lots of whitespace   ")
        assert answer == "lots of whitespace"
        assert reasoning is None
