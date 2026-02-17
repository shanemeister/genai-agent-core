"""Tests for sanitize_mermaid — LLM output cleanup for Mermaid diagrams."""

from core.api.routes.chat import sanitize_mermaid


class TestSanitizeMermaid:
    """sanitize_mermaid fixes common Mermaid syntax errors from LLM output."""

    def test_clean_input_unchanged(self):
        code = "flowchart TD\n    A[Start] --> B[End]"
        assert sanitize_mermaid(code) == code

    def test_truncated_trailing_line(self):
        """LLM hit max_tokens mid-node — unclosed bracket on last line."""
        code = "flowchart TD\n    A[Start] --> B[End]\n    C[Truncat"
        result = sanitize_mermaid(code)
        assert "Truncat" not in result
        assert "A[Start] --> B[End]" in result

    def test_quotes_in_brackets(self):
        code = 'A[Print "hello"] --> B[Done]'
        result = sanitize_mermaid(code)
        assert '"' not in result
        assert "Print hello" in result

    def test_parentheses_in_brackets(self):
        code = "A[Text (V0)] --> B[Done]"
        result = sanitize_mermaid(code)
        # Parentheses replaced with dash
        assert "(" not in result.split("-->")[0]
        assert "V0" in result

    def test_parentheses_in_diamonds(self):
        code = "A{Decision (yes)?} --> B[Done]"
        result = sanitize_mermaid(code)
        assert "(" not in result.split("-->")[0]

    def test_leading_dash_in_parens(self):
        code = "G(- End)"
        result = sanitize_mermaid(code)
        assert result == "G(End)"

    def test_leading_dash_in_brackets(self):
        code = "A[- Text]"
        result = sanitize_mermaid(code)
        assert result == "A[Text]"

    def test_semicolons_to_newlines(self):
        code = "A --> B; B --> C"
        result = sanitize_mermaid(code)
        assert ";" not in result
        assert "A --> B" in result
        assert "B --> C" in result

    def test_pipe_in_label(self):
        code = "AccuracyCalc[Calculate |A| / n]"
        result = sanitize_mermaid(code)
        # Pipes replaced with Unicode look-alike
        assert "|" not in result.split("[")[1]
        assert "\u2758" in result

    def test_redundant_id_suffix(self):
        """DeepSeek appends ' - NodeID' to labels."""
        code = "A[Input - A]"
        result = sanitize_mermaid(code)
        assert result == "A[Input]"

    def test_bare_parenthesized_node_gets_id(self):
        code = "flowchart TD\n    (Start) --> A[Step]"
        result = sanitize_mermaid(code)
        # Bare (Start) should get a node ID prefix
        assert "([Start])" in result
        assert result.count("-->") >= 1

    def test_multiple_truncated_lines(self):
        code = "flowchart TD\n    A[OK]\n    B{Trunc\n    C[Also trunc"
        result = sanitize_mermaid(code)
        assert "Trunc" not in result
        assert "Also trunc" not in result
        assert "A[OK]" in result
