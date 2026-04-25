"""Unit tests for markdown chunker."""
import pytest

from agent_core.utils.chunker import chunk_markdown, Chunk


class TestChunkMarkdown:
    def test_splits_at_h1(self):
        md = "# Chapter 1\n\nFirst content.\n\n# Chapter 2\n\nSecond content.\n"
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "Chapter 1"
        assert "First content." in chunks[0].body
        assert chunks[1].title == "Chapter 2"
        assert "Second content." in chunks[1].body

    def test_splits_at_h2_when_no_h1(self):
        md = "## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "Section A"
        assert chunks[1].title == "Section B"

    def test_prefers_h1_over_h2(self):
        md = "# Big\n\n## Sub\n\nContent.\n\n# Another\n\nMore.\n"
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "Big"
        assert "Sub" in chunks[0].body
        assert chunks[1].title == "Another"

    def test_content_before_first_heading(self):
        md = "Some intro text.\n\n# Chapter 1\n\nContent.\n"
        chunks = chunk_markdown(md, fallback_title="My Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "My Doc"
        assert "Some intro text." in chunks[0].body
        assert chunks[1].title == "Chapter 1"

    def test_no_headings_returns_single_chunk(self):
        md = "Just plain text.\n\nMore text.\n"
        chunks = chunk_markdown(md, fallback_title="Fallback")
        assert len(chunks) == 1
        assert chunks[0].title == "Fallback"
        assert "Just plain text." in chunks[0].body

    def test_single_heading_returns_single_chunk(self):
        md = "# Only One\n\nContent here.\n"
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 1
        assert chunks[0].title == "Only One"

    def test_skips_empty_body_chunks(self):
        md = "# Has Content\n\nReal content.\n\n# Empty\n\n# Also Has Content\n\nMore.\n"
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "Has Content"
        assert chunks[1].title == "Also Has Content"

    def test_empty_input(self):
        chunks = chunk_markdown("", fallback_title="Empty")
        assert len(chunks) == 0

    def test_whitespace_only_input(self):
        chunks = chunk_markdown("   \n\n  ", fallback_title="Empty")
        assert len(chunks) == 0

    def test_heading_text_stripped(self):
        md = "#   Spaces Around   \n\nContent.\n"
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert chunks[0].title == "Spaces Around"

    def test_ignores_heading_syntax_inside_backtick_fence(self):
        md = (
            "# Chapter 1\n\n"
            "Intro paragraph.\n\n"
            "```python\n"
            "# 1. Define chat model\n"
            "llm = ChatOpenAI()\n"
            "# 2. Build chain\n"
            "chain = prompt | llm\n"
            "```\n\n"
            "Outro paragraph.\n"
        )
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 1
        assert chunks[0].title == "Chapter 1"
        assert "Define chat model" in chunks[0].body
        assert "Build chain" in chunks[0].body

    def test_code_fence_between_real_headings(self):
        md = (
            "# Chapter 1\n\n"
            "```python\n"
            "# 1. Setup\n"
            "setup()\n"
            "# 2. Run\n"
            "run()\n"
            "```\n\n"
            "# Chapter 2\n\n"
            "Second chapter content.\n"
        )
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "Chapter 1"
        assert chunks[1].title == "Chapter 2"
        assert "Setup" in chunks[0].body
        assert "Run" in chunks[0].body
        assert "Second chapter content." in chunks[1].body

    def test_ignores_heading_syntax_inside_tilde_fence(self):
        md = (
            "# Chapter 1\n\n"
            "~~~\n"
            "# Not a heading\n"
            "~~~\n\n"
            "Body text.\n"
        )
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 1
        assert chunks[0].title == "Chapter 1"

    def test_code_fence_comments_do_not_override_real_heading_level(self):
        md = (
            "## Real Section\n\n"
            "```python\n"
            "# This looks like H1 but is not\n"
            "x = 1\n"
            "```\n\n"
            "Paragraph.\n\n"
            "## Another Real Section\n\n"
            "More.\n"
        )
        chunks = chunk_markdown(md, fallback_title="Doc")
        assert len(chunks) == 2
        assert chunks[0].title == "Real Section"
        assert chunks[1].title == "Another Real Section"
