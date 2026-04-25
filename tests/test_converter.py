"""Unit tests for DocumentConverter."""
from pathlib import Path

import pytest

from agent_core.utils.converter import DocumentConverter, ConvertResult, ConversionError


FIXTURES = Path(__file__).parent / "fixtures"


class TestDocumentConverter:
    def setup_method(self):
        self.converter = DocumentConverter()

    def test_convert_csv(self):
        result = self.converter.convert(FIXTURES / "sample.csv")
        assert isinstance(result, ConvertResult)
        assert "Alice" in result.text
        assert "Engineer" in result.text
        assert result.source_path == str(FIXTURES / "sample.csv")

    def test_convert_html(self, tmp_path):
        html_file = tmp_path / "page.html"
        html_file.write_text(
            "<html><head><title>Test</title></head>"
            "<body><h1>Hello</h1><p>World</p></body></html>"
        )
        result = self.converter.convert(html_file)
        assert "Hello" in result.text
        assert result.title == "Test"

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "data.json"
        bad_file.write_text('{"key": "value"}')
        with pytest.raises(ConversionError, match="Unsupported"):
            self.converter.convert(bad_file)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConversionError, match="not found"):
            self.converter.convert(tmp_path / "missing.pdf")

    def test_empty_conversion(self, tmp_path):
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        with pytest.raises(ConversionError, match="no content"):
            self.converter.convert(empty_file)

    def test_title_from_filename(self, tmp_path):
        csv_file = tmp_path / "quarterly-report.csv"
        csv_file.write_text("A,B\n1,2\n")
        result = self.converter.convert(csv_file)
        assert result.title == "quarterly-report"

    def test_supported_extensions(self):
        from agent_core.utils.converter import SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".pptx" in SUPPORTED_EXTENSIONS
        assert ".html" in SUPPORTED_EXTENSIONS
        assert ".htm" in SUPPORTED_EXTENSIONS
        assert ".epub" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".json" not in SUPPORTED_EXTENSIONS
