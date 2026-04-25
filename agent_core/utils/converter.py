"""DocumentConverter — convert local files to markdown via MarkItDown.

Supported formats: PDF, DOCX, XLSX, PPTX, HTML, HTM, EPUB, CSV.
"""
from dataclasses import dataclass
from pathlib import Path

from markitdown import MarkItDown


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".html", ".htm", ".epub", ".csv",
}


class ConversionError(Exception):
    """Raised when a file cannot be converted."""


@dataclass
class ConvertResult:
    title: str
    text: str
    source_path: str


class DocumentConverter:
    def __init__(self) -> None:
        self._md = MarkItDown()

    def convert(self, path: Path) -> ConvertResult:
        """Convert a local file to markdown.

        Args:
            path: path to the file to convert

        Returns:
            ConvertResult with title, markdown text, and source path

        Raises:
            ConversionError: if the file is missing, unsupported, or empty
        """
        if not path.exists():
            raise ConversionError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ConversionError(
                f"Unsupported format: {ext}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        try:
            result = self._md.convert(str(path))
        except Exception as exc:
            raise ConversionError(f"Conversion failed: {exc}") from exc

        text = result.text_content or ""
        if not text.strip():
            raise ConversionError(f"Conversion produced no content: {path.name}")

        title = result.title or path.stem

        return ConvertResult(
            title=title,
            text=text,
            source_path=str(path),
        )
