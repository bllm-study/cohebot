import re
import unicodedata


class WikiTextCleaner:
    """Simplified Wikipedia text cleaner.

    Hugging Face dataset already has wiki markup cleaned, so this
    class only does Unicode normalization and content validation.
    """

    NORMALIZE_PATTERNS = [
        (r"[ \t]+", " "),
        (r"\n{3,}", "\n\n"),
        (r"^[ \t]+|[ \t]+$", ""),
    ]

    def __init__(self, min_text_length: int = 100):
        """Initialize text cleaner.

        Args:
            min_text_length: Minimum text length for valid articles.
        """
        self.min_text_length = min_text_length
        self.normalize_patterns = [
            (re.compile(p, re.MULTILINE), r)
            for p, r in self.NORMALIZE_PATTERNS
        ]

    def normalize_text(self, text: str) -> str:
        """Normalize text with Unicode and spacing rules.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        text = unicodedata.normalize("NFC", text)

        for pattern, replacement in self.normalize_patterns:
            text = pattern.sub(replacement, text)

        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def clean(self, text: str) -> str:
        """Clean text (Unicode normalization only).

        Args:
            text: Text to clean.

        Returns:
            Cleaned text.
        """
        text = self.normalize_text(text)
        return text.strip()

    def is_valid(self, text: str) -> bool:
        """Validate text content.

        Args:
            text: Text to validate.

        Returns:
            True if text passes validation, False otherwise.
        """
        if len(text) < self.min_text_length:
            return False

        korean_chars = len(re.findall(r"[가-힣]", text))
        if korean_chars / max(len(text), 1) < 0.3:
            return False

        return True
