import html
import re
import unicodedata

from bs4 import BeautifulSoup


def strip_html(text):
    """Decode HTML entities and remove HTML tags while keeping their text."""
    if not isinstance(text, str):
        return ""
    decoded = html.unescape(html.unescape(text))
    if not re.search(r"</?[a-zA-Z][^>]*>", decoded):
        return decoded
    soup = BeautifulSoup(decoded, "html.parser")
    return soup.get_text(separator=" ")


def normalize_accents(text):
    """Normalize unicode accents (e.g. é -> e)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def clean_text(text, remove_accents=False):
    """Apply conservative cleaning while preserving product information."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    if remove_accents:
        text = normalize_accents(text)
    return " ".join(text.lower().split())


def combine_text(designation, description):
    """Combine cleaned designation and description without empty separators."""
    parts = [part for part in (designation, description) if part]
    return " ".join(parts)
