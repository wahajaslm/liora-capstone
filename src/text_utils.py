from bs4 import BeautifulSoup
import unicodedata


def strip_html(text):
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def normalize_accents(text):
    """Normalize unicode accents (e.g. é -> e)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def clean_text(text, remove_accents=False):
    """Clean text: remove HTML, optional accent removal, and lowercase."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    if remove_accents:
        text = normalize_accents(text)
    return " ".join(text.lower().split())
