"""Text normalisation and tokenisation shared by every stage (English + Persian aware)."""
import re
import unicodedata

# Arabic code points that commonly appear in Persian text -> Persian equivalents
_PERSIAN_MAP = str.maketrans({
    "ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ٱ": "ا",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_WS = re.compile(r"[ \t\r\f\v]+")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?؟。])\s+|\n{2,}")
_ABBREV = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Co|St|vs|etc|e\.g|i\.e|No|Fig)\.\s", re.I)
_PROTECT = "\u2024"   # one-dot leader, temporarily replaces the abbreviation period

STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "been", "by", "with", "as", "at", "it", "its", "this", "that", "these",
    "those", "from", "which", "what", "who", "how", "do", "does", "did", "not", "no",
    "can", "will", "has", "have", "had", "about", "into", "than", "then", "so", "if",
    # Persian
    "و", "در", "به", "از", "که", "این", "آن", "با", "را", "برای", "است", "هست", "بود",
    "می", "ها", "های", "یک", "تا", "بر", "هم", "نیز", "یا", "اما", "چه", "چی", "کدام",
    "شد", "شده", "شود", "کرد", "کند", "کنید", "دارد", "باشد", "ای", "هر",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PERSIAN_MAP)
    text = _DIACRITICS.sub("", text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(normalize(text).replace("\u200c", " ").lower())
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def split_sentences(text: str) -> list[str]:
    parts = []
    text = _ABBREV.sub(lambda m: m.group(0).replace(".", _PROTECT), text)
    for block in _SENT_SPLIT.split(text):
        block = block.replace(_PROTECT, ".").strip()
        if block:
            parts.append(block)
    return parts
