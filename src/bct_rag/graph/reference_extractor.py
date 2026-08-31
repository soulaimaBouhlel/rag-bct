import re


LAW_PATTERN = re.compile(
    r"loi\s+n[°º]?\s*"
    r"(\d{2,4})[-\s](\d+)",
    re.IGNORECASE,
)

CIRCULAR_PATTERN = re.compile(
    r"circulaire(?:\s+aux\s+banques(?:\s+et\s+aux\s+établissements\s+financiers)?)?"
    r"\s+n[°º]?\s*"
    r"(\d{2,4})[-\s](\d+)",
    re.IGNORECASE,
)

# "article" or "articles", singular or plural, followed by one or more
# numbers possibly chained with "et"/","/"à" (e.g. "articles 63 et 66",
# "articles 12, 13 et 15", "articles 10 à 12")
ARTICLE_PATTERN = re.compile(
    r"articles?\s+"
    r"(\d+(?:\s*(?:,|et|à)\s*\d+)*)",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r"\d+")
ANNEX_PATTERN = re.compile(
    r"\bannexe\s*(?:n[°º]?\s*)?([A-Za-z]|\d+)?\b",
    re.IGNORECASE,
)


def _normalize_year(year: str) -> str:
    """
    Convert a 2-digit year to its 4-digit form.

    Tunisian regulatory texts sometimes cite older laws/circulars with a
    2-digit year (e.g. "loi n°94-14", "circulaire n°91-22"). These always
    refer to 19xx dates in this corpus, so 2-digit years are prefixed
    with "19". 4-digit years are returned unchanged.
    """

    if len(year) == 4:
        return year

    return f"19{year}"


def _normalize_number(number: str) -> str:
    """
    Zero-pad to at least 2 digits, matching the canonical circular_ref
    format produced by parse_circular_ref() (e.g. "1" -> "01"), so
    extracted references match the Circular/Law nodes created from
    filenames in the graph pipeline.
    """

    return number.zfill(2)


def _dedupe_preserve_order(items: list) -> list:
    """Remove duplicates while keeping first-seen order."""

    return list(dict.fromkeys(items))


def extract_laws(text: str) -> list[str]:
    """
    Extract law references such as:

        loi n°2016-35
        loi n°2016-48
        loi n°94-14
    """

    matches = LAW_PATTERN.findall(text)

    return _dedupe_preserve_order([
        f"{_normalize_year(year)}-{_normalize_number(number)}"
        for year, number in matches
    ])


def extract_circulars(text: str) -> list[str]:
    """
    Extract circular references.
    """

    matches = CIRCULAR_PATTERN.findall(text)

    return _dedupe_preserve_order([
        f"{_normalize_year(year)}-{_normalize_number(number)}"
        for year, number in matches
    ])


def extract_articles(text: str) -> list[int]:
    """
    Extract article numbers appearing in the text, including enumerations
    such as "articles 63 et 66" or "articles 12, 13 et 15".
    """

    numbers: list[int] = []

    for group in ARTICLE_PATTERN.findall(text):
        numbers.extend(
            int(n) for n in _NUMBER_PATTERN.findall(group)
        )

    return _dedupe_preserve_order(numbers)


def extract_references(text: str) -> dict:
    """
    Extract regulatory references from a chunk.
    """

    return {
        "laws": extract_laws(text),
        "circulars": extract_circulars(text),
        "articles": extract_articles(text),
        "annex_number": extract_annex_number(text),
    }


def extract_annex_number(text: str) -> str | None:
    """
    Extract an annex identifier such as:

        ANNEXE 1      -> "1"
        ANNEXE I      -> "I"
        ANNEXE A      -> "A"
        Annexe n°2    -> "2"

    Returns None if no annex marker is found, or if "annexe" appears
    without an attached number/letter (e.g. bare "ANNEXE" — the caller
    should treat this as unresolved rather than guessing).
    """

    match = ANNEX_PATTERN.search(text)

    if not match or not match.group(1):
        return None

    return match.group(1).upper()