"""
Regression tests for the generation stage.

Purpose
-------
Protect the generation stage against:
    1. Numerical threshold corruption.
    2. Loss of the "respectivement" relationship.
    3. Blurring of the two dividend-distribution scenarios.
    4. Unwanted LLM editorializing.
    5. Incorrect fallback behavior when no context is available.

The fixture below is based directly on Article 1 of Circular 2026-3.

Important:
    The source PDF may contain OCR/layout formatting such as:

        2, 5 %
        3, 5 %
        35 %

while the LLM may generate:

        2,5%
        3,5%
        35%

The normalization function intentionally treats these as equivalent.
It does NOT alter the legal meaning or relationship between the values.
"""

import re

import pytest

from src.bct_rag.llm.generator import generate


# ---------------------------------------------------------------------------
# GOLDEN FIXTURE
# ---------------------------------------------------------------------------
#
# This text must come from the actual regulation/PDF, NOT from an LLM answer.
#

FIXTURES = [
    {
        "id": "circular_2026_3_article_1_dividends",

        "question": (
            "According to Circular 2026-3, what exactly are the numerical "
            "conditions concerning the solvency and Tier 1 ratios for the "
            "two dividend-distribution scenarios?"
        ),

        "documents": [
            {
                "payload": {
                    "circular_ref": "Circular 2026-3",
                    "article_number": "1",

                    "text": (
                        "Article premier - La distribution des dividendes au titre "
                        "de l’exercice 2025, par les banques et les établissements "
                        "financiers, s’effectue dans les conditions suivantes: "

                        "- Dans la limite de 35 % du bénéfice de l’exercice 2025 "
                        "pour les banques et les établissements financiers "
                        "présentant des ratios de solvabilité et Tier 1 arrêtés "
                        "à fin 2025, après déduction des dividendes à verser, "
                        "qui dépassent les niveaux minimums réglementaires "
                        "de 2, 5 % au moins; "

                        "- Sans limite et après accord préalable de la Banque "
                        "Centrale de Tunisie, pour les banques et les "
                        "établissements financiers présentant des ratios de "
                        "solvabilité et Tier 1 arrêtés à fin 2025, après "
                        "déduction des dividendes à verser, qui dépassent "
                        "les niveaux minimums réglementaires respectivement "
                        "de 2, 5 % et 3, 5 % au moins."
                    ),
                }
            }
        ],
    }
]


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Normalize superficial formatting differences without changing meaning.

    Examples
    --------
    "2, 5 %" -> "2,5%"
    "2,5 %"  -> "2,5%"
    "35 %"   -> "35%"
    "  2,5% " -> "2,5%"
    """

    text = text.lower()

    # Normalize decimal comma spacing:
    #
    # 2, 5 -> 2,5
    # 3, 5 -> 3,5
    #
    text = re.sub(
        r"(\d),\s+(\d)",
        r"\1,\2",
        text,
    )

    # Remove whitespace immediately before %
    #
    # "2,5 %" -> "2,5%"
    # "35 %"  -> "35%"
    #
    text = re.sub(
        r"\s+%",
        "%",
        text,
    )

    # Collapse remaining whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# TEST 1
# NUMERICAL FIDELITY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_preserves_numerical_thresholds(fixture):
    """
    The generated answer must preserve all numerical values explicitly
    present in Article 1:

        35%
        2,5%
        3,5%

    This test protects against the original bug where the LLM blurred
    the 2.5% / 3.5% distinction.
    """

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    normalized = _normalize(answer)

    expected_values = [
        "2,5%",
        "3,5%",
        "35%",
    ]

    for expected in expected_values:

        assert _normalize(expected) in normalized, (
            f"\nExpected numerical value '{expected}' "
            f"was missing from the generated answer.\n\n"
            f"Generated answer:\n"
            f"{answer}\n\n"
            f"Normalized answer:\n"
            f"{normalized}"
        )


# ---------------------------------------------------------------------------
# TEST 2
# VERIFY THE TWO SCENARIOS ARE BOTH PRESENT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_preserves_two_distribution_scenarios(fixture):
    """
    Article 1 contains two distinct distribution regimes:

    Scenario 1:
        Limited distribution
        -> 35% of 2025 profit

    Scenario 2:
        Unlimited distribution
        -> prior BCT approval

    The answer must not collapse them into one condition.
    """

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    normalized = _normalize(answer)

    # Scenario 1
    assert "35%" in normalized, (
        "The capped 35% distribution scenario is missing."
    )

    assert (
        "dans la limite" in normalized
        or "35%" in normalized
        or "limited" in normalized
        or "capped" in normalized
    ), (
        "The answer does not clearly preserve the capped distribution "
        "scenario."
    )

    # Scenario 2
    assert (
        "sans limite" in normalized
        or "unlimited" in normalized
        or "no limit" in normalized
    ), (
        "The unlimited distribution scenario is missing."
    )

    assert (
        "accord préalable" in normalized
        or "prior approval" in normalized
        or "bct approval" in normalized
    ), (
        "The prior BCT approval condition for unlimited distribution "
        "is missing."
    )


# ---------------------------------------------------------------------------
# TEST 3
# VERIFY "RESPECTIVEMENT" PAIRING
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_preserves_respectively_pairing(fixture):
    """
    The second scenario explicitly says:

        respectivement de 2,5 % et 3,5 %

    The generation stage must preserve the distinction rather than
    rewriting it as a generic "2.5% / 3.5%" statement without the
    relationship between the two ratios.
    """

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    normalized = _normalize(answer)

    assert "2,5%" in normalized, (
        "2,5% is missing from the answer."
    )

    assert "3,5%" in normalized, (
        "3,5% is missing from the answer."
    )

    assert "respectivement" in normalized, (
        "The word 'respectivement' is missing. "
        "The generated answer may have lost the explicit "
        "relationship between solvency and Tier 1."
    )


# ---------------------------------------------------------------------------
# TEST 4
# VERIFY CITATIONS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_cites_circular_and_article(fixture):
    """
    The answer should identify Circular 2026-3 and Article 1.
    """

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    normalized = _normalize(answer)

    assert "2026-3" in normalized, (
        "Circular 2026-3 is missing from the answer."
    )

    assert (
        "article 1" in normalized
        or "article premier" in normalized
    ), (
        "Article 1 citation is missing from the answer."
    )


# ---------------------------------------------------------------------------
# TEST 5
# NO EDITORIALIZING
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_does_not_editorialize(fixture):
    """
    The generation stage must return the answer, not commentary about
    the quality of the retrieval.

    This protects against responses such as:

        "There seems to be a discrepancy..."
        "Worth double-checking..."
        "I would flag..."
        "The retrieval may be fragile..."
        "This appears contradictory..."

    Those belong to evaluation/debugging, not the production generation
    stage.
    """

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    normalized = _normalize(answer)

    banned_phrases = [
        "discrepancy",
        "worth double-checking",
        "i'd flag",
        "i would flag",
        "verify directly against",
        "retrieval fragility",
        "provisional",
        "seems contradictory",
        "appears contradictory",
        "ambiguity",
        "the tool returned",
        "the retrieval",
        "the model",
        "i recommend checking",
        "double check",
    ]

    for phrase in banned_phrases:

        assert phrase not in normalized, (
            f"\nGeneration contains unwanted editorial commentary:\n"
            f"'{phrase}'\n\n"
            f"Generated answer:\n"
            f"{answer}"
        )


# ---------------------------------------------------------------------------
# TEST 6
# NO DOCUMENTS -> EXACT FALLBACK
# ---------------------------------------------------------------------------

def test_no_documents_returns_fallback():

    answer = generate(
        "Any question",
        [],
    )

    assert answer == (
        "I could not find this information in the available regulations."
    )


# ---------------------------------------------------------------------------
# TEST 7
# PREVENT THE SPECIFIC 2.5% COLLAPSE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_does_not_collapse_3_5_into_2_5(fixture):
    """
    Specific regression test for the original failure.

    A bad answer might say:

        "Both scenarios require 2.5%."

    That would lose the explicit 3.5% threshold.

    The answer must therefore contain both values.
    """

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    normalized = _normalize(answer)

    assert "2,5%" in normalized
    assert "3,5%" in normalized

    # If both values occur, the important numerical distinction survived.
    assert normalized.count("2,5%") >= 1
    assert normalized.count("3,5%") >= 1


# ---------------------------------------------------------------------------
# TEST 8
# VERIFY ANSWER IS NOT EMPTY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f["id"],
)
def test_generation_returns_non_empty_answer(fixture):

    answer = generate(
        fixture["question"],
        fixture["documents"],
    )

    assert answer.strip(), (
        "Generation returned an empty answer."
    )