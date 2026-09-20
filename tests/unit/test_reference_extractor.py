from src.bct_rag.graph.reference_extractor import (
    extract_laws,
    extract_circulars,
    extract_articles,
    extract_references,
    extract_annex_number,
)


def test_extract_laws():

    text = """
    Vu la loi n°2016-35 du 25 avril 2016,
    Vu la loi n°2016-48 du 11 juillet 2016.
    """

    result = extract_laws(text)

    assert "2016-35" in result
    assert "2016-48" in result


def test_extract_old_law():

    text = """
    Vu la loi n°94-14 du 31 janvier 1994.
    """

    result = extract_laws(text)

    assert "1994-14" in result


def test_extract_circulars():

    text = """
    Vu la circulaire n°91-22 du 17 décembre 1991.
    """

    result = extract_circulars(text)

    assert "1991-22" in result


def test_extract_articles():

    text = """
    notamment ses articles 63 et 66.
    """

    result = extract_articles(text)

    assert 63 in result
    assert 66 in result


def test_extract_all_references():

    text = """
    Vu la loi n°2016-35,
    Vu la loi n°2016-48,
    Vu la circulaire n°91-22,
    notamment son article 36.
    """

    result = extract_references(text)

    assert "2016-35" in result["laws"]
    assert "2016-48" in result["laws"]
    assert "1991-22" in result["circulars"]
    assert 36 in result["articles"]

def test_extract_annex_number():

    assert extract_annex_number("ANNEXE 1") == "1"
    assert extract_annex_number("ANNEXE I") == "I"
    assert extract_annex_number("ANNEXE A") == "A"
    assert extract_annex_number("Annexe n°2") == "2"


def test_extract_annex_number_missing():

    assert extract_annex_number("Voir en annexe.") is None
    assert extract_annex_number("Pas de référence ici.") is None