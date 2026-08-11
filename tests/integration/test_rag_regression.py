from src.bct_rag.pipeline import ask


def test_existing_dividend_question_still_works():

    answer = ask(
        "What are the dividend distribution conditions?"
    )

    assert answer
    assert len(answer.strip()) > 0