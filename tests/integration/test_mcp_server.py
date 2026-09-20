from src.bct_rag.mcp.mcp_server import search_regulations


def test_mcp_search_regulations_returns_answer():
    result = search_regulations(
        "Quelles sont les conditions de distribution des dividendes ?"
    )

    assert result is not None
    assert isinstance(result, str)
    assert len(result.strip()) > 0
def test_mcp_search_regulations_uses_graph_context():
    result = search_regulations(
        "Sur quelles lois se fonde la circulaire 2026-3 ?"
    )

    assert result is not None
    assert "2016-35" in result or "2016-48" in result