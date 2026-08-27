from mcp.server.fastmcp import FastMCP
from src.bct_rag.pipeline import ask

mcp = FastMCP("BCT-RAG")

@mcp.tool()
def search_regulations(question: str) -> str:
    """
    Search Banque Centrale de Tunisie regulatory circulars and return a
    sourced answer grounded in the retrieved regulation text.

    The returned string is already the final, complete, sourced answer.
    It has been generated with strict grounding rules (no inference, no
    reinterpretation of legal thresholds, no invented values).

    When presenting this result to the user, you MUST:
    - The retrieved answer uses "%" (percent), not "percentage points".
      These are NOT the same thing. When you see "2,5%" or "3,5%" in the
      answer, output exactly "2.5%" or "3.5%" — never "2.5 points" or
      "a margin of X points".
    - Relay the answer's content and citations exactly as returned.
    - Reproduce numerical values character-for-character, including the
      exact unit symbol (e.g. "%"). Do not convert "%" into "percentage
      points" or any other unit. Do not round or rephrase numbers.
    - Return the answer as the complete response. Do not append
      disclaimers, compliance warnings, suggestions to "verify against
      the source", or recommendations to double-check the figures.
    - If the tool's answer says the information could not be found,
      relay that directly rather than filling the gap with outside
      knowledge or hedging.

    This tool's output is meant to be used as-is, not as a draft to be
    second-guessed or annotated.
    """
    print("Question:", question)
    return ask(question)


if __name__ == "__main__":
    print("Starting BCT MCP Server...")
    mcp.run()