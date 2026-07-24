from mcp.server.fastmcp import FastMCP

from src.bct_rag.pipeline import ask

mcp = FastMCP("BCT-RAG")


@mcp.tool()
def search_regulations(question: str) -> str:
    return ask(question)


if __name__ == "__main__":
    mcp.run()