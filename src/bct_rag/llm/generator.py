"""
Generate answers from retrieved context using Ollama.
"""

import os
from ollama import Client
from src.bct_rag.config import OLLAMA_MODEL, OLLAMA_BASE

MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", OLLAMA_BASE)
client = Client(
    host=OLLAMA_HOST,
    timeout=300,
)

SYSTEM_PROMPT = """
You are an AI assistant specialized in Banque Centrale de Tunisie regulations.

STRICT RULES

- Answer ONLY from the provided context.
- Never use outside knowledge.
- Never invent information.
- Never explain your reasoning.
- Never think aloud.
- Never mention your analysis process.
- Return only the final answer.
- Keep the answer concise.
- Cite the circular number and article whenever possible.

If the answer is not contained in the context, reply exactly:

I could not find this information in the available regulations.
"""


def build_context(documents):
    parts = []

    for doc in documents:
        payload = doc["payload"]

        header = f"[{payload['circular_ref']}"

        if payload.get("article_number"):
            header += f" | Article {payload['article_number']}"

        header += "]"

        parts.append(
            f"{header}\n{payload['text']}"
        )

    return "\n\n".join(parts)


def generate(question, documents, graph_context: str | None = None):

    if not documents:
        return "I could not find this information in the available regulations."

    context = build_context(documents)
    print(f"Retrieved {len(documents)} chunks")
    print(f"Context length: {len(context)} characters")
    print(f"Graph context length: {len(graph_context) if graph_context else 0} characters")
    graph_section = ""
    if graph_context:
        graph_section = f"""

    Background knowledge graph (NOT a source, NOT for citation, NOT sufficient to answer alone)
    =============================================================================================

    This is background structural metadata about document relationships
    (which laws a circular cites, which articles belong to it, etc). It
    is NOT retrieved regulatory text and CANNOT be used, by itself, to
    answer the question.

    RULES:
    - If the Context section above already answers the question, you may
      use this metadata only to add a brief, natural-language mention of
      a related document — never quote these lines, never list them as
      Sources.
    - If the Context section above does NOT contain enough information to
      answer the question, you MUST reply with the standard fallback
      ("I could not find this information in the available regulations"),
      even if this background metadata seems related or suggestive. This
      metadata shows relationships, not absence or presence of specific
      facts — it can never substitute for the actual regulatory text.
    - Never treat a gap or pattern in this metadata as proof of anything
      the Context section doesn't explicitly state.

    {graph_context}
    """
        user_prompt = f"""
    Context
    =======

    {context}

    --- End of Context ---
    {graph_section}
    Question
    ========

    {question}

Instructions
============
Using ONLY the Context section above (never the background knowledge
graph alone) to determine your answer:

- Answer the user's question.
- Return ONLY the final answer.
- Do NOT explain your reasoning.
- Do NOT think step by step.
- Cite the circular number and article.
- If the answer is missing from the context, reply exactly:
Do not reinterpret or simplify legal conditions.

Preserve the meaning of the regulation.

If possible, quote numerical thresholds exactly as written in the context.
I could not find this information in the available regulations.

Format:

Answer:
...

Sources:
- Circular ...
- Article ...
"""

    # ---------- DEBUG ----------
    DEBUG = False

    if DEBUG:
        print("=" * 80)
        print("PROMPT SENT TO MODEL")
        print("=" * 80)
        print(user_prompt)
        print("=" * 80)
    # ---------------------------

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        think=False,
        stream=False,
        options={
            "temperature": 0.0,
            "top_p": 0.8,
            "num_predict": 1024,
            "repeat_penalty": 1.1,
        },
    )

    return response["message"]["content"].strip()