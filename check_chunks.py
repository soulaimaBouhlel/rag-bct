import json

with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("chunks:", len(chunks))
print("max_tokens:", max(c["token_count"] for c in chunks))
print("min_tokens:", min(c["token_count"] for c in chunks))
print("avg_tokens:", sum(c["token_count"] for c in chunks) / len(chunks))
from collections import Counter

print(Counter(c["chunk_type"] for c in chunks))
for c in chunks[:5]:
    print("=" * 80)
    print(c["chunk_id"])
    print(c["chunk_type"])
    print(c["token_count"])
    print(c["text"][:500])