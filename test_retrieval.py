from bct_rag.vectorstore.chroma_store import ChromaStore

store = ChromaStore()

query = "conditions de financement de l'importation"

results = store.query(query, k=5)

for i in range(len(results["documents"][0])):
    print("=" * 80)
    print(results["documents"][0][i][:400])