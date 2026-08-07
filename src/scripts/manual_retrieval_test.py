from src.bct_rag.retrieval.retriever import Retriever
from src.bct_rag.retrieval.retriever import print_results

retriever = Retriever()
def main():

    while True:

        question = input("\nQuestion: ")

        if question == "exit":
            break

        results = retriever.search(question)

        print_results(results)
if __name__ == "__main__":
    main()