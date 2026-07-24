from src.bct_rag.pipeline import ask

while True:

    q = input("> ")

    print()

    print(ask(q))

    print()