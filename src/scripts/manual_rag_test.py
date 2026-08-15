from src.bct_rag.pipeline import ask
def main():


    while True:

        q = input("> ")

        print()

        print(ask(q))

        print()
if __name__ == "__main__":
    main()