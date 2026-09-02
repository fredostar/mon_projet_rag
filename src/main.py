from rag.graph import app
from rag.state import RagState


def main():
    """Exécute le workflow RAG."""
    question = input("Posez votre question (ou 'quit' pour quitter) : ")

    while question.lower() != "quit":
        # État initial
        etat_initial: RagState = {
            "question": question,
            "contexte": "",
            "reponse": "",
            "nb_tentatives": 0,
            "historique": [],
            "documents": None,
        }

        # Exécuter le workflow
        resultat = app.invoke(etat_initial)

        # Afficher les résultats
        print("\n" + "=" * 50)
        print(f"🔍 QUESTION: {resultat['question']}")
        print("=" * 50)
        print("\n📄 CONTEXTE UTILISÉ:")
        print(resultat["contexte"])
        print("\n" + "=" * 50)
        print("💬 RÉPONSE FINALE:")
        print(resultat["reponse"])
        print("\n" + "=" * 50)
        print("📜 HISTORIQUE DES TENTATIVES:")
        for entry in resultat["historique"]:
            print(f"- {entry}")

        # Nouvelle question
        question = input("\nPosez une autre question (ou 'quit' pour quitter) : ")

if __name__ == "__main__":
    main()