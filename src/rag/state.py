"""
Module définissant l'état partagé du workflow RAG.

Ce module contient la classe `RagState`, qui représente les données
échangées entre les nœuds du graphe LangGraph.
"""

from typing import TypedDict, List, Optional


class RagState(TypedDict):
    """
    État partagé entre les nœuds du graphe LangGraph pour le workflow RAG.

    Attributes:
        question (str): La question posée par l'utilisateur.
        contexte (str): Le contexte extrait des documents (formaté).
        reponse (str): La réponse générée par le LLM.
        nb_tentatives (int): Nombre de tentatives de génération/amélioration.
        historique (List[str]): Liste des réponses générées à chaque tentative 
            (pour le débogage ou l'audit).
        documents (Optional[List[str]]): Liste des contenus des documents 
            utilisés pour générer le contexte.
    """
    question: str
    contexte: str
    reponse: str
    nb_tentatives: int
    historique: List[str]  # Pour le débogage
    documents: Optional[List[str]]  # Liste des documents utilisés