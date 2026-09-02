from typing import TypedDict, List, Optional

class RagState(TypedDict):
    """État partagé entre les nœuds du graphe LangGraph."""
    question: str
    contexte: str
    reponse: str
    nb_tentatives: int
    historique: List[str]  # Pour le débogage
    documents: Optional[List[str]]  # Liste des documents utilisés