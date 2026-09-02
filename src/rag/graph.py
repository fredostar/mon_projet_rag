from langgraph.graph import StateGraph, END
from typing import Literal
from .state import RagState
from .nodes import rechercher_contexte, generer_reponse, ameliorer_reponse


MAX_TENTATIVES = 3


def reponse_est_valide(state: RagState) -> Literal["ameliorer", "terminer"]:
    """Décide si la réponse doit être améliorée ou non."""
    reponse_insuffisante = (
        len(state["reponse"].split()) < 10
        or "je ne sais pas" in state["reponse"].lower()
    )
    if reponse_insuffisante and state["nb_tentatives"] < MAX_TENTATIVES:
        return "ameliorer"
    return "terminer"


def build_rag_graph() -> StateGraph:
    """Construit le graphe LangGraph pour le workflow RAG."""
    workflow = StateGraph(RagState)

    # Ajouter les nœuds
    workflow.add_node("recherche", rechercher_contexte)
    workflow.add_node("generation", generer_reponse)
    workflow.add_node("amelioration", ameliorer_reponse)

    # Définir le point d'entrée
    workflow.set_entry_point("recherche")

    # Ajouter les edges
    workflow.add_edge("recherche", "generation")

    # Routage conditionnel
    workflow.add_conditional_edges(
        "generation",
        reponse_est_valide,
        {"ameliorer": "amelioration", "terminer": END},
    )
    workflow.add_conditional_edges(
        "amelioration",
        reponse_est_valide,
        {"ameliorer": "amelioration", "terminer": END},
    )

    return workflow

# Compiler le graphe
app = build_rag_graph().compile()