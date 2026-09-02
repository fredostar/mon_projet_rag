"""
Module définissant le graphe LangGraph pour le workflow RAG.

Ce module contient :
- La logique de routage conditionnel (amélioration ou terminaison).
- La construction du graphe avec ses nœuds et ses edges.
- La compilation du graphe en une application exécutable.
"""

import warnings
from langgraph.graph import StateGraph, END
from typing import Literal
from .state import RagState
from .nodes import rechercher_contexte, generer_reponse, ameliorer_reponse


# Nombre maximum de tentatives pour améliorer une réponse
MAX_TENTATIVES = 3


def reponse_est_valide(state: RagState) -> Literal["ameliorer", "terminer"]:
    """
    Détermine si la réponse doit être améliorée ou si le workflow peut se terminer.

    Args:
        state (RagState): État courant du workflow, contenant la réponse et 
            le nombre de tentatives.

    Returns:
        Literal["ameliorer", "terminer"]: 
            - "ameliorer" : Si la réponse est insuffisante et que le nombre 
              de tentatives n'a pas dépassé `MAX_TENTATIVES`.
            - "terminer" : Sinon (y compris en cas d'erreur irréparable).

    Notes:
        Une réponse est considérée comme insuffisante si :
        - Elle contient moins de 10 mots.
        - Elle contient l'expression "je ne sais pas" (insensible à la casse).
        - Elle contient le mot "[ERREUR]" (indiquant une erreur dans le workflow).

        Si une erreur irréparable est détectée (ex: problème API persistant), 
        le workflow se termine pour éviter une boucle infinie.
    """
    # Vérifier si la réponse contient une erreur irréparable
    if "[ERREUR]" in state["reponse"]:
        warnings.warn(f"⚠️ Erreur détectée dans la réponse: {state['reponse']}")
        return "terminer"

    reponse_insuffisante = (
        len(state["reponse"].split()) < 10
        or "je ne sais pas" in state["reponse"].lower()
    )
    if reponse_insuffisante and state["nb_tentatives"] < MAX_TENTATIVES:
        return "ameliorer"
    return "terminer"


def build_rag_graph() -> StateGraph:
    """
    Construit le graphe LangGraph pour le workflow RAG.

    Returns:
        StateGraph: Un graphe configuré avec les nœuds et les edges pour le workflow RAG.

    Notes:
        Le graphe suit ce flux :
        1. **recherche** : Récupère le contexte pertinent depuis ChromaDB.
        2. **generation** : Génère une réponse initiale avec le LLM.
        3. **routage conditionnel** : 
            - Si la réponse est valide → **terminer** (fin du workflow).
            - Sinon → **amelioration** (nouvelle tentative).
        4. **amelioration** : Améliore la réponse existante.
        5. **routage conditionnel** : 
            - Si la réponse est valide → **terminer**.
            - Sinon → **amelioration** (jusqu'à `MAX_TENTATIVES`).

    Exemple de graphe :
        recherche → generation → [amelioration] → END
    """
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