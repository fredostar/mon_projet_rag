"""
Module contenant les nœuds du graphe LangGraph pour le workflow RAG.

Ce module définit les fonctions principales :
- Initialisation du retriever (ChromaDB + MistralAI Embeddings).
- Recherche de contexte dans les documents.
- Génération et amélioration des réponses avec le LLM.
"""

import os
import warnings
from typing import Dict, Any, Optional
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.exceptions import ModelRateLimitError, ModelAuthenticationError
from .state import RagState
from configs.settings import settings


# --- Exceptions personnalisées ---
class RagError(Exception):
    """Exception de base pour les erreurs liées au workflow RAG."""
    pass


class RetrieverError(RagError):
    """Erreur lors de l'initialisation ou de l'utilisation du retriever."""
    pass


class LLMError(RagError):
    """Erreur lors de l'appel au LLM (MistralAI)."""
    pass


class DocumentError(RagError):
    """Erreur liée au chargement ou au traitement des documents."""
    pass

# Initialisation du LLM MistralAI avec les paramètres de configuration
llm = ChatMistralAI(
    api_key=settings.MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS,
)

# Variable globale pour le retriever (initialisation paresseuse)
_retriever = None


def init_retriever(pdf_path: str = None) -> Any:
    """
    Initialise le retriever pour le RAG, en réutilisant l'index Chroma existant s'il existe.

    Args:
        pdf_path (str, optional): Chemin vers le fichier PDF à indexer. 
            Si None, utilise le chemin défini dans `settings.PDF_PATH`.

    Returns:
        Any: Un objet `Retriever` configuré pour rechercher dans les documents indexés.

    Raises:
        DocumentError: Si le fichier PDF est introuvable ou illisible.
        RetrieverError: Si l'initialisation de ChromaDB ou des embeddings échoue.

    Notes:
        - Si le dossier `CHROMA_PERSIST_DIR` existe et contient des données, 
          le retriever est initialisé à partir de l'index existant.
        - Sinon, le PDF est chargé, découpé en chunks, et indexé dans ChromaDB.
        - Les chunks sont générés avec un `RecursiveCharacterTextSplitter` 
          (taille: 1000 caractères, overlap: 200).
    """
    pdf_path = pdf_path or settings.PDF_PATH
    persist_dir = settings.CHROMA_PERSIST_DIR

    try:
        # Vérifier que le PDF existe si on doit l'indexer
        if not (os.path.isdir(persist_dir) and os.listdir(persist_dir)):
            if not os.path.isfile(pdf_path):
                raise DocumentError(f"Fichier PDF introuvable : {pdf_path}")

        # Initialiser les embeddings MistralAI
        embedding = MistralAIEmbeddings(api_key=settings.MISTRAL_API_KEY)

        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
        else:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
            except Exception as e:
                raise DocumentError(f"Échec du chargement du PDF {pdf_path}: {str(e)}") from e

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                persist_directory=persist_dir
            )
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        raise RetrieverError(f"Échec de l'initialisation du retriever: {str(e)}") from e


def get_retriever() -> Any:
    """
    Retourne le retriever, initialisé paresseusement au premier appel.

    Returns:
        Any: L'objet `Retriever` initialisé (ou réutilisé s'il existe déjà).

    Raises:
        RetrieverError: Si l'initialisation du retriever échoue.

    Notes:
        - Utilise un pattern **lazy loading** pour éviter de charger ChromaDB 
          tant qu'il n'est pas nécessaire.
        - Le retriever est stocké dans une variable globale `_retriever` 
          pour être réutilisé entre les appels.
        - En cas d'erreur, `_retriever` reste à None pour permettre une nouvelle tentative.
    """
    global _retriever
    if _retriever is None:
        try:
            _retriever = init_retriever()
        except RetrieverError as e:
            warnings.warn(f"⚠️ Impossible d'initialiser le retriever: {str(e)}")
            raise
    return _retriever


def rechercher_contexte(state: RagState) -> Dict[str, Any]:
    """
    Récupère le contexte depuis le Vector Store (ChromaDB) en fonction de la question.

    Args:
        state (RagState): État courant du workflow, contenant la question de l'utilisateur.

    Returns:
        Dict[str, Any]: Un dictionnaire avec :
            - `contexte` (str): Le contexte formaté (documents concaténés).
            - `documents` (List[str]): Liste des contenus des documents retrouvés.
            - Si une erreur survient, retourne un contexte vide avec un message d'erreur.

    Notes:
        - Utilise le retriever pour trouver les 3 documents les plus pertinents 
          (configuré dans `init_retriever`).
        - Le contexte est formaté avec des séparateurs pour une meilleure lisibilité.
        - En cas d'erreur, retourne un contexte vide pour éviter de bloquer le workflow.
    """
    try:
        documents = get_retriever().invoke(state["question"])
        contexte = "\n\n---\n\n".join(
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        )
        return {
            "contexte": contexte,
            "documents": [doc.page_content for doc in documents],
            "erreur": False,
        }
    except RetrieverError as e:
        warnings.warn(f"⚠️ Échec de la recherche de contexte: {str(e)}")
        return {
            "contexte": f"Impossible de récupérer le contexte: {str(e)}",
            "documents": [],
            "erreur": True,
        }
    except Exception as e:
        warnings.warn(f"⚠️ Erreur inattendue lors de la recherche de contexte: {str(e)}")
        return {
            "contexte": f"Erreur inattendue: {str(e)}",
            "documents": [],
            "erreur": True,
        }


def generer_reponse(state: RagState) -> Dict[str, Any]:
    """
    Génère une réponse avec le LLM en utilisant le contexte récupéré.

    Args:
        state (RagState): État courant du workflow, contenant la question et le contexte.

    Returns:
        Dict[str, Any]: Un dictionnaire avec :
            - `reponse` (str): La réponse générée par le LLM (ou un message d'erreur).
            - `nb_tentatives` (int): Nombre de tentatives incrémenté de 1.
            - `historique` (List[str]): Historique des réponses générées.

    Notes:
        - Le prompt demande explicitement au LLM de ne répondre **qu'à partir du contexte**.
        - Si le contexte ne contient pas la réponse, le LLM doit l'indiquer clairement.
        - L'historique est mis à jour pour tracer toutes les tentatives.
        - En cas d'erreur, retourne une réponse avec un message d'erreur.
    """
    if state.get("erreur"):
        # Le contexte n'a pas pu être récupéré : inutile d'appeler le LLM.
        return {
            "reponse": f"Impossible de générer une réponse : {state['contexte']}",
            "nb_tentatives": state.get("nb_tentatives", 0) + 1,
            "historique": state.get("historique", []) + [f"Tentative {state.get('nb_tentatives', 0) + 1}: Échec (contexte indisponible)"],
            "erreur": True,
        }
    try:
        prompt = (
            "Tu es un assistant expert en RAG. Réponds à la question suivante "
            "en t'appuyant UNIQUEMENT sur le contexte fourni. "
            "Si le contexte ne contient pas la réponse, dis-le clairement.\n\n"
            f"Contexte:\n{state['contexte']}\n\n"
            f"Question: {state['question']}"
        )
        reponse = llm.invoke(prompt)
        return {
            "reponse": reponse.content,
            "nb_tentatives": state.get("nb_tentatives", 0) + 1,
            "historique": state.get("historique", []) + [f"Tentative {state.get('nb_tentatives', 0) + 1}: {reponse.content}"],
            "erreur": False,
        }
    except (ModelRateLimitError, ModelAuthenticationError) as e:
        warnings.warn(f"⚠️ Erreur API MistralAI: {str(e)}")
        return {
            "reponse": f"Problème avec l'API MistralAI: {str(e)}. Vérifiez votre clé API ou attendez avant de réessayer.",
            "nb_tentatives": state.get("nb_tentatives", 0) + 1,
            "historique": state.get("historique", []) + [f"Tentative {state.get('nb_tentatives', 0) + 1}: Échec (erreur API)"],
            "erreur": True,
        }
    except Exception as e:
        warnings.warn(f"⚠️ Erreur inattendue lors de la génération de la réponse: {str(e)}")
        return {
            "reponse": f"Impossible de générer une réponse: {str(e)}",
            "nb_tentatives": state.get("nb_tentatives", 0) + 1,
            "historique": state.get("historique", []) + [f"Tentative {state.get('nb_tentatives', 0) + 1}: Échec (erreur inattendue)"],
            "erreur": True,
        }


def ameliorer_reponse(state: RagState) -> Dict[str, Any]:
    """
    Améliore une réponse existante si elle est trop courte ou non satisfaisante.

    Args:
        state (RagState): État courant du workflow, contenant la question, 
            le contexte et la réponse actuelle à améliorer.

    Returns:
        Dict[str, Any]: Un dictionnaire avec :
            - `reponse` (str): La réponse améliorée par le LLM (ou un message d'erreur).
            - `nb_tentatives` (int): Nombre de tentatives incrémenté de 1.
            - `historique` (List[str]): Historique mis à jour avec l'amélioration.

    Notes:
        - Le prompt demande au LLM de rendre la réponse plus détaillée et précise,
          tout en restant fidèle au contexte.
        - Utilisé dans le graphe LangGraph pour les tentatives d'amélioration 
          (voir `graph.py`).
        - En cas d'erreur, retourne une réponse avec un message d'erreur.
    """
    try:
        prompt = (
            "Améliore la réponse suivante en la rendant plus détaillée et précise, "
            "tout en restant fidèle au contexte. Ajoute des exemples si possible.\n\n"
            f"Contexte:\n{state['contexte']}\n\n"
            f"Question: {state['question']}\n"
            f"Réponse actuelle: {state['reponse']}"
        )
        reponse = llm.invoke(prompt)
        return {
            "reponse": reponse.content,
            "nb_tentatives": state["nb_tentatives"] + 1,
            "historique": state.get("historique", []) + [f"Amélioration {state['nb_tentatives'] + 1}: {reponse.content}"],
            "erreur": False,
        }
    except (ModelRateLimitError, ModelAuthenticationError) as e:
        warnings.warn(f"⚠️ Erreur API MistralAI lors de l'amélioration: {str(e)}")
        return {
            "reponse": f"Problème avec l'API MistralAI: {str(e)}. Impossible d'améliorer la réponse.",
            "nb_tentatives": state["nb_tentatives"] + 1,
            "historique": state.get("historique", []) + [f"Amélioration {state['nb_tentatives'] + 1}: Échec (erreur API)"],
            "erreur": True,
        }
    except Exception as e:
        warnings.warn(f"⚠️ Erreur inattendue lors de l'amélioration de la réponse: {str(e)}")
        return {
            "reponse": f"Impossible d'améliorer la réponse: {str(e)}",
            "nb_tentatives": state["nb_tentatives"] + 1,
            "historique": state.get("historique", []) + [f"Amélioration {state['nb_tentatives'] + 1}: Échec (erreur inattendue)"],
            "erreur": True,
        }