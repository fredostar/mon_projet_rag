import os
from typing import Dict, Any
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .state import RagState
from configs.settings import settings

llm = ChatMistralAI(
    api_key=settings.MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS,
)

_retriever = None


def init_retriever(pdf_path: str = None) -> Any:
    """Initialise le retriever pour le RAG, en réutilisant l'index Chroma existant s'il existe."""
    pdf_path = pdf_path or settings.PDF_PATH
    persist_dir = settings.CHROMA_PERSIST_DIR
    embedding = MistralAIEmbeddings(api_key=settings.MISTRAL_API_KEY)

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    else:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
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


def get_retriever() -> Any:
    """Retourne le retriever, initialisé paresseusement au premier appel."""
    global _retriever
    if _retriever is None:
        _retriever = init_retriever()
    return _retriever


def rechercher_contexte(state: RagState) -> Dict[str, Any]:
    """Récupère le contexte depuis le Vector Store."""
    documents = get_retriever().invoke(state["question"])
    contexte = "\n\n---\n\n".join(
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )
    return {
        "contexte": contexte,
        "documents": [doc.page_content for doc in documents]
    }


def generer_reponse(state: RagState) -> Dict[str, Any]:
    """Génère une réponse avec le LLM."""
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
    }


def ameliorer_reponse(state: RagState) -> Dict[str, Any]:
    """Améliore la réponse si elle est trop courte."""
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
    }