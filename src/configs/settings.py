"""
Module de configuration centralisée pour le projet RAG.

Ce module charge les variables d'environnement depuis un fichier `.env`
et les expose via une classe `Settings`. Toutes les configurations sont
accessibles via l'instance globale `settings`.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Racine du projet, indépendante du répertoire de travail courant
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    """
    Classe contenant toutes les configurations du projet.

    Les valeurs sont chargées depuis les variables d'environnement,
    avec des valeurs par défaut si elles ne sont pas définies.

    Attributes:
        MISTRAL_API_KEY (str): Clé API pour accéder à MistralAI.
            Valeur par défaut : "ta_clé_api_mistral" (à remplacer).
        PDF_PATH (str): Chemin absolu vers le fichier PDF à indexer.
            Valeur par défaut : "data/documents.pdf" (relatif à BASE_DIR).
        CHROMA_PERSIST_DIR (str): Dossier de persistance pour ChromaDB.
            Valeur par défaut : "./chroma_db" (relatif à BASE_DIR).
        MAX_TOKENS (int): Nombre maximum de tokens pour les réponses du LLM.
            Valeur par défaut : 1000.
        TEMPERATURE (float): Paramètre de température pour le LLM (0.0 à 1.0).
            Valeur par défaut : 0.3.
    """
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "ta_clé_api_mistral")
    PDF_PATH = str(BASE_DIR / os.getenv("PDF_PATH", "data/documents.pdf"))
    CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))


# Instance globale des paramètres (à importer via `from configs.settings import settings`)
settings = Settings()

# Vérification de sécurité : avertir si la clé API est la valeur par défaut
if settings.MISTRAL_API_KEY == "ta_clé_api_mistral":
    import warnings
    warnings.warn(
        "⚠️  ATTENTION : La clé API MistralAI est toujours la valeur par défaut. "
        "Veuillez la configurer dans votre fichier .env pour éviter des erreurs. "
        "Exemple : MISTRAL_API_KEY=votre_clé_api_ici",
        UserWarning,
        stacklevel=2
    )