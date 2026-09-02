import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Racine du projet, indépendante du répertoire de travail courant
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "ta_clé_api_mistral")
    PDF_PATH = str(BASE_DIR / os.getenv("PDF_PATH", "data/documents.pdf"))
    CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

settings = Settings()