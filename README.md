# mon_projet_rag

Un **workflow RAG (Retrieval-Augmented Generation)** construit avec **LangChain**, **LangGraph** et **MistralAI**. Ce projet permet de poser des questions en langage naturel sur des documents PDF et d'obtenir des réponses précises basées sur leur contenu.

---

## 📌 Fonctionnalités

- **Indexation de documents PDF** : Charge et découpe automatiquement les PDF en chunks pour une recherche optimale.
- **Recherche sémantique** : Utilise des **embeddings MistralAI** et **ChromaDB** pour trouver les passages pertinents.
- **Génération de réponses** : Le LLM (Mistral) répond en s'appuyant **uniquement** sur le contexte extrait.
- **Amélioration itérative** : Si une réponse est trop courte ou non satisfaisante, le système la retravaille jusqu'à 3 fois.
- **Historique des tentatives** : Affiche toutes les étapes de génération pour le débogage.

---

## 🛠 Prérequis

- **Python** >= 3.13 (recommandé : 3.13+)
- **Clé API MistralAI** 
- **Git** (pour cloner le projet)

---

## 🚀 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/ton-utilisateur/mon_projet_rag.git
cd mon_projet_rag
```

### 2. Créer un environnement virtuel (optionnel mais recommandé)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
Avec **uv** (recommandé) :
```bash
uv sync
```

Avec **pip** :
```bash
pip install -e .
```

---

## ⚙️ Configuration

### 1. Créer un fichier `.env` à la racine du projet
```bash
cp .env.example .env  # Si le fichier existe
```

### 2. Remplir les variables d'environnement
Édite `.env` avec tes informations :
```ini
# Clé API Mistral (obligatoire)
MISTRAL_API_KEY=ta_clé_api_ici

# Chemin vers ton PDF (par défaut : data/documents.pdf)
PDF_PATH=data/mon_document.pdf

# Dossier de persistance pour ChromaDB (par défaut : ./chroma_db)
CHROMA_PERSIST_DIR=./chroma_db

# Paramètres du LLM (optionnels)
MAX_TOKENS=1000
TEMPERATURE=0.3
```

> ⚠️ **Ne commite jamais ton `.env` dans git !** Il est déjà ignoré via `.gitignore`.

### 3. Placer ton PDF
Par défaut, le projet utilise `data/documents.pdf`. Tu peux :
- Remplacer ce fichier par le tien.
- Ou modifier `PDF_PATH` dans `.env` pour pointer vers ton document.

---

## 🎯 Utilisation

### Lancer le workflow RAG en mode interactif
```bash
python -m src.main
```

Exemple de session :
```
Posez votre question (ou 'quit' pour quitter) : Quels sont les avantages du RAG ?

==================================================
🔍 QUESTION: Quels sont les avantages du RAG ?
==================================================

📄 CONTEXTE UTILISÉ:
Document 1:
Le RAG (Retrieval-Augmented Generation) permet d'améliorer la précision des LLM en...

==================================================
💬 RÉPONSE FINALE:
Le RAG combine la puissance des modèles de langage avec une base de connaissances externe...

==================================================
📜 HISTORIQUE DES TENTATIVES:
- Tentative 1: Réponse initiale...
- Amélioration 2: Réponse détaillée...

Posez une autre question (ou 'quit' pour quitter) :
```

---

## 📂 Structure du projet

```
mon_projet_rag/
├── data/                  # Dossier des documents PDF
│   └── documents.pdf      # PDF par défaut
├── chroma_db/             # Base de données vectorielle (générée automatiquement)
├── src/
│   ├── main.py            # Point d'entrée CLI
│   ├── configs/
│   │   └── settings.py    # Configuration centralisée
│   └── rag/
│       ├── __init__.py
│       ├── state.py       # Définition de l'état du workflow
│       ├── nodes.py       # Nœuds du graphe (recherche, génération, amélioration)
│       └── graph.py       # Construction du graphe LangGraph
├── .env                   # Variables d'environnement (à ne pas commiter !)
├── pyproject.toml         # Dépendances et métadonnées
└── README.md              # Ce fichier
```

---

## 🔧 Personnalisation

### Changer le modèle Mistral
Modifie `src/rag/nodes.py` :
```python
llm = ChatMistralAI(
    api_key=settings.MISTRAL_API_KEY,
    model="mistral-medium-latest",  # ou "mistral-large-latest"
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS,
)
```

### Ajouter plusieurs PDFs
Modifie `src/rag/nodes.py` pour charger un dossier complet :
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("data/", glob="*.pdf")
docs = loader.load()
```

### Configurer le split de texte
Dans `src/rag/nodes.py`, ajuste les paramètres :
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Taille des chunks (en caractères)
    chunk_overlap=100,   # Overlap entre les chunks
    separators=["\n\n", "\n", " ", ""]
)
```

---

## 🧪 Tests

*(À venir : tests unitaires pour les nœuds et le graphe.)*

---

