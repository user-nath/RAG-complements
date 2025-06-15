# Assistant Q&A sur Documents avec RAG – Streamlit App

## Description

Cette application web interactive permet de poser des questions en langage naturel sur des documents PDF chargés par l’utilisateur.  
Elle utilise le principe de **RAG (Retrieval-Augmented Generation)**, combinant un moteur de recherche vectorielle (retrieval) et un modèle de langage (generation) pour fournir des réponses précises basées sur les contenus des documents.

Deux frameworks d’indexation sont proposés :  
- **Langchain** (avec vecteur en mémoire, Azure OpenAI embeddings et LLM)  
- **LlamaIndex** (avec embedding et LLM Azure OpenAI)

L’application permet également de collecter des feedbacks utilisateur sur la qualité des réponses, enregistrés dans une base SQLite.

---

## Fonctionnalités principales

- Import et indexation de documents PDF  
- Recherche de passages pertinents dans les documents via embeddings  
- Génération de réponses concises par un LLM Azure OpenAI  
- Support multilingue (français, anglais, espagnol, allemand)  
- Choix dynamique du backend RAG (Langchain ou LlamaIndex)  
- Collecte et sauvegarde des feedbacks utilisateurs  
- Suppression de documents indexés (pour Langchain uniquement)

---

## Prérequis

- Python 3.8+  
- Azure OpenAI account et déploiements configurés (embeddings + chat completions)  
- Modules Python :  
  - streamlit  
  - sqlite3 (standard)  
  - pandas  
  - langchain, llama-index, langchain_community (selon besoins)  
  - autres dépendances Azure OpenAI et loaders PyMuPDF

---

## Installation

1. Cloner ce dépôt ou copier les fichiers `app.py`, `langchain.py`, `llamaindex.py` dans un même dossier.  
2. Installer les dépendances Python :

```bash
pip install streamlit pandas langchain llama-index langchain_community pymupdf

---
## Limitations & améliorations possibles
- La suppression des documents n’est pas encore implémentée pour LlamaIndex.

- La recherche utilise un stockage en mémoire (Langchain) ou un vecteur simple (LlamaIndex) ; passage à une base vectorielle plus robuste possible (FAISS, Pinecone…).

- Ajout d’authentification et gestion multi-utilisateurs.

- Interface améliorée pour la gestion des documents et historique des questions/réponses.

- Support d’autres formats de documents (Word, txt, HTML…).

