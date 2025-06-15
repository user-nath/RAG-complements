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

-  **Suppression de documents non encore implémentée pour LlamaIndex.**
-  **Stockage en mémoire ou basique :** actuellement, Langchain utilise un vecteur en mémoire et LlamaIndex une forme simple d’index. Il serait pertinent d’ajouter une base vectorielle persistante comme **FAISS**, **ChromaDB** ou **Pinecone**.
-  **Pas de gestion multi-utilisateur** : l’application est mono-utilisateur et ne propose pas d’authentification.
- **Pas d’historique ni de statistiques de feedbacks** : les interactions ne sont pas visualisables par l'utilisateur pour le moment.
- **Support limité aux PDF** : les formats **Word**, **TXT**, ou **HTML** pourraient être pris en charge dans une version future.
-  **Interface à améliorer** : ajout possible de la **traduction automatique**, d’un **mode sombre**, d’un meilleur affichage responsive, etc.
- **Tests et CI/CD manquants** : ajout de **tests automatisés**, d’un pipeline de déploiement continu (CI/CD), et d’une gestion des erreurs plus robuste.




