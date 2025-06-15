# app.py

import streamlit as st
import tempfile
import os
import sqlite3
import pandas as pd

import rag.langchain as langchain
import rag.llamaindex as llamaindex

# --- Initialisation de la base SQLite ---
def init_db():
    conn = sqlite3.connect("feedbacks.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            rating TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# --- Enregistrement du feedback dans la base ---
def save_feedback(question, answer, rating):
    conn = sqlite3.connect("feedbacks.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedbacks (question, answer, rating) VALUES (?, ?, ?)",
              (question, answer, rating))
    conn.commit()
    conn.close()

# Initialisation de l'état de session
if "stored_files" not in st.session_state:
    st.session_state["stored_files"] = []
if "framework" not in st.session_state:
    st.session_state["framework"] = "langchain"
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "answer" not in st.session_state:
    st.session_state["answer"] = ""
if "langue" not in st.session_state:
    st.session_state["langue"] = "Français"

# Initialiser la base
init_db()

# --- Choix du framework ---
st.sidebar.title("Paramètres")
selected_framework = st.sidebar.radio("Framework d'indexation :", ["langchain", "llamaindex"])
selected_langue = st.sidebar.selectbox("Langue de réponse :", ["Français", "Anglais", "Espagnol", "Allemand"])

if selected_framework != st.session_state["framework"]:
    st.session_state["framework"] = selected_framework
    st.session_state["stored_files"] = []

st.session_state["langue"] = selected_langue

# Associer dynamiquement le module
framework_module = langchain if selected_framework == "langchain" else llamaindex

# --- Interface principale ---
st.title("Assistant de questions-réponses sur documents")

uploaded_files = st.file_uploader("Importer des documents PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state["stored_files"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            framework_module.store_pdf_file(tmp_path, f.name)
            st.session_state["stored_files"].append(f.name)
            os.remove(tmp_path)
    st.success("Documents indexés.")

st.markdown("---")

# --- Zone de question ---
question = st.text_input("Posez votre question :")

if st.button("Poser la question") and question:
    try:
        answer = framework_module.answer_question(question, st.session_state["langue"])
        st.session_state["question"] = question
        st.session_state["answer"] = answer
        st.markdown(f"**Réponse générée :**\n\n{answer}")
        st.write(str(answer))
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse : {e}")

# --- Affichage du feedback utilisateur ---
if st.session_state["answer"]:
    rating = st.radio("Comment évaluez-vous cette réponse ?", ["Très bien", "Bien", "Moyen", "Mauvais"], key="rating")
    if st.button("Envoyer le feedback"):
        save_feedback(st.session_state["question"], st.session_state["answer"], rating)
        st.success("Merci pour votre feedback !")
        st.session_state["question"] = ""
        st.session_state["answer"] = ""

# --- Suppression de document (optionnelle) ---
st.markdown("---")
if st.session_state["stored_files"]:
    doc_to_delete = st.selectbox("Supprimer un document :", st.session_state["stored_files"])
    if st.button("Supprimer"):
        try:
            framework_module.delete_file_from_store(doc_to_delete)
            st.session_state["stored_files"].remove(doc_to_delete)
            st.success("Document supprimé.")
        except NotImplementedError:
            st.warning("La suppression n'est pas disponible pour LlamaIndex.")
