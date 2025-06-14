import os
import tempfile

import sqlite3
import streamlit as st
import pandas as pd

import rag.langchain as langchain
import rag.llamaindex as llamaindex

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []
if "framework" not in st.session_state:
    st.session_state["framework"] = "langchain"
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 5

# Choix du framework
st.sidebar.title("Paramètres")
selected_framework = st.sidebar.radio("Framework d'indexation :", ["langchain", "llamaindex"])
if selected_framework != st.session_state["framework"]:
    st.session_state["framework"] = selected_framework
    st.session_state["stored_files"] = []
# Associer dynamiquement le module
framework_module = langchain if selected_framework == "langchain" else llamaindex

# Choix de la langue
selected_language = st.sidebar.selectbox(
    "Langue de réponse :",
    ["Français", "Anglais", "Espagnol", "Allemand"]
)

# Sélection du nombre de documents similaires à récupérer
top_k = st.sidebar.slider("Nombre de documents à récupérer (k)", min_value=1, max_value=20, value=st.session_state["top_k"])
st.session_state["top_k"] = top_k
    

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
        answer = framework_module.answer_question(
            question, langue=selected_language, top_k=top_k
        )
        st.markdown("### Réponse :")
        st.write(answer)
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse : {e}")

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
            

#----Feedback utilisateur----

# --- Fonction pour initialiser la base SQLite ---
def init_db():
    conn = sqlite3.connect('feedbacks.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# --- Fonction pour insérer un feedback ---
def insert_feedback(question, answer, rating):
    conn = sqlite3.connect('feedbacks.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedbacks (question, answer, rating)
        VALUES (?, ?, ?)
    ''', (question, answer, rating))
    conn.commit()
    conn.close()

# --- le main / logique Streamlit ---
def main():
    init_db()  # Initialisation DB (safe à appeler plusieurs fois)

    st.title("Assistant de questions-réponses sur documents")

    # Supposons que tu as déjà ta logique d'upload, question, etc.
    question = st.text_input("Posez votre question :")
    # Simulation d'une réponse (tu remplaces par ton call à answer_question)
    # ou appelle ta fonction answer_question(question, langue, top_k)
    
    if st.button("Poser la question") and question:
        answer = "Réponse simulée pour l'exemple"  # Remplace par ta vraie réponse
        st.write("### Réponse :")
        st.write(answer)

        rating = st.radio("Quelle est la qualité de la réponse ?", 
                          options=["Très bien", "Bien", "Moyen", "Mauvais"])
        if rating:
            insert_feedback(question, answer, rating)
            st.success("Merci pour votre feedback !")
            print(f"Feedback enregistré : Question='{question}', Note='{rating}'")

if __name__ == "__main__":
    main()
