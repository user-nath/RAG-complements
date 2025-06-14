import os
import tempfile

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
