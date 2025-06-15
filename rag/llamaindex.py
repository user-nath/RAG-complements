import streamlit as st
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

config = st.secrets

llm = AzureOpenAI(
    model=config["chat_azure_deployment"],
    deployment_name=config["chat_azure_deployment"],
    api_key=config["chat_azure_api_key"],
    azure_endpoint=config["chat_azure_endpoint"],
    api_version=config["chat_azure_api_version"],
)

embedder = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name=config["embedding_azure_deployment"],
    api_key=config["embedding_azure_api_key"],
    azure_endpoint=config["embedding_azure_endpoint"],
    api_version=config["embedding_azure_api_version"],
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()


def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFReader()
    documents = loader.load(file_path)

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embedder.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    vector_store.add(nodes)


def delete_file_from_store(name: str) -> int:
    raise NotImplementedError('function not implemented for Llamaindex')


def inspect_vector_store(top_n: int = 10) -> list:
    raise NotImplementedError('function not implemented for Llamaindex')


def get_vector_store_info():
    raise NotImplementedError('function not implemented for Llamaindex')


def retrieve(question: str, top_k: int = 5):
    query_embedding = embedder.get_query_embedding(question)

    query_mode = "default"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=top_k, mode=query_mode
    )

    try:
        query_result = vector_store.query(vector_store_query)
    except Exception as e:
        print(f"Erreur lors de la requête au vector store : {e}")
        return []

    if not query_result or not query_result.nodes:
        print("Aucun résultat trouvé dans le vector store.")
        return []

    return query_result.nodes


def build_qa_messages(question: str, context: str, langue: str) -> list[str]:
    langue_prompt = {
        "Français": "Tu es un assistant IA qui répond toujours en français.",
        "Anglais": "You are an AI assistant who always replies in English.",
        "Espagnol": "Eres un asistente de IA que siempre responde en español.",
        "Allemand": "Du bist ein KI-Assistent, der immer auf Deutsch antwortet."
    }

    task_prompt = {
        "Français": f"""Utilise les extraits de contexte suivants pour répondre à la question.
        Si tu ne sais pas répondre, dis que tu ne sais pas.
        Utilise trois phrases maximum et reste concis.
        {context}""",
        "Anglais": f"""Use the following pieces of context to answer the question.
        If you don't know the answer, just say you don't know.
        Use three sentences maximum and keep the answer concise.
        {context}""",
        "Espagnol": f"""Usa los siguientes fragmentos de contexto para responder a la pregunta.
        Si no sabes la respuesta, simplemente dilo.
        Usa un máximo de tres frases y sé conciso.
        {context}""",
        "Allemand": f"""Verwende die folgenden Kontexte, um die Frage zu beantworten.
        Wenn du die Antwort nicht weißt, gib dies an.
        Antworte in höchstens drei Sätzen und sei prägnant.
        {context}""",
    }

    messages = [
        ("system", langue_prompt.get(langue, langue_prompt["Français"])),
        ("user", question),
    ]
    return messages


def answer_question(question: str, langue: str, top_k: int = 5) -> str:
    docs = retrieve(question, top_k)
    if not docs:
        return "Désolé, je n'ai trouvé aucun document pertinent pour répondre à la question."

    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.id)
        print(doc.page_content)
        print("------")
    messages = build_qa_messages(question, docs_content, langue)
    response = llm.invoke(messages)
    return response.content
