import streamlit as st

from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI


CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200


config = st.secrets


embedder = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding_azure_endpoint"],
    azure_deployment=config["embedding_azure_deployment"],
    openai_api_version=config["embedding_azure_api_version"],
    api_key=config["embedding_azure_api_key"]
)

vector_store = InMemoryVectorStore(embedder)

llm = AzureChatOpenAI(
    azure_endpoint=config["chat_azure_endpoint"],
    azure_deployment=config["chat_azure_deployment"],
    openai_api_version=config["chat_azure_api_version"],
    api_key=config["chat_azure_api_key"]
)

def get_meta_doc(extract: str) -> str:
    """Generate a synthetic metadata description of the content.
    """
    messages = [
    (
        "system",
        "You are a librarian extracting metadata from documents.",
    ),
    (
        "user",
        """Extract from the content the following metadata.
        Answer 'unknown' if you cannot find or generate the information.
        Metadata list:
        - title
        - author
        - source
        - type of content (e.g. scientific paper, litterature, news, etc.)
        - language
        - themes as a list of keywords

        <content>
        {}
        </content>
        """.format(extract),
    ),]
    response = llm.invoke(messages)
    return response.content


def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    """Store a pdf file in the vector store.

    Args:
        file_path (str): file path to the PDF file
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    # TODO: make a constant of chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)
    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
            }
    if use_meta_doc:
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(page_content=get_meta_doc(extract),
                            metadata={
                                'document_name': doc_name,
                                'insert_date': datetime.now()
                                })
        all_splits.append(meta_doc)
    _ = vector_store.add_documents(documents=all_splits)
    return


def delete_file_from_store(name: str) -> int:
    ids_to_remove = []
    for (id, doc) in vector_store.store.items():
        if name == doc['metadata']['document_name']:
            ids_to_remove.append(id)
    vector_store.delete(ids_to_remove)
    #print('File deleted:', name)
    return len(ids_to_remove)


def inspect_vector_store(top_n: int=10) -> list:
    docs = []
    for index, (id, doc) in enumerate(vector_store.store.items()):
        if index < top_n:
            docs.append({
                'id': id,
                'document_name': doc['metadata']['document_name'],
                'insert_date': doc['metadata']['insert_date'],
                'text': doc['text']
                })
            # docs have keys 'id', 'vector', 'text', 'metadata'
            # print(f"{id} {doc['metadata']['document_name']}: {doc['text']}")
        else:
            break
    return docs


def get_vector_store_info():
    nb_docs = 0
    max_date, min_date = None, None
    documents = set()
    for (id, doc) in vector_store.store.items():
        nb_docs += 1
        if max_date is None or max_date < doc['metadata']['insert_date']:
            max_date = doc['metadata']['insert_date']
        if min_date is None or min_date > doc['metadata']['insert_date']:
            min_date = doc['metadata']['insert_date']
        documents.add(doc['metadata']['document_name'])
    return {
        'nb_chunks': nb_docs,
        'min_insert_date': min_date,
        'max_insert_date': max_date,
        'nb_documents': len(documents)
    }


def retrieve(question: str, top_k: int = 5):
    """Retrieve documents similar to a question.

    Args:
        question (str): text of the question

    Returns:
        list[TODO]: list of similar documents retrieved from the vector store
    """
    retrieved_docs = vector_store.similarity_search(question, top_k=top_k)
    return retrieved_docs


def build_qa_messages(question: str, context: str, langue: str) -> list[str]:
    # Messages système adaptés à la langue
    langue_prompt = {
        "Français": "Tu es un assistant IA qui répond toujours en français.",
        "Anglais": "You are an AI assistant who always replies in English.",
        "Espagnol": "Eres un asistente de IA que siempre responde en español.",
        "Allemand": "Du bist ein KI-Assistent, der immer auf Deutsch antwortet."
    }

    # Prompt d’instruction adapté à la langue
    task_prompt = {
        "Français": """Utilise les extraits de contexte suivants pour répondre à la question.
        Si tu ne sais pas répondre, dis que tu ne sais pas.
        Utilise trois phrases maximum et reste concis.
        {}""".format(context),
        "Anglais": """Use the following pieces of context to answer the question.
        If you don't know the answer, just say you don't know.
        Use three sentences maximum and keep the answer concise.
        {}""".format(context),
        "Espagnol": """Usa los siguientes fragmentos de contexto para responder a la pregunta.
        Si no sabes la respuesta, simplemente dilo.
        Usa un máximo de tres frases y sé conciso.
        {}""".format(context),
        "Allemand": """Verwende die folgenden Kontexte, um die Frage zu beantworten.
        Wenn du die Antwort nicht weißt, gib dies an.
        Antworte in höchstens drei Sätzen und sei prägnant.
        {}""".format(context),
    }

    messages = [
        ("system", langue_prompt.get(langue, langue_prompt["Français"])),
        ("system", task_prompt.get(langue, task_prompt["Français"])),
        ("user", question),
    ]
    return messages


def answer_question(question: str, langue:str, top_k:int = 5) -> str:
    """Answer a question by retrieving similar documents in the store.

    Args:
        question (str): text of the question

    Returns:
        str: text of the answer
    """
    inspect_vector_store()
    docs = retrieve(question, top_k=top_k)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.id)
        print(doc.page_content)
        print("------")
    messages = build_qa_messages(question, docs_content,langue)
    response = llm.invoke(messages)
    return response.content

