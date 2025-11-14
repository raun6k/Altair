import json
import os
import tempfile
from typing import List, Optional, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

CONFIG_FILENAME = "config.json"


def _normalize_model_list(values: Optional[List]) -> List[dict]:
    normalized = []
    for value in values or []:
        if isinstance(value, str):
            normalized.append({"label": value, "value": value})
        elif isinstance(value, dict):
            label = value.get("label") or value.get("name") or value.get("value")
            model_value = value.get("value") or label
            if label and model_value:
                normalized.append({"label": label, "value": model_value})
    return normalized


def load_model_options(config_path: str | None = None) -> Tuple[List[dict], List[dict], str | None]:
    """Return embedding/LLM model option lists sourced from config.json."""
    config_path = config_path or os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    config = {}
    warnings: List[str] = []

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except json.JSONDecodeError:
            warnings.append("Could not parse config.json. Ensure it contains valid JSON.")
    else:
        warnings.append("config.json not found. Create one to list embedding and LLM models.")

    embedding_models = _normalize_model_list(config.get("embedding_models"))
    llm_models = _normalize_model_list(config.get("llm_models"))

    if not embedding_models:
        warnings.append("No embedding models defined. Update config.json with at least one entry.")
    if not llm_models:
        warnings.append("No LLM models defined. Update config.json with at least one entry.")

    warning_message = " ".join(warnings) if warnings else None
    return embedding_models, llm_models, warning_message


def load_and_process_docs(files, chunk_size: int, chunk_overlap: int, embedding_model: str, vectorstore_type: str):
    """Load PDFs, split into chunks, and create vector store."""
    all_documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            all_documents.extend(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(all_documents)

        embeddings = OllamaEmbeddings(model=embedding_model)

        if vectorstore_type == "Chroma":
            vectorstore = Chroma.from_documents(
                chunks,
                embeddings,
                collection_name="simple-rag",
            )
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def run_rag(question_text: str, model_name: str, vectorstore):
    """Execute the RAG chain for a given question and model name."""
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question:
If you don't know the answer, then answer from your own knowledge and dont give just one word answer, and dont tell the user that you are answering from your knowledge.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_name)
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question_text)
