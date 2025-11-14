import streamlit as st

from backend import (
    load_model_options,
    load_and_process_docs,
    run_rag,
)

st.set_page_config(page_title="Altair RAG Chat", page_icon="💬", layout="centered")

EMBEDDING_OPTIONS, LLM_OPTIONS, CONFIG_WARNING = load_model_options()
if CONFIG_WARNING:
    st.warning(CONFIG_WARNING)

if not EMBEDDING_OPTIONS or not LLM_OPTIONS:
    st.error(
        "No models available. Please add both `embedding_models` and `llm_models` entries to "
        "`config.json`, then refresh the app.",
    )
    st.stop()

EMBEDDING_LOOKUP = {option["value"]: option["label"] for option in EMBEDDING_OPTIONS}
LLM_LOOKUP = {option["value"]: option["label"] for option in LLM_OPTIONS}
FALLBACK_LLM_MODEL = LLM_OPTIONS[0]["value"]

STATE_DEFAULTS = {
    "vectorstore": None,
    "chat_history": [],
    "num_chunks": 0,
    "processed_files": [],
    "chunk_size": 1000,
    "chunk_overlap": 20,
    "vectorstore_type": "Chroma",
    "embedding_model": EMBEDDING_OPTIONS[0]["value"],
    "llm_model": LLM_OPTIONS[0]["value"],
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Altair RAG Chat")
st.caption("Upload PDF files, process them once, and ask quick questions about their contents.")

with st.expander("Configuration", expanded=False):
    st.session_state.chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        step=100,
        value=st.session_state.chunk_size,
    )
    st.session_state.chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        step=10,
        value=st.session_state.chunk_overlap,
    )

    embedding_values = [option["value"] for option in EMBEDDING_OPTIONS]
    st.session_state.embedding_model = st.selectbox(
        "Embedding Model",
        embedding_values,
        index=embedding_values.index(st.session_state.embedding_model),
        format_func=lambda value: EMBEDDING_LOOKUP.get(value, value),
    )

    llm_values = [option["value"] for option in LLM_OPTIONS]
    st.session_state.llm_model = st.selectbox(
        "Language Model",
        llm_values,
        index=llm_values.index(st.session_state.llm_model),
        format_func=lambda value: LLM_LOOKUP.get(value, value),
    )

    st.session_state.vectorstore_type = st.radio(
        "Vector Store",
        ["Chroma", "FAISS"],
        index=0 if st.session_state.vectorstore_type == "Chroma" else 1,
    )

st.divider()

uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
)
process_button = st.button("Process Documents", use_container_width=True)

if process_button:
    if uploaded_files:
        with st.spinner("Processing documents..."):
            try:
                vectorstore, num_chunks = load_and_process_docs(
                    uploaded_files,
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap,
                    st.session_state.embedding_model,
                    st.session_state.vectorstore_type,
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = num_chunks
                st.session_state.processed_files = [file.name for file in uploaded_files]
                st.success(f"Processed {len(uploaded_files)} file(s) into {num_chunks} chunks.")
            except Exception as exc:
                st.error(f"Error processing documents: {exc}")
    else:
        st.warning("Upload at least one PDF before processing.")

st.divider()

question = st.text_input(
    "Enter your question",
    placeholder="What is covered in this document?",
)
ask_button = st.button("Get Answer", use_container_width=True)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vectorstore is None:
        st.warning("Process at least one document first.")
    else:
        with st.spinner("Generating answer..."):
            answer = None
            last_error = None
            used_model = None
            primary_model = st.session_state.llm_model
            models_to_try = [primary_model]
            if FALLBACK_LLM_MODEL != primary_model:
                models_to_try.append(FALLBACK_LLM_MODEL)

            for model_name in models_to_try:
                try:
                    answer = run_rag(question, model_name, st.session_state.vectorstore)
                    used_model = model_name
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc

            if answer is None:
                error_text = str(last_error) if last_error else "Unknown error."
                if "Ollama call failed" in error_text or "runner process" in error_text:
                    st.warning(
                        "The selected language model is unavailable right now. "
                        "Try processing with a different model or restart Ollama.",
                    )
                else:
                    st.error(f"Error generating answer: {error_text}")
            else:
                st.session_state.chat_history.append({"question": question, "answer": answer})
                if used_model != primary_model:
                    fallback_label = LLM_LOOKUP.get(used_model, used_model)
                    primary_label = LLM_LOOKUP.get(primary_model, primary_model)
                    st.info(
                        f"Primary model '{primary_label}' failed, fallback '{fallback_label}' produced the answer.",
                    )

if st.session_state.chat_history:
    st.subheader("Chat History")
    for index, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        st.markdown(f"**Q{index}: {chat['question']}**")
        st.write(chat["answer"])
        st.divider()
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.info("No questions yet. Upload files, process them, and ask anything.")

st.caption("Built with Streamlit, LangChain, and Ollama")
