# Altair RAG Chat

Altair RAG Chat is a Streamlit application that lets you upload local PDF documents, index them with LangChain + Ollama embeddings, and ask questions via a retrieval-augmented generation (RAG) workflow. The UI stays lightweight so you can focus on experimenting without exposing any private data—the repo only contains safe defaults, while personal settings remain ignored.

## Features

- Upload one or more PDFs, split them into chunks, and store them in either Chroma or FAISS.
- Select any locally available Ollama embedding model and LLM through `config.json`.
- Automatic fallback: if the current LLM errors out, the app retries using the first LLM in your config.
- Chat history panel with a clear button so you can reset quickly between experiments.

## Requirements

1. Python 3.10+
2. [Ollama](https://ollama.com) running locally with the models you intend to use
3. The Python dependencies listed in `requirements.txt`
4. A local `config.json` file that declares the embedding/LLM options available in the UI

## Quickstart

1. Clone the repository.
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example configuration and tailor it locally (the file is gitignored):
   ```bash
   cp config.example.json config.json  # Windows: copy config.example.json config.json
   ```
5. Pull the Ollama models you reference in `config.json`, e.g.:
   ```bash
   ollama pull phi3:mini
   ollama pull nomic-embed-text
   ```
6. Launch the UI:
   ```bash
   streamlit run app.py
   ```

## Configuration

- `config.json` is never committed; it lives beside `app.py` and follows the same structure shown in `config.example.json`.
- Each entry inside `embedding_models` or `llm_models` is either a string (`"nomic-embed-text"`) or an object with `label` and `value`. Labels control how options appear in the dropdowns.
- The app automatically uses the first LLM entry as the fallback anytime the selected model fails.
- Keep any PDFs or generated vectorstores in folders that are listed in `.gitignore` (e.g., `Documents/`) so personal data never leaves your machine.

## Usage

1. Open the Streamlit app (usually `http://localhost:8501`).
2. Expand **Configuration** to tune chunk size/overlap, pick embedding + LLM models, and select Chroma or FAISS.
3. Upload PDFs and click **Process Documents**. The app loads each file, splits it into chunks, embeds them, and writes to the selected vector store.
4. Ask a question and click **Get Answer**. Results cite the retrieved chunks, and the sidebar informs you if the fallback LLM was needed.
5. Use **Chat History** to revisit questions or clear them with one click.

## Troubleshooting

- **Empty model dropdowns** → confirm `config.json` exists and has valid JSON for both model lists.
- **Ollama errors / runner terminated** → restart Ollama, repull the model, or choose a smaller model that fits in memory.
- **Slow indexing** → reduce chunk size/overlap or process fewer PDFs at once.

## Repository Layout

- `app.py` – Streamlit UI and state management.
- `backend.py` – Helpers for config loading, PDF processing, and the RAG chain.
- `requirements.txt` – Python dependencies.
- `config.example.json` – Safe template to create your own `config.json`.
- `Documents/` – Optional folder for your PDFs and other local-only assets (ignored by Git).

## License

This project is provided as-is for internal experimentation. Adapt it freely for your own use cases.

