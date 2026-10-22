# Generative AI Learning Examples

This repository contains small, focused examples for learning generative AI
application development with Python and LangChain. The examples demonstrate
model providers, prompt templates, structured responses, output parsers, and
LCEL chains.

## Topics covered

- Gemini chat and text-generation models
- Ollama chat models and local workflows
- Hugging Face hosted and local models
- Gemini and Hugging Face embeddings
- Prompt templates, chat messages, and message history
- Typed dictionaries, Pydantic models, and JSON structured output
- String, structured, Pydantic, and JSON output parsers
- Sequential, parallel, and conditional runnable chains
- Document loading, text splitting, and Chroma vector stores
- Small Streamlit prompt and chatbot examples

## Project structure

All learning examples are in [`langchain/`](./langchain/):

```text
langchain/
├── 1.models/
│   ├── 1.LLM/
│   ├── 2.ChatModels/
│   └── 3.EmbeddingModel/
├── 2.prompts/
├── 3.structured_output/
├── 4.output_parser/
├── 5.chains/
├── 7.document_loader/
├── 8.text_splitters/
├── 9.vector_stores/
│   └── chroma/
├── 10.retrievers/
├── 11.other_retrievers/
└── requirements.txt
```

The numbered folders are ordered roughly from foundational concepts to more
composed LangChain workflows. Most files are standalone examples and can be
run independently.

## Requirements

- Python 3.10 or newer
- API access for the providers used by the example you want to run
- Ollama installed locally for the Ollama examples
- A compatible Python environment for local Hugging Face examples

## Setup

From the repository root, create and activate a virtual environment:

```bash
cd langchain
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Environment variables

Examples load environment variables from a `.env` file. Create
`langchain/.env` and add only the credentials needed by the examples you are
running:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_ai_api_key
HF_TOKEN=your_huggingface_token
OLLAMA_API_KEY=your_ollama_api_key
```

Never commit `.env` files or API keys. If a credential has been exposed,
revoke it with the provider and create a replacement.

## Running examples

Run an example from the `langchain` directory after activating the virtual
environment:

```bash
python 5.chains/1.simple_chain.py
python 3.structured_output/4.with_structured_output_pydantic.py
python 1.models/3.EmbeddingModel/4.\ document_similarity.py
```

Run the Streamlit examples with:

```bash
streamlit run 2.prompts/1.prompt_ui.py
streamlit run 2.prompts/7.chatbot.py
```

Some examples download models on their first run, require provider-specific
credentials, or expect a local Ollama model. Read the selected file before
running it and adjust the model name or provider configuration as needed.

## Chroma vector store examples

The examples in
[`langchain/9.vector_stores/chroma/`](./langchain/9.vector_stores/chroma/)
demonstrate a complete Chroma workflow:

1. [`1.chroma_db_create.py`](./langchain/9.vector_stores/chroma/1.chroma_db_create.py)
   creates `Document` objects, generates Google embeddings, and stores them in
   a persistent Chroma collection.
2. [`2.chroma_db.get.py`](./langchain/9.vector_stores/chroma/2.chroma_db.get.py)
   reads documents and performs similarity searches, including metadata
   filtering.
3. [`3.chroma_db_update.py`](./langchain/9.vector_stores/chroma/3.chroma_db_update.py)
   replaces a document by its Chroma document ID.
4. [`4.chroma_db_delete.py`](./langchain/9.vector_stores/chroma/4.chroma_db_delete.py)
   deletes a document by ID and displays the remaining collection contents.

Install the dependencies from the `langchain` directory, create a `.env`
file with a Google API key, and run the examples in order:

```bash
cd langchain
python -m pip install -r requirements.txt
python 9.vector_stores/chroma/1.chroma_db_create.py
python 9.vector_stores/chroma/2.chroma_db.get.py
python 9.vector_stores/chroma/3.chroma_db_update.py
python 9.vector_stores/chroma/4.chroma_db_delete.py
```

The examples persist data in the `my_chorma_db/` directory. The directory is
created relative to the current working directory, so run the commands from
`langchain/`. The create script prints generated document IDs; the update and
delete scripts use one of those IDs. The scripts use the standalone
`langchain-chroma` integration and the `gemini-embedding-2-preview` embedding
model, so a valid Google API key and network access are required.

## Retrievers backed by external sources

The examples in
[`langchain/11.other_retrievers/`](./langchain/11.other_retrievers/) retrieve
documents directly from public research databases instead of searching a
locally built vector store:

- [`1.arxiv_retriever.py`](./langchain/11.other_retrievers/1.arxiv_retriever.py)
  searches arXiv and returns research-paper summaries.
- [`2.pubmed_retriever.py`](./langchain/11.other_retrievers/2.pubmed_retriever.py)
  searches PubMed and returns biomedical article summaries.

Run them from the `langchain/` directory:

```bash
python 11.other_retrievers/1.arxiv_retriever.py
python 11.other_retrievers/2.pubmed_retriever.py
```

These examples need network access and may be affected by the public APIs'
availability or rate limits. They do not require a local vector database or
embedding model.

## Retriever examples

The examples in [`langchain/10.retrievers/`](./langchain/10.retrievers/)
demonstrate different retrieval strategies:

- `1.wikipedia_retriever.py` searches the Wikipedia API directly.
- `2.vector_store_retriver.py` exposes a FAISS vector store as a retriever.
- `3.mmr.py` uses Maximal Marginal Relevance to balance relevance and
  diversity.
- `4.mqr.py` uses an LLM to generate multiple query variations before
  retrieving documents.
- `5.ccr.py` compresses retrieved documents to keep only query-relevant text.

The vector-store-based examples use Google embeddings and require
`GOOGLE_API_KEY`. The multi-query and contextual-compression examples also
call Gemini chat models, while the Wikipedia example requires network access
and can be affected by Wikimedia rate limits.

## Notes

This is an educational repository rather than a reusable application package.
The examples intentionally favor readability and experimentation over shared
abstractions. Dependencies are collected in
[`langchain/requirements.txt`](./langchain/requirements.txt).
