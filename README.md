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
- Built-in tools, custom tools, toolkits, tool binding, and tool-calling agents
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
├── 6.runnables/
├── 7.document_loader/
├── 8.text_splitters/
├── 9.vector_stores/
│   └── chroma/
├── 10.retrievers/
├── 11.youtube_chat_app/
├── 12.tool_Calling/
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
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
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
streamlit run 11.youtube_chat_app/python_code/youtube-chat-app-streamlit.py
```

The chatbot example keeps `HumanMessage` and `AIMessage` instances in
Streamlit session state. Each request renders a `ChatPromptTemplate` with a
`MessagesPlaceholder`, so the complete conversation is sent to Gemini and
remains visible across Streamlit reruns. Use **New chat** in the sidebar to
clear the current conversation.

The YouTube transcript application uses the same pattern for RAG conversations:
run it with `streamlit run
11.youtube_chat_app/python_code/youtube-chat-app-streamlit.py`, enter a
YouTube URL, and choose an embedding provider. It preserves
`SystemMessage`/`HumanMessage`/`AIMessage` history in session state and uses a
`ChatPromptTemplate` with `MessagesPlaceholder` for follow-up questions. The
application requires `GROQ_API_KEY`; Gemini embeddings additionally require
`GOOGLE_API_KEY` or `GEMINI_API_KEY`.

The Streamlit UI also includes a browser-local IndexedDB archive. Provider keys,
conversation titles, and message history are stored in the current browser
profile, not in the repository. The sidebar contains the latest conversations,
a key panel, and a native collapse control. Conversation rendering starts with
the latest ten messages; use **Load 10 earlier messages** to page backward
without rendering the entire archive at once.

Transcript answering uses a safe full-context shortcut for transcripts up to
`FULL_CONTEXT_LIMIT` characters (24,000 by default), then falls back to the
retrieval pipeline for longer videos. The threshold can be changed in `.env`,
but it should stay within the input-token limit of the configured Groq
service.

The YouTube chat UI lets you choose the text-generation provider independently
from the embedding provider: Groq uses `GROQ_API_KEY`, while Gemini uses
`GEMINI_API_KEY` or `GOOGLE_API_KEY`. Conversations and provider keys are
hydrated from the browser's IndexedDB, while provider keys are kept only in
Streamlit session memory for this BYOK deployment. Keys are never written to
IndexedDB, and the app waits for the initial IndexedDB read before creating a
new conversation.

Text generation also supports a local Hugging Face SLM. It defaults to
`Qwen/Qwen2.5-3B-Instruct` and uses the local Transformers pipeline. The
planner and evidence stages use schema-directed JSON prompting and Pydantic
validation for structured output; native provider tool calling is used only
when the selected hosted model supports it. Override the local model with
`HF_GENERATION_MODEL_ID` in `.env`.

Some examples download models on their first run, require provider-specific
credentials, or expect a local Ollama model. Read the selected file before
running it and adjust the model name or provider configuration as needed.

## Tool-calling examples

The examples in
[`langchain/12.tool_Calling/`](./langchain/12.tool_Calling/) introduce tool
calling in increasing levels of abstraction:

1. `1.basic_tc/` demonstrates built-in tools and three ways to define a custom
   tool: the `@tool` decorator, `StructuredTool`, and `BaseTool`.
2. `2.toolkit/` shows how related tools can be grouped into a custom toolkit or
   loaded from the GitHub integration.
3. `3.toolbinding/` demonstrates binding tools to a chat model and creating an
   agent that selects and sequences tools.

The search and shell examples may require additional local configuration. The
GitHub toolkit uses `GITHUB_PERSONAL_ACCESS_TOKEN`, tool binding uses
`GROQ_API_KEY`, and the agent example uses `OPENROUTER_API_KEY`. Review the
selected file before running it, because tool calls can access external
services or execute commands in the current environment.

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
