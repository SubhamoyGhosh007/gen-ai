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

## Notes

This is an educational repository rather than a reusable application package.
The examples intentionally favor readability and experimentation over shared
abstractions. Dependencies are collected in
[`langchain/requirements.txt`](./langchain/requirements.txt).
