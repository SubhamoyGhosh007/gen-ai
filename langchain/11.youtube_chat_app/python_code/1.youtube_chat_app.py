from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


DEFAULT_VIDEO_ID = "Gfr50f6ZBvo"
DEFAULT_QUESTION = (
    "Is the topic of nuclear fusion discussed in this video? "
    "If yes, what was discussed?"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
CHROMA_DIRECTORY = Path(__file__).resolve().parents[1] / "chroma_db"


def parse_args() -> argparse.Namespace:
    """Parse optional video and question values for command-line use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-id",
        default=DEFAULT_VIDEO_ID,
        help=f"YouTube video ID (default: {DEFAULT_VIDEO_ID})",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to answer from the transcript.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve (default: 4).",
    )
    return parser.parse_args()


def fetch_transcript(video_id: str) -> Document:
    """Fetch the English transcript, or the first available transcript."""
    api = YouTubeTranscriptApi()
    available_transcripts = list(api.list(video_id))
    if not available_transcripts:
        raise RuntimeError(f"No transcripts are available for video {video_id}.")

    # English is preferred because the prompt and the expected answer are in
    # English. A different language remains a useful fallback.
    selected_transcript = next(
        (
            transcript
            for transcript in available_transcripts
            if transcript.language_code in {"en", "en-US"}
        ),
        available_transcripts[0],
    )
    transcript = selected_transcript.fetch()

    return Document(
        page_content=" ".join(snippet.text for snippet in transcript.snippets),
        metadata={
            "source": f"https://www.youtube.com/watch?v={video_id}",
            "language": selected_transcript.language_code,
            "generated": selected_transcript.is_generated,
        },
    )


def split_transcript(document: Document) -> list[Document]:
    """Split the transcript into chunks suitable for semantic search."""
    # Chunk overlap keeps sentences near a boundary available to both chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=200,
    )
    return splitter.split_documents([document])


def build_vector_store(chunks: list[Document], video_id: str) -> Chroma:
    """Embed transcript chunks and persist them in a video-specific collection."""
    embedding_model = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    collection_name = f"youtube_chat_{video_id}"
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIRECTORY),
        collection_name=collection_name,
    )


def format_documents(documents: list[Document]) -> str:
    """Combine retrieved documents into the context expected by the prompt."""
    return "\n\n".join(document.page_content for document in documents)


def build_rag_chain(vector_store: Chroma, k: int):
    """Create the retrieval-augmented generation chain from the notebook."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Transcript context:
{context}

Question: {question}
"""
    )
    model = ChatGoogleGenerativeAI(model=DEFAULT_CHAT_MODEL, temperature=0.2)

    # The parallel runnable retrieves context and passes the original question
    # forward at the same time, matching the notebook's LCEL example.
    retrieval_chain = RunnableParallel(
        context=retriever | RunnableLambda(format_documents),
        question=RunnablePassthrough(),
    )
    return retrieval_chain | prompt | model | StrOutputParser()


def main() -> None:
    """Run transcript ingestion, indexing, retrieval, and generation."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in the environment or in .env.")

    args = parse_args()
    document = fetch_transcript(args.video_id)
    chunks = split_transcript(document)
    vector_store = build_vector_store(chunks, args.video_id)
    rag_chain = build_rag_chain(vector_store, args.k)
    answer = rag_chain.invoke(args.question)

    print(f"Selected language: {document.metadata['language']}")
    print(f"Auto-generated: {document.metadata['generated']}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"\nQuestion: {args.question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
