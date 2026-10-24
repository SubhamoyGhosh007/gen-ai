import os
import re
import json
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


os.environ["HF_HOME"] = r"G:\Gen AI\gen-ai\hf_models"
load_dotenv()

MAXIMUM_PASS = 3
LOCAL_EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_EMBEDDING_MODEL_ID = "gemini-embedding-2-preview"
GROQ_MODEL_ID = "openai/gpt-oss-120b"
RETRIEVAL_QUERY_VARIANTS = (
    "first place winner winning team champion",
    "winner project what did the winning team build",
    "competition results first second third place",
)


class SubQuestion(BaseModel):
    id: int = Field(..., description="Unique positive ID for this sub-question")
    question: str = Field(
        ...,
        description=(
            "A minimum question. Dependent questions may refer to earlier answers "
            "with placeholders such as {q1}."
        ),
    )
    depends_on: List[int] = Field(
        default_factory=list,
        description="IDs whose answers are required before answering this question",
    )
    layer: int = Field(
        default=1,
        description="Dependency layer; initial questions belong to layer 1",
    )


class Plan(BaseModel):
    sub_questions: List[SubQuestion] = Field(
        ...,
        description="The minimum dependency-aware set of questions",
    )


class FollowUpQuestion(BaseModel):
    question: str = Field(
        ...,
        description="A new transcript question needed to complete the current answer",
    )
    depends_on: List[int] = Field(
        default_factory=list,
        description="Existing question IDs needed by this follow-up",
    )


class QuestionAnswer(BaseModel):
    evidence: str = Field(
        ...,
        description=(
            "Exact sentence or sentences from the supplied context supporting "
            "the answer, or 'none' when no evidence exists"
        ),
    )
    answer: str = Field(
        ...,
        description="Answer using only the supplied transcript context",
    )
    short_answer: str = Field(
        ...,
        description=(
            "The shortest useful entity or answer for dependent questions, such "
            "as a person's name, team name, or project name"
        ),
    )
    follow_up_questions: List[FollowUpQuestion] = Field(
        default_factory=list,
        description=(
            "New dependent questions required to complete this answer. Empty when "
            "no more information is needed."
        ),
    )


def choose_provider():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "Groq generation requires GROQ_API_KEY in the environment or .env file."
        )

    generation_model = ChatGroq(
        model=GROQ_MODEL_ID,
        temperature=0.2,
        max_tokens=2048,
    )
    print(f"Using Groq generation model: {GROQ_MODEL_ID}")

    while True:
        choice = input(
            "Choose provider (1 = Gemini from environment, "
            "2 = local Hugging Face embeddings): "
        ).strip().lower()
        if choice in {"1", "gemini", "g"}:
            if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                raise ValueError(
                    "Gemini selected, but GOOGLE_API_KEY or GEMINI_API_KEY "
                    "was not found in the environment or .env file."
                )
            print("Using Gemini embeddings from the configured environment key.")
            return (
                generation_model,
                GoogleGenerativeAIEmbeddings(
                    model=GEMINI_EMBEDDING_MODEL_ID,
                ),
                "gemini",
            )
        if choice in {"2", "huggingface", "hf", "h"}:
            print("Using local Hugging Face embeddings.")
            return generation_model, None, "huggingface"
        print("Please choose 1 for Gemini or 2 for local Hugging Face.")


def invoke_structured(prompt, schema, llm):
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    structured_prompt = (
        f"{prompt_text}\n\nReturn only valid JSON matching this schema:\n"
        f"{schema.model_json_schema()}"
    )
    response = llm.invoke(structured_prompt)
    content = response.content if hasattr(response, "content") else str(response)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    object_start = content.find("{")
    if object_start > 0:
        content = content[object_start:]
    try:
        parsed = json.JSONDecoder().raw_decode(content)[0]
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{schema.__name__} model output was not complete JSON: {error}"
        ) from error
    return schema.model_validate(parsed)


def get_video_id(url: str) -> str | None:
    parsed_url = urlparse(url.strip())
    hostname = (parsed_url.hostname or "").lower()

    if hostname == "youtu.be":
        return parsed_url.path.lstrip("/") or None

    if hostname in {
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    return None


def validate_plan(plan: Plan) -> Dict[int, SubQuestion]:
    questions: Dict[int, SubQuestion] = {}

    for question in plan.sub_questions:
        if question.id <= 0:
            raise ValueError(f"Question IDs must be positive: q{question.id}")
        if question.id in questions:
            raise ValueError(f"Duplicate question ID: q{question.id}")
        questions[question.id] = question

    known_ids = set(questions)
    for question in questions.values():
        missing = set(question.depends_on) - known_ids
        if missing:
            missing_ids = ", ".join(f"q{item}" for item in sorted(missing))
            raise ValueError(
                f"q{question.id} depends on unknown question IDs: {missing_ids}"
            )
        if question.id in question.depends_on:
            raise ValueError(f"q{question.id} cannot depend on itself")

    return questions


def replace_answer_references(
    question: str,
    answers: Dict[int, QuestionAnswer],
) -> str:
    def replace_reference(match: re.Match[str]) -> str:
        question_id = int(match.group(1))
        answer = answers.get(question_id)
        return answer.short_answer if answer else match.group(0)

    return re.sub(r"\{q(\d+)\}", replace_reference, question)


def dependency_context(
    question: SubQuestion,
    answers: Dict[int, QuestionAnswer],
) -> str:
    if not question.depends_on:
        return "No earlier answers are required."

    return "\n\n".join(
        f"Answer to q{dependency_id}:\n"
        f"{answers[dependency_id].answer}\n"
        f"Relevant entity: {answers[dependency_id].short_answer}\n"
        f"Evidence: {answers[dependency_id].evidence}"
        for dependency_id in question.depends_on
    )


class FocusedRetriever:
    def __init__(self, vector_store, retriever, chunks, limit: int = 8):
        self.vector_store = vector_store
        self.retriever = retriever
        self.chunks = chunks
        self.limit = limit

    def invoke(self, query: str):
        print(f"\n[retrieval] query: {query}")
        lowered_query = query.casefold()
        focused_candidates = []
        if any(
            term in lowered_query
            for term in ("winner", "won", "first place", "winning", "project")
        ):
            print("[retrieval] applying winner-focused query variants")
            for variant in RETRIEVAL_QUERY_VARIANTS:
                variant_documents = self.vector_store.similarity_search(variant, k=2)
                print(
                    f"[retrieval] variant={variant!r} "
                    f"matches={len(variant_documents)}"
                )
                focused_candidates.extend(variant_documents)

        semantic_candidates = list(self.retriever.invoke(query))
        print(f"[retrieval] multi-query candidates: {len(semantic_candidates)}")
        candidates = focused_candidates + semantic_candidates
        unique_documents = []
        seen_ids = set()
        for document in candidates:
            document_id = document.id or document.page_content
            if document_id not in seen_ids:
                seen_ids.add(document_id)
                unique_documents.append(document)
            if len(unique_documents) >= self.limit:
                break

        selected_indexes = {
            document.metadata.get("chunk_index")
            for document in unique_documents
            if document.metadata.get("chunk_index") is not None
        }
        expanded_indexes = {
            index + offset
            for index in selected_indexes
            for offset in (-1, 0, 1)
            if 0 <= index + offset < len(self.chunks)
        }
        documents = [
            self.chunks[index]
            for index in sorted(expanded_indexes)
        ]
        print(
            f"[retrieval] selected documents: {len(unique_documents)}; "
            f"with neighbors: {len(documents)}"
        )
        for index, document in enumerate(documents, start=1):
            print(
                f"[retrieval] doc {index}: "
                f"{document.page_content[:240].replace(chr(10), ' ')}"
            )
        return documents


def answer_ready_questions(
    ready_questions: List[SubQuestion],
    answers: Dict[int, QuestionAnswer],
    retriever,
    answer_model,
) -> Dict[int, QuestionAnswer]:
    answer_prompt = PromptTemplate(
        template="""
You answer questions about a YouTube transcript.
Use only the transcript context. If the context is insufficient, say so clearly.
Treat ranking words such as first, second, third, winner, finalist, and champion
as important evidence. Do not say that a fact is absent merely because one
retrieved passage does not contain it; inspect every supplied passage first.
The transcript contains auto-generated captions, so names may be misspelled or
split. Match names by sound and meaning, not exact spelling. For example,
"Raktosh", "Rakt Kosh", and "Rakt Khoj" may refer to the same project or team.
If a passage plausibly refers to the subject under a caption variant, use it and
mention the variant you matched. Say the information was not found only when no
passage matches, even phonetically or semantically. Never use an unrelated
project or passage.

Question:
{question}

Answers to required earlier questions:
{dependencies}

Transcript context:
{context}

Return supporting evidence copied from the context, the answer, a short entity
or answer for dependent questions, and only when necessary, new dependent
transcript questions. Keep evidence to at most two short sentences, answer to
at most 80 words, short_answer to at most 12 words, and follow-ups to at most
two items.
Every follow-up must include the current question ID {current_id} in depends_on.
Do not duplicate an earlier question and do not ask for information already
present in the context or earlier answers.
""",
        input_variables=["question", "dependencies", "context", "current_id"],
    )
    answer_chain = answer_prompt | answer_model

    def answer_one(question: SubQuestion) -> QuestionAnswer:
        rendered_question = replace_answer_references(question.question, answers)
        if question.depends_on:
            entities = ", ".join(
                answers[dependency_id].short_answer
                for dependency_id in question.depends_on
            )
            rendered_question = (
                f"{rendered_question} The subject identified by the earlier "
                f"answer is: {entities}."
            )
        documents = retriever.invoke(rendered_question)
        context = "\n\n".join(document.page_content for document in documents)
        print(
            f"[answer] q{question.id} context characters: {len(context)}"
        )
        if not context.strip():
            print(f"[answer] WARNING q{question.id} received empty context")
        result = answer_chain.invoke(
            {
                "question": rendered_question,
                "dependencies": dependency_context(question, answers),
                "context": context,
                "current_id": question.id,
            }
        )
        if result.evidence.casefold() == "none":
            print(f"[answer] q{question.id} returned no evidence")
        else:
            print(f"[answer] q{question.id} evidence: {result.evidence}")
        print(f"[answer] q{question.id}: {result.answer}")
        print(f"[answer] q{question.id} short answer: {result.short_answer}")
        if result.follow_up_questions:
            print(
                f"[answer] q{question.id} follow-ups: "
                + "; ".join(item.question for item in result.follow_up_questions)
            )
        return result

    parallel_chain = RunnableParallel(
        {
            str(question.id): RunnableLambda(
                lambda _input, question=question: answer_one(question)
            )
            for question in ready_questions
        }
    )
    results = parallel_chain.invoke({})
    return {int(question_id): result for question_id, result in results.items()}


def add_follow_up_questions(
    questions: Dict[int, SubQuestion],
    completed_question: SubQuestion,
    result: QuestionAnswer,
    next_id: int,
    pass_number: int,
) -> int:
    known_ids = set(questions)
    existing_question_text = {
        question.question.strip().casefold() for question in questions.values()
    }

    for follow_up in result.follow_up_questions:
        dependencies = set(follow_up.depends_on)
        dependencies.add(completed_question.id)
        missing = dependencies - known_ids
        if missing:
            missing_ids = ", ".join(f"q{item}" for item in sorted(missing))
            print(f"Skipping follow-up with unknown dependencies: {missing_ids}")
            continue

        normalized_text = follow_up.question.strip().casefold()
        if not normalized_text or normalized_text in existing_question_text:
            continue

        questions[next_id] = SubQuestion(
            id=next_id,
            question=follow_up.question.strip(),
            depends_on=sorted(dependencies),
            layer=pass_number + 1,
        )
        existing_question_text.add(normalized_text)
        next_id += 1

    return next_id


def solve_plan(
    plan: Plan,
    retriever,
    answer_model,
    max_passes: int = MAXIMUM_PASS,
) -> Dict[int, QuestionAnswer]:
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    questions = validate_plan(plan)
    answers: Dict[int, QuestionAnswer] = {}
    next_id = max(questions, default=0) + 1

    for pass_number in range(1, max_passes + 1):
        ready_questions = [
            question
            for question in questions.values()
            if question.id not in answers
            and set(question.depends_on).issubset(answers)
        ]
        if not ready_questions:
            break

        print(
            f"Pass {pass_number}/{max_passes}: answering "
            + ", ".join(f"q{question.id}" for question in ready_questions)
        )
        batch_answers = answer_ready_questions(
            ready_questions,
            answers,
            retriever,
            answer_model,
        )

        for question in ready_questions:
            result = batch_answers[question.id]
            answers[question.id] = result
            next_id = add_follow_up_questions(
                questions,
                question,
                result,
                next_id,
                pass_number,
            )

    unresolved = [
        question for question in questions.values() if question.id not in answers
    ]
    if unresolved:
        unresolved_ids = ", ".join(f"q{question.id}" for question in unresolved)
        print(f"Unresolved after {max_passes} passes: {unresolved_ids}")

    return answers


def build_video_retriever(video_id: str, llm, provider: str, embedding_model=None):
    youtube_api = YouTubeTranscriptApi()
    available_transcripts = list(youtube_api.list(video_id))
    if not available_transcripts:
        raise ValueError("No transcripts were found for this video")

    selected_transcript = next(
        (
            transcript
            for transcript in available_transcripts
            if transcript.language_code in {"en", "en-US"}
        ),
        available_transcripts[0],
    )
    loader = YoutubeLoader(
        video_id=video_id,
        language=[selected_transcript.language_code],
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=8,
    )
    transcript_documents = loader.load()
    transcript_text = " ".join(
        document.page_content for document in transcript_documents
    )
    if not transcript_text.strip():
        raise ValueError("The selected transcript is empty")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
    )
    chunks = splitter.create_documents([transcript_text])
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    print(f"[transcript] source documents: {len(transcript_documents)}")
    print(f"[transcript] total characters: {len(transcript_text)}")
    print(f"[transcript] chunks: {len(chunks)}")
    keyword_matches = (
        "first place",
        "winner",
        "winning team",
        "champion",
        "second place",
        "third place",
    )
    matching_chunks = [
        (index, chunk)
        for index, chunk in enumerate(chunks)
        if any(keyword in chunk.page_content.casefold() for keyword in keyword_matches)
    ]
    print(f"[transcript] ranking-related chunks: {len(matching_chunks)}")
    for index, chunk in matching_chunks:
        print(
            f"[transcript] matching chunk {index}: "
            f"{chunk.page_content[:500].replace(chr(10), ' ')}"
        )

    if provider == "huggingface":
        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL_ID,
            model_kwargs={"device": embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        collection_version = "huggingface-v2"
        persist_directory = (
            r"G:\Gen AI\gen-ai\youtube-chat-app-rag\huggingface"
        )
    else:
        if embedding_model is None:
            raise ValueError("Gemini embedding model was not initialized")
        collection_version = "gemini-v1"
        persist_directory = r"G:\Gen AI\gen-ai\youtube-chat-app-rag\gemini"

    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_name=f"youtube-chat-app-{collection_version}-{video_id}",
    )
    vector_store.add_documents(
        documents=chunks,
        ids=[f"{video_id}-{index}" for index in range(len(chunks))],
    )
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            # Retrieve a small candidate window, then compression keeps only
            # relevant sentences before the answer model sees the context.
            "k": min(5, len(chunks)),
            "lambda_mult": 0.7,
        },
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )
    return FocusedRetriever(
        vector_store,
        multi_query_retriever,
        chunks,
    )


def main() -> None:
    video_url = input("Enter the YouTube URL: ")
    video_id = get_video_id(video_url)
    if not video_id:
        raise ValueError("URL is not a valid YouTube URL")

    llm, embedding_model, provider = choose_provider()
    retriever = build_video_retriever(
        video_id,
        llm,
        provider,
        embedding_model,
    )
    query = input("What is your query? : ").strip()
    if not query:
        raise ValueError("The question cannot be empty")
    planning_prompt = PromptTemplate(
        template="""
Split the user's question into the minimum number of questions needed to answer
it from the transcript. Do not split a question that can stand alone.

A question is independent only when it can be written without another answer.
For every dependent question, list the required IDs in depends_on and include
each required answer explicitly with a placeholder such as {{q1}} in the
question text. Never write a dependent question such as "What was their
project?" without identifying which earlier answer "their" refers to.
Use unique positive IDs, include every dependency, and set layer to 1 for the
initial questions.

User question: {query}
""",
        input_variables=["query"],
    )
    plan = invoke_structured(planning_prompt.invoke({"query": query}), Plan, llm)
    print("[planner] sub-questions:")
    for question in plan.sub_questions:
        print(
            f"[planner] q{question.id}: {question.question} "
            f"depends_on={question.depends_on} layer={question.layer}"
        )

    answer_model = RunnableLambda(
        lambda prompt: invoke_structured(prompt, QuestionAnswer, llm)
    )
    answers = solve_plan(plan, retriever, answer_model)
    answer_text = "\n\n".join(
        f"q{question_id}: {result.answer}\n"
        f"Evidence: {result.evidence}"
        for question_id, result in sorted(answers.items())
    )

    final_chain = (
        PromptTemplate(
            template="""
Answer the original question using only these transcript-grounded answers.
Combine them into one concise, coherent response. Preserve any limitation that
the transcript does not contain enough information.
Only state facts that appear in the sub-question answers. Never combine a name
from one answer with a project from an unrelated answer.

Original question: {query}
Sub-question answers:
{answers}
""",
            input_variables=["query", "answers"],
        )
        | llm
        | StrOutputParser()
    )
    print("[final] sub-question answers:")
    print(answer_text)
    final_answer = final_chain.invoke({"query": query, "answers": answer_text})
    print("\nFinal answer:\n" + final_answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
