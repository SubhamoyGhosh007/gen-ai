import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_groq import ChatGroq
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


os.environ["HF_HOME"] = r"G:\Gen AI\gen-ai\hf_models"
load_dotenv()

MAXIMUM_PASS = 3
LOCAL_EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_EMBEDDING_MODEL_ID = "gemini-embedding-2-preview"
GROQ_MODEL_ID = "openai/gpt-oss-120b"
HF_GENERATION_MODEL_ID = os.getenv(
    "HF_GENERATION_MODEL_ID",
    "Qwen/Qwen2.5-3B-Instruct",
)
FULL_CONTEXT_LIMIT = int(os.getenv("FULL_CONTEXT_LIMIT", "24000"))
GEMINI_FULL_CONTEXT_LIMIT = int(
    os.getenv("GEMINI_FULL_CONTEXT_LIMIT", "80000")
)
MAX_CONTEXT_CHARACTERS = 14000
MAX_HISTORY_MESSAGE_CHARACTERS = 2000
ANSWER_TIMEOUT_SECONDS = int(os.getenv("ANSWER_TIMEOUT_SECONDS", "90"))


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


def bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n[remaining context omitted]"


def bounded_history(messages):
    bounded_messages = []
    for message in messages[-6:]:
        content = bounded_text(
            str(message.content),
            MAX_HISTORY_MESSAGE_CHARACTERS,
        )
        if isinstance(message, HumanMessage):
            bounded_messages.append(HumanMessage(content=content))
        elif isinstance(message, AIMessage):
            bounded_messages.append(AIMessage(content=content))
        else:
            bounded_messages.append(SystemMessage(content=content))
    return bounded_messages


class FocusedRetriever:
    def __init__(self, vector_store, retriever, chunks, limit: int = 8):
        self.vector_store = vector_store
        self.retriever = retriever
        self.chunks = chunks
        self.limit = limit

    def invoke(self, query: str):
        print(f"\n[retrieval] query: {query}")
        direct_candidates = self.vector_store.similarity_search(
            query,
            k=min(8, len(self.chunks)),
        )
        semantic_candidates = list(self.retriever.invoke(query))
        print(
            f"[retrieval] direct candidates: {len(direct_candidates)}; "
            f"multi-query candidates: {len(semantic_candidates)}"
        )
        candidates = direct_candidates + semantic_candidates
        unique_documents = []
        seen_ids = set()
        for document in candidates:
            document_id = document.id or document.page_content
            if document_id not in seen_ids:
                seen_ids.add(document_id)
                unique_documents.append(document)
            if len(unique_documents) >= self.limit:
                break

        documents = list(unique_documents)
        expanded_indexes = set()
        for document in unique_documents:
            index = document.metadata.get("chunk_index")
            if index is None:
                continue
            expanded_indexes.add(index)
        for document in unique_documents:
            index = document.metadata.get("chunk_index")
            if index is None:
                continue
            for neighbor_index in (index - 1, index + 1):
                if (
                    0 <= neighbor_index < len(self.chunks)
                    and neighbor_index not in expanded_indexes
                ):
                    expanded_indexes.add(neighbor_index)
                    documents.append(self.chunks[neighbor_index])
                    if len(documents) >= self.limit * 2:
                        break
            if len(documents) >= self.limit * 2:
                break
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
When a question asks for a relationship between entities, state that
relationship only when the supplied evidence explicitly supports it. Keep
similarly named people, teams, products, and projects separate unless the
transcript explicitly connects them. If the evidence contains competing
relationships, report the ambiguity instead of combining facts from unrelated
passages.
When a question asks for an entity's project or description, inspect all
provided passages, including passages from the entity's earlier presentation,
not only the passage that announces a ranking or outcome. A name and its
description may be supported by different passages, but the transcript must
explicitly connect them before you combine them.
The transcript contains auto-generated captions, so names may be misspelled or
split. Match names by sound and meaning, but do not merge two names unless the
surrounding evidence supports that they refer to the same entity.
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
        context = bounded_text(
            "\n\n".join(document.page_content for document in documents),
            MAX_CONTEXT_CHARACTERS,
        )
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


def load_transcript_text(video_id: str) -> str:
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
    return transcript_text


def build_video_retriever(video_id: str, llm, provider: str, embedding_model=None):
    transcript_text = load_transcript_text(video_id)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
    )
    chunks = splitter.create_documents([transcript_text])
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    print(f"[transcript] total characters: {len(transcript_text)}")
    print(f"[transcript] chunks: {len(chunks)}")
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
    if vector_store._collection.count() == 0:
        vector_store.add_documents(
            documents=chunks,
            ids=[f"{video_id}-{index}" for index in range(len(chunks))],
        )
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": min(8, len(chunks)),
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


import time
import uuid

import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about a YouTube "
            "transcript. Use only transcript-grounded answers and preserve "
            "uncertainty when the transcript does not contain enough information. "
            "Do not combine entities or relationships from unrelated evidence. "
            "Only state a relationship when the transcript explicitly supports "
            "it; otherwise explain the uncertainty.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{transcript_answers}"),
    ]
)

full_context_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer questions about this YouTube transcript using only the "
            "transcript and recent conversation. Captions may misspell or split "
            "names. Match names by sound and meaning when the surrounding "
            "evidence supports that they refer to the same entity. Combine "
            "facts from separate passages only when the transcript explicitly "
            "connects them. If the evidence is insufficient, say so clearly. "
            "Be concise.\n\nTranscript:\n{transcript}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def browser_store(expression: str, key: str):
    return streamlit_js_eval(
        js_expressions=expression,
        key=key,
    )


def load_browser_state():
    result = browser_store(
        """
        (async () => {
          const db = await new Promise((resolve, reject) => {
            const request = indexedDB.open("youtube-transcript-chat", 1);
            request.onupgradeneeded = () =>
              request.result.createObjectStore("state");
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          });
          return await new Promise((resolve, reject) => {
            const request = db.transaction("state").objectStore("state").get("app");
            request.onsuccess = () => resolve(request.result || null);
            request.onerror = () => reject(request.error);
          });
        })()
        """,
        "load-indexeddb-state",
    )
    return result


def save_browser_state(state):
    browser_store(
        f"""
        (async () => {{
          const value = {json.dumps(state)};
          const db = await new Promise((resolve, reject) => {{
            const request = indexedDB.open("youtube-transcript-chat", 1);
            request.onupgradeneeded = () =>
              request.result.createObjectStore("state");
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          await new Promise((resolve, reject) => {{
            const request = db.transaction("state", "readwrite")
              .objectStore("state").put(value, "app");
            request.onsuccess = resolve;
            request.onerror = () => reject(request.error);
          }});
          return true;
        }})()
        """,
        f"save-indexeddb-state-{time.time_ns()}",
    )


def message_to_record(message):
    return {
        "role": "user" if isinstance(message, HumanMessage) else "assistant",
        "content": str(message.content),
    }


def record_to_message(record):
    if record.get("role") == "user":
        return HumanMessage(content=record.get("content", ""))
    return AIMessage(content=record.get("content", ""))


def new_conversation():
    return {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "updated_at": int(time.time()),
        "messages": [],
    }


@st.cache_resource(show_spinner="Loading the selected text-generation model...")
def get_generation_model(provider: str, api_key: str):
    if provider == "gemini":
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            retries=0,
            request_timeout=60,
        )
    if provider == "huggingface":
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=HF_GENERATION_MODEL_ID,
            task="text-generation",
            pipeline_kwargs={
                "temperature": 0.2,
                "max_new_tokens": 1024,
            },
        )
        return ChatHuggingFace(llm=pipeline)
    return ChatGroq(
        model=GROQ_MODEL_ID,
        temperature=0.2,
        max_tokens=2048,
    )


@st.cache_resource(show_spinner="Preparing the transcript search index...")
def get_retriever(
    video_id: str,
    embedding_provider: str,
    generation_provider: str,
    generation_key: str,
    embedding_key: str,
):
    del embedding_key
    generation_model = get_generation_model(generation_provider, generation_key)
    embedding_model = None
    if embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedding_model = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL_ID
        )
    return build_video_retriever(
        video_id,
        generation_model,
        embedding_provider,
        embedding_model,
    ), generation_model


@st.cache_resource(show_spinner="Loading the transcript...")
def get_transcript(video_id: str) -> str:
    return load_transcript_text(video_id)


def answer_full_context(query, transcript, llm, chat_history):
    chain = full_context_template | llm | StrOutputParser()
    return chain.invoke(
        {
            "transcript": transcript,
            "chat_history": bounded_history(chat_history),
            "question": query,
        }
    )


def run_answer_with_timeout(answer_fn):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(answer_fn)
    try:
        return future.result(timeout=ANSWER_TIMEOUT_SECONDS)
    except TimeoutError as error:
        future.cancel()
        raise TimeoutError(
            f"The {ANSWER_TIMEOUT_SECONDS}-second answer timeout was reached. "
            "Check the provider key, network connection, and Gemini quota."
        ) from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def answer_query(query, retriever, llm, chat_history):
    conversation_context = "\n".join(
        f"{message.type}: "
        f"{bounded_text(str(message.content), MAX_HISTORY_MESSAGE_CHARACTERS)}"
        for message in chat_history[-6:]
    )
    planning_prompt = PromptTemplate(
        template="""
Split the user's question into the minimum number of questions needed to answer
it from the transcript. Do not split a question that can stand alone.

A question is independent only when it can be written without another answer.
For every dependent question, list the required IDs in depends_on and include
each required answer explicitly with a placeholder such as {{q1}} in the
question text. Resolve pronouns and references such as "their", "that project",
"it", and "the winner" using the recent conversation below. Rewrite each
question as a standalone transcript question with the resolved name or project.
Do not create a clarification question when the conversation already identifies
the subject. Use unique positive IDs and set layer to 1 for initial questions.

Recent conversation:
{conversation_context}

User question: {query}
""",
        input_variables=["conversation_context", "query"],
    )
    plan = invoke_structured(
        planning_prompt.invoke(
            {
                "conversation_context": conversation_context
                or "No earlier conversation.",
                "query": query,
            }
        ),
        Plan,
        llm,
    )
    answer_model = RunnableLambda(
        lambda prompt: invoke_structured(
            prompt,
            QuestionAnswer,
            llm,
        )
    )
    answers = solve_plan(plan, retriever, answer_model)
    answer_text = "\n\n".join(
        f"q{question_id}: {result.answer}\nEvidence: {result.evidence}"
        for question_id, result in sorted(answers.items())
    )
    final_chain = (
        chat_template
        | llm
        | StrOutputParser()
    )
    return final_chain.invoke(
        {
            "chat_history": bounded_history(chat_history),
            "transcript_answers": (
                f"Original question: {query}\n"
                f"Transcript-grounded sub-question answers:\n{answer_text}\n\n"
                "Respond directly to the original question."
            ),
        }
    )


def main():
    st.set_page_config(page_title="YouTube Transcript Chat", page_icon="YT")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { border-right: 1px solid #243447; }
        [data-testid="stSidebar"] button { text-align: left; }
        .chat-kicker { color: #78dcca; letter-spacing: .08em; font-size: .72rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "browser_state_loaded" not in st.session_state:
        loaded_state = load_browser_state()
        if loaded_state is None:
            # The browser bridge can return no value during its first render.
            # Do not block the chat UI if that component does not trigger a
            # second Streamlit rerun.
            loaded_state = {}
            st.warning(
                "Browser history is still loading. New messages will be "
                "saved to this browser."
            )
        if not isinstance(loaded_state, dict):
            loaded_state = {}
        # Provider keys are BYOK session data, never durable browser data.
        had_persisted_keys = loaded_state.pop("keys", None) is not None
        if had_persisted_keys:
            save_browser_state(loaded_state)
        st.session_state.browser_state = loaded_state
        st.session_state.provider_keys = {}
        st.session_state.browser_state_loaded = True
    state = st.session_state.browser_state
    if not isinstance(state, dict):
        state = {}
        st.session_state.browser_state = state
    conversations = state.setdefault("conversations", [])
    keys = st.session_state.provider_keys
    if not conversations:
        conversations.append(new_conversation())
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = conversations[-1]["id"]
    current = next(
        (item for item in conversations
         if item["id"] == st.session_state.conversation_id),
        conversations[-1],
    )

    with st.sidebar:
        st.markdown('<div class="chat-kicker">YOUTUBE TRANSCRIPT CHAT</div>',
                    unsafe_allow_html=True)
        if st.button("New conversation", use_container_width=True):
            conversation = new_conversation()
            conversations.append(conversation)
            st.session_state.conversation_id = conversation["id"]
            st.session_state.browser_state = state
            save_browser_state(state)
            st.rerun()

        st.subheader("Recent conversations")
        conversation_limit = st.session_state.get("conversation_limit", 20)
        archived_conversations = [
            conversation
            for conversation in conversations
            if conversation["messages"] or conversation["id"] != current["id"]
        ]
        visible_conversations = archived_conversations[-conversation_limit:]
        for conversation in reversed(visible_conversations):
            title = conversation["title"][:42]
            if st.button(title or "Untitled conversation",
                         key=f"conversation-{conversation['id']}",
                         use_container_width=True):
                st.session_state.conversation_id = conversation["id"]
                st.rerun()
        if len(archived_conversations) > len(visible_conversations):
            if st.button("Load older conversations", use_container_width=True):
                st.session_state.conversation_limit = min(
                    conversation_limit + 20, len(archived_conversations)
                )
                st.rerun()

        with st.expander("Provider keys", expanded=not keys):
            groq_key = st.text_input(
                "Groq API key", value=keys.get("groq", ""), type="password"
            )
            gemini_key = st.text_input(
                "Gemini API key", value=keys.get("gemini", ""), type="password"
            )
            if st.button("Save keys", use_container_width=True):
                keys.update({"groq": groq_key, "gemini": gemini_key})
                st.session_state.provider_keys = keys
                st.success("Keys saved for this session only.")

        provider = st.radio(
            "Embedding provider",
            options=("huggingface", "gemini"),
            format_func=lambda item: (
                "Local Hugging Face" if item == "huggingface" else "Gemini"
            ),
        )
        generation_provider = st.radio(
            "Text generation provider",
            options=("groq", "gemini", "huggingface"),
            format_func=lambda item: (
                {
                    "groq": "Groq",
                    "gemini": "Gemini",
                    "huggingface": "Local Hugging Face SLM",
                }[item]
            ),
        )

    st.title(current["title"] if current["title"] != "New conversation"
             else "YouTube Transcript Chat")
    st.caption("A local conversation archive for transcript-grounded follow-ups.")
    video_id = current.get("video_id")
    video_url = current.get("video_url", "")
    if not video_id and video_url:
        video_id = get_video_id(video_url)
        if video_id:
            current["video_id"] = video_id
    if video_id:
        st.caption(f"Transcript source: {video_url}")
    else:
        st.subheader("Set the conversation source")
        source_url = st.text_input(
            "YouTube URL",
            key=f"source-url-{current['id']}",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        if st.button("Use this video", type="primary"):
            selected_video_id = get_video_id(source_url)
            if not selected_video_id:
                st.error(
                    "Enter a YouTube watch URL, for example: "
                    "https://www.youtube.com/watch?v=okJqVvkQTQ8"
                )
            else:
                current["video_url"] = source_url.strip()
                current["video_id"] = selected_video_id
                state["conversations"] = conversations
                st.session_state.browser_state = state
                save_browser_state(state)
                st.rerun()
        st.info("Choose a video once. Its source is locked for this conversation.")

    current_messages = current.get("messages", [])
    message_limit = st.session_state.get(
        f"message_limit-{current['id']}", min(10, len(current_messages))
    )
    if len(current_messages) > message_limit and st.button(
        "Load 10 earlier messages", use_container_width=True
    ):
        st.session_state[f"message_limit-{current['id']}"] = min(
            message_limit + 10, len(current_messages)
        )
        st.rerun()

    for record in current_messages[-message_limit:]:
        with st.chat_message(record["role"]):
            st.markdown(record["content"])

    if query := st.chat_input("Ask about the transcript..."):
        if not video_id:
            st.error("Choose a YouTube video for this conversation first.")
            st.stop()
        groq_key = keys.get("groq", "")
        gemini_key = keys.get("gemini", "")
        generation_key = {
            "groq": groq_key,
            "gemini": gemini_key,
            "huggingface": "local",
        }[generation_provider]
        if not generation_key:
            st.error(
                f"Add a {generation_provider.title()} API key in the "
                "Provider keys panel."
            )
            st.stop()
        if provider == "gemini" and not gemini_key:
            st.error("Add a Gemini API key to use Gemini embeddings.")
            st.stop()
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["GOOGLE_API_KEY"] = gemini_key

        history = [record_to_message(record) for record in current_messages]
        user_message = HumanMessage(content=query)
        current["messages"].append(message_to_record(user_message))
        current["updated_at"] = int(time.time())
        if current["title"] == "New conversation":
            current["title"] = query[:48]
        state["conversations"] = conversations
        save_browser_state(state)
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            try:
                status = st.status(
                    "Starting transcript search...",
                    expanded=True,
                )
                with status:
                    status.write("Loading the transcript...")
                    transcript = get_transcript(video_id)
                    context_limit = (
                        GEMINI_FULL_CONTEXT_LIMIT
                        if generation_provider == "gemini"
                        else FULL_CONTEXT_LIMIT
                    )
                    if len(transcript) <= context_limit:
                        status.write("Reading the transcript with the selected model...")
                        response = run_answer_with_timeout(
                            lambda: answer_full_context(
                                query,
                                transcript,
                                get_generation_model(
                                    generation_provider,
                                    generation_key,
                                ),
                                history,
                            )
                        )
                    else:
                        status.write("Preparing the transcript search index...")
                        def run_retrieval_answer():
                            retriever, llm = get_retriever(
                                video_id,
                                provider,
                                generation_provider,
                                generation_key,
                                gemini_key if provider == "gemini" else "",
                            )
                            return answer_query(
                                query,
                                retriever,
                                llm,
                                history + [user_message],
                            )

                        status.write(
                            "Searching transcript evidence and generating an answer..."
                        )
                        response = run_answer_with_timeout(run_retrieval_answer)
                    status.update(
                        label="Answer ready",
                        state="complete",
                        expanded=False,
                    )
                st.markdown(response)
                current["messages"].append(
                    message_to_record(AIMessage(content=response))
                )
                state["conversations"] = conversations
                st.session_state.browser_state = state
                save_browser_state(state)
            except Exception as error:
                st.error(f"Unable to answer the question: {error}")


if __name__ == "__main__":
    main()
