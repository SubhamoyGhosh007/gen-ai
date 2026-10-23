import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Optional,List
from dotenv import load_dotenv

os.environ['HF_HOME'] = r"G:\Gen AI\gen-ai\hf_models"

load_dotenv()

def url2id(url):
    parsed_url = urlparse(url)
    print(parsed_url)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com', 'music.youtube.com'):
        query_params = parse_qs(parsed_url.query)
        print(query_params)
        if 'v' in query_params:
            return query_params['v'][0]
        else:
            return None
    else:
        return None
    
video_url = input("Enter the YouTube URL: ")
youtube_api = YouTubeTranscriptApi()

try:
    video_id = url2id(video_url)
    if not video_id:
        print('URL is not valid')
        exit(1)
    print(video_id)
    available_transcripts = list(youtube_api.list(video_id))
    if not available_transcripts:
        print("No transcripts were found")
        exit(1)
    
    transcript_fetch = next(
        (
          transcript  for transcript in available_transcripts if transcript.language_code in ['en' , 'en-US']    
        ), available_transcripts[0]
    )
    
    transcript = transcript_fetch.fetch()
    
    loader = YoutubeLoader(
        video_id, 
        language = [transcript.language_code],
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=8
    )
    docs = loader.load()
    
    print(len(docs))
    # print(docs)
    # for i, doc in enumerate(docs):
    #     print(doc.page_content)
    #     print(doc.metadata)
    #     print('-' * 20)
    #     if i > 2:
    #         break
    transcript_text = " ".join(doc.page_content for doc in docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = splitter.create_documents(
        [transcript_text]
    )
    # print(chunks[0])
    # print(len(chunks))
    
    embedding_model = HuggingFaceEmbeddings(
        model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=r"G:\Gen AI\gen-ai\youtube-chat-app-rag",
        collection_name='youtube-chat-app'
    )
    
    doc_ids = vector_store.add_documents(chunks)
    
    query = input("What is your query? :")
    
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash',
        temperature = 0.2
    )
    
    class SubQ(BaseModel):
        id: int = Field(description='ID of the subquestiom')
        question: str =Field(description="Sub-question. May refer to earlier ones as their ids like {q1}, {q2}.")
        depends_on: List[int] =Field(default_factory=list,
            description="IDs of sub-questions whose ANSWERS are needed to write this one. Empty if independent.")
    
    class Plan(BaseModel):
        sub_question: List[SubQ] = Field(description='IDs of the subquestions to be answered to answer this question')
    
    structured_llm = llm.with_structured_output(Plan)
    
    template = PromptTemplate(
        template= """
            Split the {query} into the minimum number of sub-questions
            needed to answer it from a video transcript. Don't split if one question is enough.
            
            For each sub-question, apply this test: can it be written completely
            before seeing any other answer?
            - Yes: depends_on = [].
            - No: list the IDs it needs, and refer to their answers as q1, q2.

            Example:
            Q: "Who won and what did they build?"
            1: "Who won the competition?"                   depends_on []
            2: "What did q1 build or demonstrate?"      depends_on [1]
        """,
        input_variables= ['query']
    )
    prompt = template.invoke({
        "query":query
    })
    response1 = structured_llm.invoke(prompt)
    print(response1)
    independent_questions = [
        question
        for question in response1.sub_question if len(question.depends_on) == 0
    ]
    print(independent_questions)
    
    
    
except Exception as e:
    print(e)