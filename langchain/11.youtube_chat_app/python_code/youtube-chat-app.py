import os
from urllib.parse import urlparse, parse_qs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

os.environ['HF_HOME'] = r"G:\Gen AI\gen-ai\hf_models"

load_dotenv()
def get_video_id(url):
    parsed_url = urlparse(url)
    print('Parse URL:')
    print(parsed_url)
    print('-' * 20)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]

    if parsed_url.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com', 'music.youtube.com'):
        query_params = parse_qs(parsed_url.query)
        print(query_params)
        if 'v' in query_params:
            return query_params['v'][0]


videourl = "https://www.youtube.com/watch?v=okJqVvkQTQ8"
video_id = get_video_id(videourl)
print("Video ID: ", video_id)

youtube_api = YouTubeTranscriptApi()
available_transcripts = list(youtube_api.list(video_id=video_id))

if not available_transcripts:
    print('The video does not have any transcripts')

transcript_fetch = next(
    (
        transcript for transcript in available_transcripts if transcript.language_code in ['en', 'en-US']
    ),
    available_transcripts[0]
)

transcript = transcript_fetch.fetch()
print(transcript.language_code)

loader = YoutubeLoader(
    video_id= video_id,
    language=[transcript.language_code],
)

docs = loader.load()

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)

total_transcript = " ".join(doc.page_content for doc in docs)
print(total_transcript)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.create_documents([total_transcript])

embedding_models = HuggingFaceEmbeddings(
    model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)
vector_store = Chroma(
    collection_name='youtube-chat-app',
    persist_directory= r'G:\Gen AI\gen-ai\youtube-chat-app-rag',
    embedding_function= embedding_models
)

doc_ids = vector_store.add_documents(
    documents=chunks,
    ids=[f'{video_id}-{chunk_number}' for chunk_number in range(len(chunks))]
)

print(doc_ids)


query = "Who won the competetion and what they did ?"

retriever = vector_store.as_retriever(search_type = 'mmr', search_kwargs = {
    'k': min(5, len(chunks)),
    'lambda_mult': 0.5
})

related_docs = retriever.invoke(query)

print(related_docs)
template = PromptTemplate(
    template= """
        You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables= ['context', 'question']
)

context_text = '\n\n'.join(doc.page_content for doc in related_docs)

prompt = template.invoke({
    'question': query,
    'context': context_text
})

llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0.2
)

response = llm.invoke(prompt)
print('-' * 30)
print(response.content)
