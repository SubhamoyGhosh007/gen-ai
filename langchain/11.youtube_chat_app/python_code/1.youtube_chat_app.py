from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApiException

video_id = "J5_-l7WIO_w"

try:
    loader = YoutubeLoader(
        video_id,
        language=['en', 'en-US'],
        add_video_info=False
    )
    
    docs = loader.load()
    for i, doc in enumerate(docs):
        print("Data for ", i)
        print(doc.page_content[:500])
        print(doc.metadata)
        print("-" * 20)
except YouTubeTranscriptApiException as e:
    print(f"Transcript unavailable: {e}")
