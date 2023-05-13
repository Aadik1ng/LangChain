from youtube_transcript_api import YouTubeTranscriptApi
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import textwrap
import os
import openai
import os
from pytube import Playlist
from art import *


print(os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
print(os.getenv("OPENAI_API_KEY"))


# create a function to get urls form list of playlist
def get_playlist(playlists):
    urls = []
    # iteratively get watch links from playlist
    for playlist in playlists:
        playlist_urls = Playlist(playlist)
        for url in playlist_urls:
            urls.append(url)
    return urls


playlist = ["playlist URL"]
pl_urls = get_playlist(playlist)
print("Urls successfully saved into " + os.getcwd())
p = []
embeddings = OpenAIEmbeddings()
file1 = open(r"path. txt", "r")
Lines = file1.readlines()
# Strips the newline character
for line in Lines:
    p.append(line.strip())


playlist = ["playlist URL"]
pl_urls = get_playlist(playlist)
print("Urls successfully saved into " + os.getcwd())
p = []
embeddings = OpenAIEmbeddings()
file1 = open(r"path. txt", "r")
Lines = file1.readlines()
# Strips the newline character
for line in Lines:
    p.append(line.strip())
count = 1
# Assume that the embeddings list has at least one element


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    video_id = video_url.split("v=")[1]
    loader = YoutubeLoader.from_youtube_url(video_id)
    print(loader.video_id)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    try:
        db = FAISS.from_documents(docs, embeddings)
    except:
        print("Subtitles not found/n")
        return
    return db


def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, ask for more information like "Give me mor info".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


for i in p:
    print("The ", count, " url", i)  # URL COunt
    db = create_db_from_youtube_video_url(i)
    query = "Query"
    response, docs = get_response_from_query(db, query)
    print(textwrap.fill(response, width=85))
    count += 1


db.save_local(r"Save Path")
db = FAISS.load_local("Save Path.pkl", OpenAIembeddings, "Save Path.faiss")


print(db.similarity_search("Query"))
