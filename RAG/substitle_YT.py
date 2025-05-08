from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


video_id = "2GZ2SNXWK-c"
lang_transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])

transcript = " ".join(chunk["text"] for chunk in lang_transcript)

spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

split_transcript = spliter.create_documents([transcript])


transcript_embed = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(split_transcript,transcript_embed)

# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['6c0f19e9-83d7-4572-9f97-87dbbdf52299']))

retrievers = vector_store.as_retriever(search_type='similarity',search_kwargs={"k":4})

# print(retrievers)

# print(retrievers.invoke("what is AI agents?"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template= """
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,

    input_variables=['context','question']

)



#without chains 
# question = " When you create an AI agent?"

# retriver_doc = retrievers.invoke(question)

# context_text = "n/n/".join(doc.page_content for doc in retriver_doc)

# final_context = prompt.invoke({"context":context_text,"question":question})

# final_result = llm.invoke(final_context)

# print(final_result.content)

# with chains

def format_docs(retriver_doc):
  context_text = "\n\n".join(doc.page_content for doc in retriver_doc)
  return context_text



parallel_chain = RunnableParallel({
    'context' : retrievers | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

final_answer = parallel_chain | prompt | llm | parser

print(final_answer.invoke("summarize the video"))
