from langchain.text_splitter import TextSplitter,RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()
loader = PyPDFLoader('dl-curriculum.pdf')
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=0,
)

prompt = PromptTemplate(
    template= "Make questions and answers on the following Text {text}",
    input_variables=['text']
)

file = loader.load()

splitter_text = splitter.split_documents(file)

model= ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text':splitter_text[0].page_content})

print(result)



