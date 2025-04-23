from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI()
load = TextLoader('poem.txt',encoding='utf-8')
docs = load.load()
template = PromptTemplate(
    template = "Write a summary on given text {poem}",
    input_variables=['poem']
)

parser = StrOutputParser()

chain = template | model | parser
result = chain.invoke({'poem':docs[0].page_content})

print(result)
