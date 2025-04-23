from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


load_dotenv()

model = ChatOpenAI()

template = PromptTemplate(
    template= "Discuss the about the given topic {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'White Dwarf'})

print(result)


chain.get_graph().print_ascii()
