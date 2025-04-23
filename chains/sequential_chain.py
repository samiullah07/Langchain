from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


load_dotenv()

model = ChatOpenAI()

template1 = PromptTemplate(
    template= "Discuss the about the given topic {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = 'Write 5 lines summary in points of Given text {text}',
    input_variables=['text']

)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'White Dwarf'})

print(result)


chain.get_graph().print_ascii()
