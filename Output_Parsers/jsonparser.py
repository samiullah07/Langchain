from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


model = ChatOpenAI()


parser = JsonOutputParser()


template = PromptTemplate(
    template= "Write a name, city and location of {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables= {'format_instructions': parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'topic': 'Atif Aslam'})

print(result)

# prompt = template.format()

# result = model.invoke(prompt)

# print(result.content)