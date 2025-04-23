from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(description="Age of the Person")
    bio : str = Field(description="Biography of the person")
    struggle : str = Field(description = "Struggling face of the person")


parser = PydanticOutputParser(pydantic_object=Person)
model = ChatOpenAI()

template = PromptTemplate(
    template = "Short Detail of this person {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'King of Bollywood'})

print(result)