from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

schema = [
    ResponseSchema(name='Fact 1',description="Write fact 1 about the topic"),
    ResponseSchema(name='Fact 2',description="Write fact 1 about the topic"),
    ResponseSchema(name='Fact 3',description="Write fact 1 about the topic"),
    ResponseSchema(name='Fact 4',description="Write fact 1 about the topic")



]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= "Write 5 facts summary of this {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables= {'format_instruction':parser.get_format_instructions()}

    
)


chain = template | model | parser

result = chain.invoke({'topic':'black Whole'})

print(result)