from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel,RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model = ChatOpenAI()


class Review(BaseModel):
    sentiment :Literal["positive",'negative'] = Field(description="Describe the sentiment of the feedback")


parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template="Classify the following feedback into positive or negative {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables= {"format_instruction":parser2.get_format_instructions()}
    

)


classify_feedback = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template= "Write down the appropriate Response about positive feedback {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "Write down the appropriate response about negative feedback {feedback} ",
    input_variables=['feedback']

)

branch_feedback = RunnableBranch(
    (lambda x:x.sentiment =='positive',prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative',prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not find sentiment')
)


chain = classify_feedback | branch_feedback


result = chain.invoke({'feedback':'The overall experience was fair — not outstanding, but not disappointing either. The product worked as expected, though there were a few minor issues that could be improved.'})


print(result)

chain.get_graph().print_ascii()