from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core import output_parsers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI()

template2 = PromptTemplate(
    template="Write a 5 lines summary on the following text. /n {text} ",
    input_variables=["text"]
)

prompt1 = template1.invoke({'topic':'Black Whole'})

# result1 = model.invoke(prompt1)

# prompt2 = template2.invoke({'text':result1.content})

# result2 = model.invoke(prompt2)
# print(result1.content)
# print(result2.content)


parse = StrOutputParser()

chain = template1 | model | parse | template2 | model | parse

result = chain.invoke({"topic":"Glaxies"})

print(result)