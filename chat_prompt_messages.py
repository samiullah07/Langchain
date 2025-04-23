from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


model = ChatOpenAI()
chat_prompt = ChatPromptTemplate([
    ("system", "You are Powerfull Power BI {domain}"),
    ("human","Who invented {topic}")
])

messages = chat_prompt.invoke({"domain" : "Cricket","topic" : "dusra"})

result = model.invoke(messages)

print(result.content)