# from langchain_openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()


# openai = OpenAI(model="gpt-3.5-turbo-instruct")

# result = openai.invoke("Capital of Pakistan and Africa?")

# print(result)

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')  # or 'claude-2' / 'claude-1'
response = model.invoke("Hello!")
print(response.content)
