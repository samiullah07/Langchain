from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4")

result = chat_model("Who is Prevaiz Musharaf?")

print(result)