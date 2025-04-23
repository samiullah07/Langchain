from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


model = ChatOpenAI()

# st.header("Chatbot Text Model")
chat_history = []

chat_history.append(SystemMessage(content="You are AI Assistant"))

while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break

    ai_output = model.invoke(chat_history)
    chat_history.append(AIMessage(content = ai_output.content))
    print("AI : " , ai_output.content)
    # st.write(ai_output.content)


print(chat_history)


    




