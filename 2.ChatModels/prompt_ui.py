from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3")

st.header('Research Tool')
model = ChatOpenAI(model="gpt-4")
user_input = st.selectbox("Select Research Paper",["Attension all you need !!","BERT : Pre-Training of Deep Bidirectional Transformer","GPT-3: Language Models are Few-Short Learners","Diffusions Models beat GANS on Image Synthesis"])
style_input = st.selectbox("Select the Standard",["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])
length_input = st.selectbox("Select Explanation Length",["Short (1-2 Paragraphs)","Medium (4-5 Paragraphs)","Long (Detailed Explanation)"])

prompt_template = load_prompt('2.ChatModels/prompt.json')

# Format the prompt with user selections
final_prompt = prompt_template.invoke({
    "user_input":user_input,
    "style_input":style_input,
    "length_input":length_input
})

if st.button("Summarize"):
    result = model.invoke(final_prompt)
    st.write(result.content)
