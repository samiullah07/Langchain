from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate


# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="""
You are an AI assistant specialized in explaining advanced research papers in a specific style and depth.

Research Paper: "{user_input}"
Style: "{style_input}"
Explanation Length: "{length_input}"

Provide an explanation of the research paper above based on the selected style and explanation length. 
Make it informative, engaging, and tailored to the audience level specified.
"""
)



prompt_template.save("prompt.json")