from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel


load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

template1 = PromptTemplate(
    template= "Generate the notes from the following text {text}",
    input_variables=['text']
)

template2 = PromptTemplate(
    template="Generate the questions answers from the following notes {text}",
    input_variables=['text']
)

template3 = PromptTemplate(
    template = 'Merge the notes and questions into single document notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

runablePrompt = RunnableParallel({
    'notes': template1 | model1 | parser,
    'quiz' : template2 | model2 | parser
}
)

marge_prompt = template3 | model1 | parser
chain = runablePrompt | marge_prompt 


text = """
Time travel is the fascinating idea of moving backward or forward through time, 
beyond the natural flow we experience every day. Often imagined in science fiction,
it allows people to revisit the past or explore the future, unlocking endless possibilities and mysteries.
Scientists like Albert Einstein introduced theories, such as relativity, which suggest that under extreme conditions —
like moving at the speed of light or being near a black hole — time can bend and behave differently. Although time travel
remains unproven in real life, it sparks imagination and deep questions about destiny, choices, and the nature of reality.
From books and movies to physics debates, time travel continues to inspire both curiosity and wonder, offering a glimpse into 
worlds where the past and future are never truly out of reach."""


result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()