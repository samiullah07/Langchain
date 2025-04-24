from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv


retriver = WikipediaRetriever(top_k_results=2,lang='en')

query = "Who is spouse of Nouman Ijaz"

result = retriver.invoke(query)

for i,doc in enumerate (result):
    print(f'----/ Result {i+1}')
    print(f'details --- / {doc.page_content}')

