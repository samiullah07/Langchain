from langchain.vectorstores import Chroma, Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


from langchain_core.documents import Document

docs = [
    Document(
        page_content="Kamran Akmal was the top batsman in PSL 3, scoring 425 runs in the season. He played for Peshawar Zalmi and was known for his explosive starts at the top of the order.",
        metadata={"player": "Kamran Akmal", "role": "Batsman", "team": "Peshawar Zalmi", "season": "PSL 3"}
    ),
    Document(
        page_content="Babar Azam scored 402 runs in PSL 3 while playing for Karachi Kings. His consistency and classical technique made him one of the standout performers.",
        metadata={"player": "Babar Azam", "role": "Batsman", "team": "Karachi Kings", "season": "PSL 3"}
    ),
    Document(
        page_content="Faheem Ashraf was the leading wicket-taker in PSL 3 with 18 wickets. He represented Islamabad United and played a vital role in their championship-winning campaign.",
        metadata={"player": "Faheem Ashraf", "role": "Bowler", "team": "Islamabad United", "season": "PSL 3"}
    ),
    Document(
        page_content="Wahab Riaz picked up 18 wickets in PSL 3 for Peshawar Zalmi. His fiery pace and aggressive bowling made him a constant threat for opposition batsmen.",
        metadata={"player": "Wahab Riaz", "role": "Bowler", "team": "Peshawar Zalmi", "season": "PSL 3"}
    ),
    Document(
        page_content="In PSL 3, Islamabad United won the title under Misbah-ul-Haq's captaincy. Their strong bowling lineup and top-order stability were key factors in their success.",
        metadata={"team": "Islamabad United", "achievement": "Champion", "season": "PSL 3"}
    )
]

# embedding = OpenAIEmbeddings()
# index_name = "psl-index"
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(index_name, dimension=1536)

# # Connect to index
# index = pinecone.Index(index_name)

# vector_store = Pinecone.from_documents(
#     docs,embedding,
# )

vector_stores = Chroma(
    embedding_function= OpenAIEmbeddings(),
    persist_directory="chrome_db",
    collection_name='sample'
)

J = vector_stores.add_documents(docs)

K = vector_stores.get(include=["embeddings","documents","metadatas"])

search = vector_stores.similarity_search(
    query='Bowlers in these',
    k=2
)

score =  vector_stores.similarity_search_with_score(
    query='Bowlers in these',
    k=1
)

print(score)