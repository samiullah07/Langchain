from typing import TypedDict,Annotated,Optional,Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    pros : Annotated[Optional[list[str]],"All Pros of the text"]
    cons : Annotated[list[str],"All Cons of the text"]
    summary : Annotated[str, "Brief summary of the text"]
    sentiment : Annotated[str,"Sentiment of the text"]




structure_review = model.with_structured_output(Review)
result = structure_review.invoke("The shoes are okay overall — they look nice and the design is modern, which I appreciate. They fit fairly well, but the comfort level isn’t as high as I expected, especially if you're on your feet for long periods. The build quality seems average; not bad, but not exceptional either. For the price, they do the job, but I think there are better options out there if you're looking for long-lasting comfort and durability.")
print(result)