from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import  Optional, List,Literal
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI()
class Review(BaseModel):
    key_Themes : list[str] = Field(description="Write down all key themes discused in review in a list")
    sentiment : Literal["pos","cons"] =Field(description="Return sentiment of the review either positve or negative or neutral")
    summary : str = Field(description= "Brief summary of the text")

    pros : list[str] = Field(description="List of all pros in the review")
    cons : list[str] = Field(description="List of all cons in the review")



pydantic_review = model.with_structured_output(Review)
result = pydantic_review.invoke("The shoes are okay overall — they look nice and the design is modern, which I appreciate. They fit fairly well, but the comfort level isn’t as high as I expected, especially if you're on your feet for long periods. The build quality seems average; not bad, but not exceptional either. For the price, they do the job, but I think there are better options out there if you're looking for long-lasting comfort and durability.")
print(result.summary)



