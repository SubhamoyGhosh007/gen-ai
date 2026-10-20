from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 1.5)
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 1.5)
model3 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0.7)

res1 = model1.invoke("Write a 5 line poem on cricket")
res2 = model2.invoke("Write a 5 line poem on cricket")
res3 = model3.invoke("Write a 5 line poem on cricket")

print(f"Reponse from model 1: \n {res1.content}")
print("-"*50)
print(f"Reponse from model 2: \n {res2.content}")
print("-"*50)
print(f"Reponse from model 3: \n {res3.content}")