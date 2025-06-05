from langchain.chat_models import ChatOpenAI

def basic_query(prompt: str):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
    return llm.predict(prompt)