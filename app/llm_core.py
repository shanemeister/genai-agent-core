from langchain_openai import ChatOpenAI  # newer module
def basic_query(prompt: str):
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    return llm.invoke(prompt)