# LangChain: Q&A over Documents

An example might be a tool that would allow you to query a product catalog for items of interest.


```python
#pip install --upgrade langchain
```


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```

Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.


```python
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
```


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
```


```python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
```


```python
from langchain.indexes import VectorstoreIndexCreator
```


```python
#pip install docarray
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
```


```python
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
```

**Note**:
- The notebook uses `langchain==0.0.179` and `openai==0.27.7`
- For these library versions, `VectorstoreIndexCreator` uses `text-davinci-003` as the base model, which has been deprecated since 1 January 2024.
- The replacement model, `gpt-3.5-turbo-instruct` will be used instead for the `query`.
- The `response` format might be different than the video because of this replacement model.


```python
llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)
```


```python
display(Markdown(response))
```

## Step By Step


```python
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)
```


```python
docs = loader.load()
```


```python
docs[0]
```


```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```


```python
embed = embeddings.embed_query("Hi my name is Harrison")
```


```python
print(len(embed))
```


```python
print(embed[:5])
```


```python
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
```


```python
query = "Please suggest a shirt with sunblocking"
```


```python
docs = db.similarity_search(query)
```


```python
len(docs)
```


```python
docs[0]
```


```python
retriever = db.as_retriever()
```


```python
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
```


```python
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

```


```python
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

```


```python
display(Markdown(response))
```


```python
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```


```python
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
```


```python
response = qa_stuff.run(query)
```


```python
display(Markdown(response))
```


```python
response = index.query(query, llm=llm)
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
```

Reminder: Download your notebook to you local computer to save your work.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
