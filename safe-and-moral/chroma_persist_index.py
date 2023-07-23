from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.document_loaders import DirectoryLoader
from pprint import pprint

loader = DirectoryLoader('data', glob="*.txt")
data = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(collection_name="langchain_store",
                     embedding_function=embeddings,
                     persist_directory="./tmp")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separators=["\n"]
)

texts = text_splitter.split_documents(data)
texts = list(map(lambda x: x.page_content.replace("\n",""), texts))
texts = list(filter(lambda x: len(x) > 10, texts))

#pprint(texts)

# Add data to the vector store
vectorstore.add_texts(texts)

# Persist the data to disk
vectorstore.persist()

