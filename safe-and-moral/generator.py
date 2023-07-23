# Copyright 2023 Mark Watson. All rights reserved.

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI
llm = OpenAI(temperature=0.9)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(collection_name="langchain_store",
                     embedding_function=embeddings,
                     persist_directory="./tmp")

def get_help(thing_to_do):
    results = vectorstore.similarity_search(thing_to_do, k=3)
    context = " ".join(list(map(lambda x: x.page_content.replace("\n",""), results)))
    prompt=f"Given the context:\n{context}\n\nPlease give me moral advice and guidance for {thing_to_do}?"
  
    print(f"\n{prompt}:")
    return llm(prompt)

print(get_help("I want to be fair to my friend"))
print(get_help("My business partner is stealing from me"))

