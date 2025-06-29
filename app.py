from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('jobless.pdf')
document = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=70)
text = splitter.split_documents(document)
# for chunk in text:
#     print(chunk.page_content)

embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(text, embedding_model)

vector_embeddings = vector_store.index
# print(vector_store.index.reconstruct(0))

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GEMINI_API_KEY'))

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vector_store.as_retriever(search_kwargs={"k":2}),
    return_source_documents = True
    )

while True:
    query = input("ask a question (or 'exit'):")
    if query.lower() == "exit":
        break
    response = qa_chain.invoke({'query':query})
    print('gemini:',response["result"])

