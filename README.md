🔹 Imports
from langchain_community.document_loaders import PyPDFLoader
→ Loads and reads content from a PDF file.

from langchain.text_splitter import CharacterTextSplitter
→ Splits long text into smaller overlapping chunks.

from langchain_community.embeddings import HuggingFaceEmbeddings
→ Generates vector embeddings using a Hugging Face model.

from langchain_community.vectorstores import FAISS
→ Stores and retrieves vectors using the FAISS similarity search library.

import numpy as np
→ Imports NumPy (not used in this script directly, but commonly needed with embeddings).

from langchain.chains import RetrievalQA
-> used to chaining the answers and question embedding

from langchain_google_genai import ChatGoogleGenerativeAI
->this will you with interacting with google models
import os
-> in this project os will help you to get the env 

from dotenv import load_dotenv
->dotenv is where your api is stored than with the help of os you can get the api key

🔹 PDF Loading
loader = PyPDFLoader('surya.pdf')
→ Initializes the PDF loader with your file.

document = loader.load()
→ Loads and parses the PDF into a list of document objects.

🔹 Text Splitting
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
→ Sets the chunk size to 50 characters with a 10-character overlap between chunks.

text = splitter.split_documents(document)
→ Splits the entire PDF content into manageable text chunks.

🔹 Embedding Generation
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
→ Loads a lightweight and efficient sentence embedding model.
-> it is a sentence level embedding model it will create a vector size of 384 for each chunk i told you that chunk mean a sentence if a sentence have 1 word nun the less this model creates 384 vector size the same size of a full sentence with 50 words 


🔹 Vector Store Creation
vector_store = FAISS.from_documents(text, embedding_model)
→ Converts the chunks into embeddings and stores them in a FAISS vector index.

🔹 Accessing the FAISS Index
vector_embeddings = vector_store.index
→ Retrieves the raw FAISS index which contains all the vector embeddings.
-> A single vector of size 384 for each chunk.

 <!-- LLM Setup – Gemini API -->

    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GEMINI_API_KEY'))
    → Initializes the Gemini LLM using your API key stored in .env.

🔹 RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

→ Combines your retriever (FAISS) and Gemini LLM into a single question-answering chain.

✔️ It searches the top 2 relevant chunks based on your query.
✔️ It embeds your question internally, fetches relevant data, and sends it to Gemini for answering.

🔹 Continuous Q&A Loop

while True:
    query = input("ask a question (or 'exit'):")
    if query.lower() == "exit":
        break
    response = qa_chain.invoke({'query': query})
    print('gemini:', response["result"])

→ This loop allows you to ask multiple questions continuously.
→ Type 'exit' to quit.
→ Gemini will answer based only on the content inside your PDF.