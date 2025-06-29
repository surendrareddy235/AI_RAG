from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

loader = PyPDFLoader('surya.pdf')
document = loader.load()

splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
text = splitter.split_documents(document)
# for chunk in text:
#     print(chunk.page_content)

embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(text, embedding_model)

vector_embeddings = vector_store.index
print(vector_store.index.reconstruct(0))


