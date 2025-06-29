Simple RAG System with LangChain, FAISS, and Gemini API

This project builds a simple Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, and a local Hugging Face embedding model. It extracts text from a PDF file, splits it into smaller chunks, generates embeddings for each chunk, and stores them in a FAISS vector store. This setup can be used for question-answering systems where a user query retrieves the most relevant document chunks based on similarity.

Requirements

These are the Python libraries required for this project:

langchain – the core framework that manages chaining components together
langchain-community – for community-supported document loaders and vector stores
langchain-huggingface – to use Hugging Face embedding models
sentence-transformers – provides access to models like all-MiniLM-L6-v2
faiss-cpu – fast vector similarity search engine
google-generativeai – to use Gemini as your LLM (optional in this stage)
unstructured – helps LangChain extract and parse content from PDFs and other documents

Install everything using the following command:

pip install -r requirements.txt

Imports Used

These are the required Python modules imported in your script:

PyPDFLoader from langchain_community loads PDF files into structured LangChain documents

CharacterTextSplitter breaks long text into manageable chunks for embedding

HuggingFaceEmbeddings loads and runs the local sentence-transformers model

FAISS stores embeddings for fast similarity search

numpy is used optionally if you want to inspect raw vectors

Step 1: Load the PDF

PyPDFLoader reads the contents of your PDF file, page by page, and converts it into a list of Document objects. These documents contain both the raw text and metadata such as page number and source. You loaded "surya.pdf" using this loader.

Step 2: Split Text into Chunks

CharacterTextSplitter breaks the raw document text into smaller pieces.

chunk_size=50: splits the document into parts of about 50 characters

chunk_overlap=10: ensures that 10 characters from the end of one chunk are repeated at the start of the next. This overlap helps maintain the context across chunks and prevents meaning loss during retrieval.

The result is a list of small text chunks stored in the variable text.

Step 3: Embed Chunks and Store in FAISS

HuggingFaceEmbeddings loads the model all-MiniLM-L6-v2 from your local Hugging Face cache and generates embeddings for each chunk in text. These embeddings are then stored in a FAISS index using FAISS.from_documents.

This makes your text searchable. When a user asks a question, you can retrieve the most similar text chunks based on the vector similarity.

Step 4: Access the Raw Embeddings (Optional)

If you want to check the actual embedding vectors stored in FAISS, you can access the underlying FAISS index object (vector_store.index) and use the .reconstruct() method to print the raw vector of any chunk by its index.

Next Step