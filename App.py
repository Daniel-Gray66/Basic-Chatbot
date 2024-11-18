# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS  # Use FAISS for vector search
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Replace with your preferred LLM
import streamlit as st

# Set up the Streamlit app title
st.title("Chat with Your PDF")

# Option to use a file uploader or load a predefined file
use_predefined_file = st.checkbox("Use a predefined PDF file instead of uploading")

if use_predefined_file:
    # Option 1: Use a predefined PDF file in the directory
    pdf_path = "C:/Personal Projects/ChatBot/tsla-Earings20240723.pdf"  # Replace with whatever file you want 
    try:
        st.write(f"Loading the predefined file: `{pdf_path}`...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except FileNotFoundError:
        st.error(f"The file `{pdf_path}` was not found. Please ensure it is in the same directory as this script.")
        documents = None
else:
    # Option 2: Upload a PDF file through Streamlit
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        st.write("Loading the uploaded document...")
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
    else:
        documents = None

if documents:
    try:
        # Step 1: Split the document into smaller chunks
        st.write("Splitting the document into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        st.write(f"Document successfully split into {len(docs)} chunks!")

        # Step 2: Create vector embeddings using FAISS
        st.write("Creating vector index...")
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.write("Vector index successfully created!")

        # Step 3: Set up retrieval-based QA
        st.write("Setting up the RetrievalQA system...")
        retriever = vectorstore.as_retriever()  # FAISS supports this method
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(model="gpt-3.5-turbo"),  # Updated to use a supported model
            retriever=retriever
        )
        st.write("System ready! You can now ask questions.")

        # Step 4: User input for chatbot interaction
        prompt = st.text_input("Enter your question:")
        if prompt:
            st.write("Processing your question...")
            try:
                response = qa_chain.run(prompt)
                st.write("Answer:", response)
            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")
    except Exception as e:
        st.error(f"An error occurred during setup: {e}")
else:
    st.write("Please upload a PDF file or ensure the predefined file is available.")
