import openai
from pypdf import PdfReader
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from icecream import ic
import streamlit as st
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# Set your OpenAI API key
openai.api_key = api_key
lc_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key,
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    dimensions=1024
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key
    # base_url="...",
    # organization="...",
    # other params...
)
# Function to extract text from a single PDF using pypdf
def extract_text_from_pdf(pdf_path):
    text_data = []
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:  # Ensure text is not None
            text_data.append(text)
    return text_data

# Function to extract text from multiple PDFs
def extract_texts_from_pdfs(pdf_paths):
    all_texts = []
    for pdf_path in pdf_paths:
        texts = extract_text_from_pdf(pdf_path)
        all_texts.extend(texts)
    return all_texts

# List of PDF files to process
pdf_paths = ['./data/doc2.pdf']  # Replace with your PDF file paths

# Extract text from all the PDFs
texts = extract_texts_from_pdfs(pdf_paths)

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    # Chunking texts for batching as the embedding request may have a token limit
    batch_size = 1000  # Adjust batch size as needed
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        # response = openai.Embedding.create(
        #     input=chunk,
        #     model="text-embedding-ada-002"  # Choose an appropriate embedding model
        # )
        # Get the embeddings for the texts
        embedding_vectors = lc_embeddings.embed_documents(texts)
        embeddings.extend([vector for text, vector in zip(texts, embedding_vectors)])
    return np.array(embeddings)

# Get embeddings for all the text extracted from the PDFs
doc_embeddings = get_embeddings(texts)

# Create a FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)
index.add(doc_embeddings)

# Function to get the most relevant text from the indexed documents
def retrieve_documents(query, k=3):
    query_embedding = get_embeddings([query])[0].reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Function to generate a response using GPT-4
def generate_response(query):
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    print("*"*20)
    # ic(context)
    # ic("*"*20)
    prompt = """Important: Answer the question asked by the user from the context provided below. Do not answer anything outside the 
    provided context. If you do not find and answer in the provided context then say "I don't know".
    Context: {context}\n\nQuestion: {query}\n\nAnswer:"""
    


    response = llm.invoke(query).content
    # print(response)
    # Print the response from the model
    # print(response['choices'][0]['message']['content'])
    
    return response.strip()

# Example usage
# query = "give Definitions of marketing: According to kotler in 3-4 lines"
# response = generate_response(query)
# print(response)

# Streamlit app starts here
st.title('PDF Information Retrieval System')

# uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
query = st.text_input("Enter your query here:")
search_button = st.button("Search")

if query and search_button:
    # doc_embeddings = get_embeddings(texts)
    # dimension = doc_embeddings.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    # index.add(doc_embeddings)

    # query_embedding = get_embeddings([query])[0].reshape(1, -1)
    # distances, indices = index.search(query_embedding, 3)  # Retrieving top 3 relevant texts
    # results = [texts[i] for i in indices[0]]
    response = generate_response(query)
    st.write("Results:")
    st.text(response)
    # for result in results:
        
else:
    st.write("Please upload some PDF files and enter a query to start the search.")