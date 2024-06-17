import os 
import streamlit as st 
from langchain_groq import ChatGroq
#Convertin Documents to Chunks 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
#Helps in setting up the context
from langchain.chains.combine_documents import create_stuff_documents_chain
# To Create prompt custom template 
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
#Embedding Vectors to DB
from langchain_community.vectorstores import FAISS
#FAISS internally performs semantic search to give  results 
#Read the PDF files  from folder and then use RecursiveCharacterTextSplitter to Create chunks of data
from langchain_community.document_loaders import PyPDFDirectoryLoader 
# Vector Embbedding Techinqies 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#Load all out enviorment variables 
from dotenv import load_dotenv

load_dotenv()

#Load the GROQ and Google GenAI embeddings API Key from .env file

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("Google_API_KEY")



st.title('Model Document Q&A Chatbot')

#Calling LLM model

llm = ChatGroq(groq_api_key=groq_api_key,model_name = 'Gemma-7b-it')


#Settting up the Prompt Template 

prompt = ChatPromptTemplate.from_template(

"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

"""
-->Creating a vector embbeding from where we will be reading all the PDF 
file. 
-->Converting them into Chunks
--> Applying vector embeddings
-->Storing in vectorDB in FAISS and keep it in server state so that we 
can use it anywhere we requires

"""

def vector_embbedding():

    if 'vectors' not in st.session_state:
        #Define Embeddings 
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model ="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census-20240616T142716Z-001") #Data Ingestion phase 
        st.session_state.docs = st.session_state.loader.load() #Loading Documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session.state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs) #Converting to chunks 
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings) # Creating Vector Embedding


prompt1 =st.text('What you want to ask from the document')

"""
The button code refers to when we click this button the entire vector_embbedding process should work 
and after vector embedding is created we will just give a prompt telling Vector DB created as from this vector 
store DB we will be querying  anything so it should be ready to use

"""

if st.button('Creating Vector Store'):
    vector_embbedding()
    st.write('Vector store  DB is ready ')


## Let us record the time also 

import time 

if prompt1:
    document_chain =  create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()#Creates a interface to store and retrive question asked 
    retrieval_chain = create_retrieval_chain(retriever,document_chain) # Creating a chain so as we required it to run in change

    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response['answer'])


    # With a streamlit expander we are trying to get the context also with the answer 
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
