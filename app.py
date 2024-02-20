import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

DB_FAISS_PATH = '/home/rahul/Desktop/LLM_Project/Gemini_with_Pdf/llm_project_with_Gemini_pdf/vectorstores/db_faiss'


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_text(text=text_chunks, embeddings=embeddings)
    vector_store.save_local('faiss_index')


def get_conversation_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to 
    provide  all the details, if the answer is not in provided context just say, "answer is not
    available in the context", do not provide the wrong answer\n\n
    Context: \n{context}?\n
    Question: \n{question}\n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff",
                          prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings)
    print('------')
    print(new_db)
    print('--------')
    docs = new_db.similarity_search(user_question)
    print(docs)
    chain = get_conversation_chain()

    response = chain(
        inputs={"input_documents": docs,
                "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response['output_text'])


def main():

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini-Pro")

    user_question = st.text_input("Ask a Question from the PDF File")

    if user_question:
        user_input(user_question)

    # with st.sidebar:
    #     st.title('Menu:')
    #     pdf_doc = st.file_uploader("Upload your pdf file and click on the submit button")
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing...."):
    #             raw_text = get_pdf_text(pdf_docs=pdf_doc)
    #             text_chunks = get_text_chunks(raw_text)
    #             get_vector_store(text_chunks=text_chunks)
    #             st.success("Done")


if __name__ == "__main__":
    main()



