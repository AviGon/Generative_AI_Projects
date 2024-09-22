import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

#Setup your streamlit website
st.set_page_config(page_title="PDF question answering system",page_icon="🧮")
st.title("PDF Answering Bot")

groq_api_key = st.sidebar.text_input(label="Groq API Key",value="",type="password")

if not groq_api_key:
    st.error("Please enter the groq api key")

if groq_api_key:
    model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

    pdf = st.file_uploader('Upload a PDF file with text in English. PDFs that only contain images will not be recognized.', type=['pdf']) 

    query = st.text_input('Ask question about the PDF you entered!', max_chars=300)

if pdf is not None and query:
    if st.button("Answer"):
        try:
            with st.spinner("Generating.."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf.read())
                    tmp_file_path = tmp_file.name

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                loader = PyPDFLoader(tmp_file_path)

                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                splits = text_splitter.split_documents(docs)

                vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())

                retriever = vectorstore.as_retriever()

                system_prompt = (
                f"You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know and keep the answer concise."
                "\n\n"
                "{context}"
                )

                prompt = ChatPromptTemplate.from_messages(
                [
                ("system", system_prompt),
                ("human", "{input}"),
                ]
            )
                
                question_answer_chain = create_stuff_documents_chain(model, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                results = rag_chain.invoke({"input": f"{query}"})

                st.success(results['answer'])

        except Exception as e:
            st.exception(f"Exception: {e}")








