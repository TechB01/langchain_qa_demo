import time
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from dotenv import load_dotenv
from langchain import PromptTemplate

st.set_page_config(
    page_title="Generative Question-Answering with LLM",
    page_icon="📘",
    initial_sidebar_state="expanded",
    )

st.markdown('''
        <style>
            #GithubIcon {
                visibility: hidden;
            }
        <style>''',
        unsafe_allow_html=True)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

path = "data"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)

def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=100
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name='langchain-chatbot'
    )
    return doc_db

llm = OpenAI(temperature=0.1, top_p=0.5)
doc_db = embedding_db()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(search_kwargs={'k': 6}, search_type="mmr"),
    # return_source_documents=True
    )
    query = query
    result = qa.run(query)
    return result

def main():
    st.subheader("Generative Question-Answering with LLM")
    uploaded_file = st.file_uploader(label='File upload', label_visibility='hidden', type=['pdf'])
    if uploaded_file is not None:
        st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)
        with open(os.path.join("data",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
            file_sucess = st.success("File uploaded!")
            time.sleep(1) # Wait for 3 seconds
            file_sucess.empty() # Clear the alert
            st.cache_resource.clear()
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
    text_input = st.chat_input("Ask your query...") 
    # if st.button("Ask Query"):
    if text_input:
            st.info("Your Query: " + text_input)
            prompt_template = PromptTemplate.from_template(
                # "{text_input}, Use only given pieces of context to answer the question at the end. If a question is out of the knowledge you politely refuse or If you don't know the answer just say that you don't know, Do not give any extra commentary about it by your own."
                "{text_input}"
            )
            answer = retrieval_answer(prompt_template.format(text_input=text_input))
            st.success(answer)

if __name__ == "__main__":
    main()

    







