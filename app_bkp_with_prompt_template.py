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
from dotenv import load_dotenv
from langchain import PromptTemplate

load_dotenv()


PINECONE_API_KEY = '6c72c863-bd58-4d1d-bb76-f137e62386c4'
PINECONE_ENV = 'us-west1-gcp-free'
OPENAI_API_KEY = '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'

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
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

llm = OpenAI(temperature=0.6)
doc_db = embedding_db()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result

def main():
    st.title("Question and Answering App powered by LLM and Pinecone")
    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            prompt_template = (
                PromptTemplate.from_template("{text_input}")
                + ", Use only given pieces of context to answer the question at the end. If a question is out of the knowledge, you politely refuse or If you don't know the answer, just say that Based on the provided context I cannot answer this question"
                + ", Do not give any extra commentary about it by your own."
            )
            # prompt_template = PromptTemplate.from_template(
            #     "{text_input}, Use only given pieces of context to answer the question at the end. If a question is out of the knowledge, you politely refuse or If you don't know the answer, just say that Based on the provided context I don't know the answer, Do not give any extra commentary about it by your own."
            # )
            answer = retrieval_answer(prompt_template.format(text_input=text_input))
            # answer = retrieval_answer(prompt_template.format(language="spanish"))
            st.success(answer)


if __name__ == "__main__":
    main()

    







