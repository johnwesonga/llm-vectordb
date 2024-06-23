import traceback
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from llama_index.core import query_engine, retrievers
from langchain_openai import ChatOpenAI

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os
from pathlib import Path
import openai
import streamlit as st
import faiss
from streamlit_chat import message

#openai.api_key = os.environ["OPENAI_API_KEY"]

def handle_session_state():
    """Initialize the session state if not already done."""
    st.session_state.setdefault("generated", [])
    st.session_state.setdefault("past", [])
    st.session_state.setdefault("q_count", 0)
   


def initialize_llm():
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    )
    return llm

def load_data(file_path):
    """Load movie data from a CSV file."""
    loader = CSVLoader(
        file_path=file_path, encoding="utf-8", csv_args={"delimiter": ","}
    )
    data = loader.load()
    return data

def create_embeddings():
    return OpenAIEmbeddings(model='text-embedding-3-small')

def create_index(documents, embedding):
    """Create and return db store."""
    if "index" not in st.session_state:
        path = Path("faiss_store")
        if path.exists():
            index = FAISS.load_local("faiss_store", embedding,
                                     allow_dangerous_deserialization=True)
        else:
            index = FAISS.from_documents(
                    documents,
                    embedding,
                )
            index.save_local("faiss_store")
        st.session_state["index"] = index
    return st.session_state["index"]

def create_query_engine(index):
    """Create and return a query engine instance."""

    if "query_engine" not in st.session_state:
        retriever = index.as_retriever()
        st.session_state["query_engine"] = retriever
    return st.session_state["query_engine"]

def form_callback():
    """Handle the form callback."""
    st.session_state["input_value"] = st.session_state["input"]

def get_text(q_count):
    """Display a text input field on the UI and return the user's input."""
    label = "Type a question about a movie: "
    value = "Who directed the movie Jaws?\n"
    return st.text_input(label=label, value=value, key="input", on_change=form_callback)

def query(payload, llm, retriever):
    """Process a query and return a response."""
    query = payload["inputs"]["text"]

    template = """
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = llm

    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke(query)
    # Transform response to string and remove leading newline character if present
    return str(response).lstrip("\n")


def main():
    st.title("LangChain")
    st.markdown("### Welcome to the Rotten Tomatoes Movies Bot")
    user_openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", placeholder="sk-...", type="password")
    if user_openai_api_key:
        openai.api_key = user_openai_api_key
        st.write(
            "Has environment variables been set:",
            user_openai_api_key == st.secrets["OPENAI_API_KEY"],
            )
        try:        
            handle_session_state()
            documents = load_data(Path("rotten_tomatoes_movies.csv"))
            embeddings = create_embeddings()
            index = create_index(documents, embeddings)
            query_engine = create_query_engine(index)
            llm = initialize_llm()
            # Get the user input
            user_input = get_text(q_count=st.session_state["q_count"])
            if user_input and user_input.strip() != "":
                output = query(
                    {
                        "inputs": {
                            "text": user_input,
                        }
                    },
                    llm,
                    query_engine,
                )
                # Increment q_count, append user input and generated output to session state
                st.session_state["q_count"] += 1
                st.session_state["past"].append(user_input)
                if output:
                    st.session_state["generated"].append(output)
                if st.session_state["generated"]:
                    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                        message(st.session_state["past"][i], is_user=True, key=f"{str(i)}_user")
                        message(st.session_state["generated"][i], key=str(i))
   
        except Exception as e:
            tb = traceback.format_exc()
            #print(tb)
            st.error(f"Something went wrong: {e}\n\nTraceback:\n{tb} Please try again.")

    else:
        st.warning("Please enter your OpenAI API key to proceed.")
         

if __name__ == "__main__":
    main()