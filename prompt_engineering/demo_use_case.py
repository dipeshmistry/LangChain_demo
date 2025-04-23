import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="My Project Guide", page_icon="",layout="wide")
st.title("My Project Guide")

# Load LLM & Retriever (cache to avoid reloads)
@st.cache_resource
def setup_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"])

    project_context = """# Product Catalogue API

## 📌 Project Overview

This project is a **Product Catalogue API** designed for an eCommerce platform. It follows a **microservices architecture** using **Node.js, Express.js, PostgreSQL, and Sequelize**. The API allows managing **categories, products, and variants** with full CRUD operations.

"""



    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful assistant answering questions from a Given Project data. If question come out of the context, say 'i dont know'"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    rag_chain = RunnableMap({
        "context": lambda x: "\n\n".join(project_context).join("\n\n").join([doc.page_content for doc in retriever.get_relevant_documents(x["question"]) if doc.page_content]),
        "question": lambda x: x["question"],
        "history": lambda x: x["chat_history"]
    }) | prompt | llm

    return rag_chain

rag_chain = setup_chain()

# Chat history state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask a question about the survey...")

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Button to view full chat history
if st.button("View Full Chat History"):
    with st.expander("Full Chat History", expanded=True):
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**User**: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"**Assistant**: {msg.content}")

# On user input
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                st.markdown(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error: {e}")
