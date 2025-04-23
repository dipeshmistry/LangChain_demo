import os
from dotenv import load_dotenv
from pinecone import 

from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import ChatOpenAI ,OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
print(OPENAI_API_KEY)
print(INDEX_NAME)
# 2. Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(INDEX_NAME)

# 3. Connect to existing index
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index,
    embedding=embeddings,
    text_key="text"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 4. Initialize the Chat model
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY
)

# 5. Build a conversational retrieval chain with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# 6. Chat loop for insights

def chat_loop():
    print("Welcome to Survey Insights Chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = qa_chain({"question": user_input})
        answer = result.get("answer", "No answer generated.")
        print(f"Assistant: {answer}\n")

        # Optionally print source docs
        for doc in result.get("source_documents", []):
            print(f"【source】 {doc.metadata.get('source', '')}: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    chat_loop()
