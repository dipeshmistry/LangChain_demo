# injection to pinecone database
import os
from dotenv import load_dotenv
load_dotenv()

# load different types of documents; check langchain/components in docs for futher info like csvloader,whatsapploader etc
from langchain_community.document_loaders import TextLoader

# creates chunks for further proccessing as diff models have diff size of token limits
from langchain_text_splitters import CharacterTextSplitter

# create vectors from text
from langchain_openai import OpenAIEmbeddings

# stores vectors
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import PyPDFLoader


# Set the path to your folder
folder_path = r"D:\Projects\AI & ML\prompt_engineering\py-docs"

# Collect all file paths
documents = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(root, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")



print(f"Total documents loaded: {len(documents)}")

# Split documents
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_documents(documents)

print("Total chunks after splitting:", len(texts))

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Store vectors in Pinecone
# PineconeVectorStore.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     index_name=os.environ["PINECONE_INDEX_NAME"],
# )

vector_store = PineconeVectorStore.from_existing_index(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings,
    )

# Upload in smaller batches (e.g., 100 chunks at a time)
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    vector_store.add_documents(batch)
    print(f"Uploaded batch {i//batch_size + 1}")