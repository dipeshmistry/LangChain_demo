import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

# 🔍 Recursively load specific file types
def load_code_files_as_documents(root_folder, file_extensions=(".js", ".yaml", ".yml", ".sh", ".md")):
    documents = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(file_extensions):
                full_path = os.path.join(dirpath, filename)
                file_type = os.path.splitext(filename)[1]  
                try:
                    with open(full_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        metadata = {
                            "path": full_path,
                            "file_name": filename,
                            "file_type": file_type
                        }   
                        documents.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    print(f"❌ Could not read {full_path}: {e}")
    return documents

if __name__ == "__main__":
    code_folder = "C:/Users/Dipesh/Downloads/Product-Catalogue-API-main/Product-Catalogue-API-main"  # 🔁 Change to your folder path

    # 📄 Load all matching files
    documents = load_code_files_as_documents(code_folder)

    # ✂️ Split documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents=documents)
    print(f"✅ Total files loaded: {len(documents)}")
    print(f"✅ Total chunks generated: {len(texts)}")

    # 🔐 Embeddings and Pinecone injection
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
