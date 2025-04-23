import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

def load_all_sheets_as_documents(filepath):
    documents = []
    sheet_names = pd.ExcelFile(filepath).sheet_names[:3]
    excel = pd.read_excel(filepath, sheet_name=sheet_names)  # Load all sheets as dict
    for sheet_name, df in excel.items():
        df = df.dropna(how='all')  # Remove completely empty rows
        for idx, row in df.iterrows():
            text = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
            documents.append(Document(page_content=text, metadata={"sheet": sheet_name, "row": idx}))
    return documents

if __name__ == "__main__":
    excel_path = "D:/Projects/AI & ML/prompt_engineering/Consolidated Schneider Electric Survey Responses.xlsx"
    
    # 📄 Load Excel content from all sheets
    document = load_all_sheets_as_documents(excel_path)

    # ✂️ Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents=document)
    print(f"✅ Total rows loaded: {len(document)}")
    print(f"✅ Total chunks generated: {len(texts)}")

    # 🔐 Embeddings and Pinecone injection
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=os.environ["OPENAI_API_KEY"])

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
