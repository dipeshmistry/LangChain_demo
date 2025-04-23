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


if __name__ == "__main__":
    # load doc
    loader = TextLoader("D:/Projects/AI & ML/prompt_engineering/abc.txt")
    document = loader.load()
    
    # print(document)
    # spilt doc
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents=document)
    print("texts = ", len(texts))

    # injest  / create vectors amd stroing
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
