import argparse
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

def main(pdf_path):
    # Check if the file exists
    if not os.path.isfile(pdf_path):
        print(f"Error: The file {pdf_path} does not exist.")
        return
    
    # Print the path to the PDF file
    print(f"Adding PDF file to knowledge base: {pdf_path}")

    load_dotenv('var.env')
    
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    split_documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="baai/bge-multilingual-gemma2",
        base_url="https://dekallm.cloudeka.ai/"
    )

    vectorstore = FAISS.from_documents(split_documents, embeddings)
    vectorstore.save_local("faiss_index")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Add a pdf to knowledge base")
    parser.add_argument('--path', type=str, required=True, help='Path to the PDF file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.path)