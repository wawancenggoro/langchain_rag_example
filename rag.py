import argparse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

def main(prompt):
    load_dotenv('var.env')

    embeddings = OpenAIEmbeddings(
        model="baai/bge-multilingual-gemma2",
        base_url="https://dekallm.cloudeka.ai/"
    )

    llm = ChatOpenAI(
        model="qwen/qwen25-72b-instruct",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url="https://dekallm.cloudeka.ai/"
    )
    # Load the vector store
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """  
        Given this context: {context}
        Answer this question = {question}
        You need to answer the question in the sentence as same as in the context. 
        If the answer is not in the context, answer "I cannot answer"
        """
    prompt_template = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(prompt)
    print(response)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Prompt to LLM with RAG")
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to LLM with RAG')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.prompt)