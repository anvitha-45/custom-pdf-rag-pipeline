import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

# Initialize embedding and language model components globally
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "google_genai:gemini-2.5-flash"
DB_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "research_collection"

def initialize_vector_store(file_path: str):
    """
    Downloads, parses, chunks, and indexes a PDF document into a Chroma vector store.
    """
    print(f"🔄 Loading document from: {file_path}...")
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    
    print("✂️ Splitting document into semantic text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(doc)
    
    print(f"🧬 Generating text embeddings using {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("💾 Persisting data splits into Chroma DB vector storage...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=DB_DIR
    )
    vector_store.add_documents(documents=all_splits)
    print("✅ Vector database ingestion completed successfully.")
    return vector_store

def retrieve_context(vector_store, query: str, k: int = 2):
    """
    Queries the vector database for nearest neighbor document contexts.
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)
    docs_content = ""
    for doc in retrieved_docs:
        docs_content += f"Source: {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"
    return docs_content, retrieved_docs

def docu_chat(vector_store, model, user_query: str):
    """
    Executes context-enriched retrieval-augmented generation pipelines.
    """
    context, source_docs = retrieve_context(vector_store, user_query, k=2)
    
    system_message = (
        "You are a helpful academic chatbot.\n"
        "Use ONLY the following pieces of context to answer the question. "
        "Do not make up any new information or extrapolate beyond the provided text.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    
    response = model.invoke(messages)
    return {
        "answer": response.content,
        "source_documents": source_docs,
        "context_used": context
    }

if __name__ == "__main__":
    # Standard testing file setup using the historical Attention is All You Need research paper
    target_pdf = "https://arxiv.org/pdf/1706.03762"
    
    # Instantiate the underlying storage index
    db_store = initialize_vector_store(target_pdf)
    
    # Initialize the LLM client securely utilizing environmental keys
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Missing 'GEMINI_API_KEY' inside environment variables.")
        
    chat_model = init_chat_model(LLM_MODEL_NAME, api_key=api_key)
    
    # Run test verification query
    test_query = "Explain what is the use of decoders in transformers?"
    print(f"\n🙋 User Query: {test_query}")
    result = docu_chat(db_store, chat_model, test_query)
    
    print("\n🤖 AI Answer:")
    print(result["answer"])
