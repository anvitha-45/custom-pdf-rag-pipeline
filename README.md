# Custom PDF Retrieval-Augmented Generation (RAG) Pipeline

A complete, production-ready Retrieval-Augmented Generation (RAG) system engineered using Python and the LangChain ecosystem to ingest, index, and query dense document corpora via semantic embedding matches.

## 🚀 Architectural Design
- **Document Loading & Extraction:** Implements an asynchronous web parsing pipeline using `PyPDFLoader` to download and read unstructured data layouts directly from URLs.
- **Semantic Text Chunking:** Employs an intelligent `RecursiveCharacterTextSplitter` configured with a 1000-character chunk bound and 200-character step overlap to preserve document context metrics across split limits.
- **Vector Embeddings Execution:** Leverages Hugging Face's `sentence-transformers/all-mpnet-base-v2` package to turn textual indices into high-dimensional vector representations.
- **Storage Layer Persistence:** Embeds data points inside a local, optimized `Chroma` database vector collection to scale fast cosine similarity lookups.
- **Context-Restricted Inference:** Enforces a strict grounding layer onto `gemini-2.5-flash` via dynamic system prompts to prevent model hallucinations.

## 🛠️ Tech Stack
- **Framework Suite:** LangChain, LangChain-Community, LangChain-Chroma
- **Embedding Transformer:** Sentence-Transformers (`all-mpnet-base-v2`)
- **Vector Base:** Chroma DB
- **LLM Reasoning Engine:** Google Gemini API Matrix (`gemini-2.5-flash`)
- **Data Ingestion Tool:** PyPDF Engine
