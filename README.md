#  DeepSeek PDF Chatbot — Streaming Answers with LangChain, Ollama & ChromaDB

This project is an **interactive PDF Q&A chatbot** that:
- Loads and processes a PDF document
- Splits it into chunks and stores embeddings in **ChromaDB**
- Uses **Ollama** LLM with streaming responses
- Provides **conversational memory** so the chatbot remembers context
- Runs on a **Gradio** interface with a custom dark theme

##  Features
- **PDF Loader**: Reads any PDF file using `PyPDFLoader`
- **Vector Store**: Persists embeddings locally with `ChromaDB`
- **Streaming Chat**: Token-by-token response streaming for a natural chat feel
- **Conversational Retrieval Chain**: Memory-enabled context-aware answers
- **Custom UI**: Dark theme chat interface via Gradio CSS

##  Tech Stack
- **LangChain** (Prompt handling, ConversationalRetrievalChain)
- **Ollama** (LLM & embeddings)
- **ChromaDB** (Vector database for document search)
- **PyPDFLoader** (PDF text extraction)
- **Gradio** (User interface)
- **Python** (Core implementation)


