from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
import gradio as gr
import os
import queue
import threading


PDF_PATH = "Deep Learning with Python.pdf"
PERSIST_DIR = "./chroma_store"
LLM_MODEL = "deepseek-r1"
EMBED_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3



class QueueCallbackHandler(BaseCallbackHandler):
    """Puts tokens into a queue for streaming to Gradio."""
    def __init__(self, token_queue: "queue.Queue"):
        self.token_queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs):
        self.token_queue.put(("token", token))

    def on_llm_end(self, *args, **kwargs):
        self.token_queue.put(("end", None))

    def on_llm_error(self, error, **kwargs):
        self.token_queue.put(("error", str(error)))


embeddings = OllamaEmbeddings(model=EMBED_MODEL)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(docs)

if os.path.exists(PERSIST_DIR):
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vector_db.persist()

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)


def warm_up_model():
    try:
        Ollama(model=LLM_MODEL).invoke("Hello!")
    except Exception:
        pass


threading.Thread(target=warm_up_model, daemon=True).start()



def chat_with_pdf_stream(user_input, chat_history):
    if chat_history is None:
        chat_history = []

    token_q = queue.Queue()
    handler = QueueCallbackHandler(token_q)

    
    llm = Ollama(model=LLM_MODEL, callbacks=[handler])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        memory=memory,
        return_source_documents=False,
    )

    prompt = f"""
Summarize the answer from the PDF in a bullet list, with enough detail for clarity.
Make each bullet short and easy to follow, with a one-line explanation.
Add a light touch of wit if appropriate.
Question: {user_input}
"""

    def run_chain():
        try:
            qa_chain({"question": prompt})
        except Exception as e:
            token_q.put(("error", str(e)))
        finally:
            token_q.put(("end", None))

    threading.Thread(target=run_chain, daemon=True).start()

    bot_text = ""
    chat_history = chat_history + [(user_input, "")]
    yield chat_history, ""

    while True:
        tag, payload = token_q.get()
        if tag == "token":
            bot_text += payload
            chat_history[-1] = (user_input, bot_text)
            yield chat_history, ""
        elif tag == "error":
            chat_history[-1] = (user_input, f"[Error] {payload}")
            yield chat_history, ""
            break
        elif tag == "end":
            final_text = bot_text.strip()
            if not final_text.startswith("•") and "\n" in final_text:
                formatted = "\n".join(
                    [f"• {line.strip()}" for line in final_text.split("\n") if line.strip()]
                )
            else:
                formatted = final_text
            chat_history[-1] = (user_input, formatted)
            yield chat_history, ""
            break


custom_css = """
#chatbot {height: 600px; overflow-y: auto;}
.message.user {
    background-color: #1e1e1e;
    color: white;
    padding: 10px;
    border-radius: 10px;
    max-width: 95%;
    margin-left: auto;
    white-space: pre-wrap;
}
.message.bot {
    background-color: #2b2b2b;
    color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    max-width: 95%;
    margin-right: auto;
    white-space: pre-wrap;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h2 style='text-align:center'>DeepSeek PDF Chatbot — streaming answers</h2>")
    chatbot = gr.Chatbot(elem_id="chatbot", height=600)
    msg = gr.Textbox(placeholder="Ask me anything from the PDF...", show_label=False)
    clear_btn = gr.Button("Clear Chat")

    msg.submit(chat_with_pdf_stream, [msg, chatbot], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

print("Starting Gradio app — warming model in background.")
demo.launch()
