import streamlit as st
import os
import time
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- BULLETPROOF CONFIG ---
WATCH_PATH = "./knowledge_base"
DB_PATH = "./chroma_db"
MODEL_NAME = "qwen3:4b"  
EMBED_MODEL = "nomic-embed-text:latest"

os.makedirs(WATCH_PATH, exist_ok=True)

@st.cache_resource
def setup_rag():
    """Fresh Chroma setup every time"""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Always fresh - no persist issues
    vectorstore = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
    return vectorstore, llm

vectorstore, llm = setup_rag()

def index_folder():
    """Index ALL files in folder - manual verification"""
    files = list(WATCH_PATH.glob("*.pdf")) + list(WATCH_PATH.glob("*.txt"))
    
    if not files:
        return "No PDF/TXT files found"
    
    total_chunks = 0
    for file in files:
        try:
            if file.suffix == '.pdf':
                loader = PyPDFLoader(str(file))
            else:
                loader = TextLoader(str(file))
            
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800)
            chunks = splitter.split_documents(docs)
            
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            
        except Exception as e:
            return f"Error {file.name}: {e}"
    
    return f"✅ Indexed {len(files)} files → {total_chunks} chunks"

def query_docs(question):
    """Simple query with debug"""
    # Count before search
    count = vectorstore._collection.count()
    st.caption(f"**DB size:** {count} chunks")
    
    if count == 0:
        return "❌ No documents indexed. Add PDF/TXT files and click INDEX."
    
    docs = vectorstore.similarity_search(question, k=min(3, count))
    st.caption(f"🔍 Found **{len(docs)}** docs")
    
    if not docs:
        return "No relevant docs found for your question"
    
    # Show preview
    preview = docs[0].page_content[:150] + "..."
    st.caption(f"**Preview:** {preview}")
    
    # Generate answer
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = f"""Based on this context only:

{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()

# --- UI ---
st.set_page_config(layout="wide")
st.title("📚 Document Chat - INDEX FIRST!")

# Sidebar - MANUAL INDEXING (no watcher issues)
with st.sidebar:
    st.header("📁 Add Documents")
    
    # File uploader as backup
    uploaded = st.file_uploader("Or upload here", type=['pdf','txt'])
    if uploaded:
        with open(f"{WATCH_PATH}/{uploaded.name}", "wb") as f:
            f.write(uploaded.getvalue())
        st.success(f"Saved {uploaded.name}")
        st.rerun()
    
    # Folder contents
    files = [f for f in os.listdir(WATCH_PATH) if f.endswith(('.pdf','.txt'))]
    st.write("**Files:**")
    for f in files:
        st.write(f"• {f}")
    
    # MANUAL INDEX BUTTON
    if st.button("🔄 INDEX ALL FILES", type="primary"):
        with st.spinner("Indexing..."):
            result = index_folder()
            st.success(result)
            st.rerun()

# Main chat
st.markdown("### 💬 Chat with your documents")
st.info("""
1. **Add PDF/TXT** files to `./knowledge_base/` OR upload above
2. **Click INDEX ALL FILES** 
3. **Ask questions** about your docs!
""")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("What do you want to know?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = query_docs(prompt)
        st.markdown(response)
        
        st.session_state.history.append(("user", prompt))
        st.session_state.history.append(("assistant", response))

if st.button("🗑️ Clear Chat"):
    st.session_state.history = []
    st.rerun()