import streamlit as st
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG ---
WATCH_PATH = Path("./knowledge_base")
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text:latest"

# Available models for user selection
AVAILABLE_MODELS = [
    "qwen3:4b",
    "qwen2.5-coder:3b"
]

WATCH_PATH.mkdir(exist_ok=True)

@st.cache_resource
def setup_rag():
    """Initialize embeddings and vector store (no LLM here)"""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    vectorstore = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
        collection_name="documents"
    )
    
    return vectorstore

vectorstore = setup_rag()

def query_docs(question, model_name):
    """Retrieve relevant documents and generate answer with selected model"""
    
    # Create LLM with user-selected model
    llm = ChatOllama(model=model_name, temperature=0.1)
    
    try:
        count = vectorstore._collection.count()
    except:
        count = 0
    
    if count == 0:
        return "❌ No documents indexed. Add PDF/TXT files to `./knowledge_base/` and click **INDEX ALL FILES**."
    
    docs = vectorstore.similarity_search(question, k=min(3, max(1, count)))
    
    if not docs:
        return "No relevant documents found for your question."
    
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    
    prompt = f"""Based on this context only:

{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()

def index_folder():
    """Index all PDF and TXT files in the knowledge base folder"""
    pdf_files = list(WATCH_PATH.glob("*.pdf"))
    txt_files = list(WATCH_PATH.glob("*.txt"))
    files = pdf_files + txt_files
    
    if not files:
        return "No PDF/TXT files found in ./knowledge_base/"
    
    total_chunks = 0
    
    for file in files:
        try:
            if file.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file))
            else:
                loader = TextLoader(str(file))
            
            docs = loader.load()
            
            if not docs:
                continue
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)
            
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            
        except Exception as e:
            return f"Error processing {file.name}: {str(e)}"
    
    return f"✅ Indexed {len(files)} files → {total_chunks} chunks"

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="📚 Document Chat")
st.title("📚 Document Chat")

# Sidebar: File management and model selection
with st.sidebar:
    st.header("📁 Manage Documents")
    
    st.subheader("Upload File")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a file to add to your knowledge base"
    )
    
    if uploaded_file:
        try:
            file_path = WATCH_PATH / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Saved: {uploaded_file.name}")
            st.write(f"📍 Location: `{file_path}`")
        except Exception as e:
            st.error(f"Error saving file: {e}")
    
    st.subheader("Current Files")
    files = list(WATCH_PATH.glob("*.pdf")) + list(WATCH_PATH.glob("*.txt"))
    
    if files:
        for f in sorted(files):
            st.write(f"• {f.name}")
    else:
        st.info("No files added yet. Upload files above.")
    
    st.subheader("Indexing")
    if st.button("🔄 INDEX ALL FILES", type="primary", use_container_width=True):
        with st.spinner("Indexing documents..."):
            result = index_folder()
            st.success(result)
    
    # Model selection
    st.divider()
    st.subheader("🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose a model:",
        AVAILABLE_MODELS,
        index=0,
        help="Select which Ollama model to use for responses"
    )
    st.session_state.selected_model = selected_model

# Initialize chat history and model selection
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS[0]

# Main chat area
st.markdown("### 💬 Chat with Your Documents")

st.info("""
**Getting started:**
1. Upload PDF/TXT files in the sidebar
2. Click **INDEX ALL FILES** to process them
3. Select a model in the sidebar
4. Ask questions about your documents below
""")

# Display chat history
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
prompt = st.chat_input("What do you want to know about your documents?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking (using {st.session_state.selected_model})..."):
            response = query_docs(prompt, st.session_state.selected_model)
        st.markdown(response)
    
    st.session_state.history.append(("user", prompt))
    st.session_state.history.append(("assistant", response))

# Clear chat button
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

with col2:
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.history = []
        import shutil
        if Path(DB_PATH).exists():
            shutil.rmtree(DB_PATH)
        st.success("Database cleared!")
        st.rerun()