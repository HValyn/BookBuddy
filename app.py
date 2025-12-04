"""
Book Chat Application
A Streamlit app for reading books and discussing them with an LLM buddy.
"""

import os
import streamlit as st
from src.document_loader import load_from_bytes
from src.text_processor import create_chunks_with_metadata, chunk_text
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


# Page configuration
st.set_page_config(
    page_title="Book Buddy",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Warm, book-friendly color scheme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Open+Sans:wght@400;600&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #faf8f5 0%, #f5f1eb 100%);
    }
    
    /* Hide default header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Main header */
    .app-header {
        font-family: 'Merriweather', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #3d2914;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #d4a574;
        margin-bottom: 1rem;
    }
    
    /* Book reader panel */
    .book-reader {
        background: #fffef9;
        border: 1px solid #e8dcc8;
        border-radius: 12px;
        padding: 1.5rem;
        height: 70vh;
        overflow-y: auto;
        font-family: 'Merriweather', serif;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #2c2416;
        box-shadow: 0 4px 20px rgba(139, 90, 43, 0.1);
    }
    
    .book-reader::-webkit-scrollbar {
        width: 8px;
    }
    
    .book-reader::-webkit-scrollbar-track {
        background: #f5f1eb;
        border-radius: 4px;
    }
    
    .book-reader::-webkit-scrollbar-thumb {
        background: #d4a574;
        border-radius: 4px;
    }
    
    .book-title {
        font-family: 'Merriweather', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #5c3d1e;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e8dcc8;
    }
    
    .page-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin-top: 1rem;
        padding: 0.5rem;
        background: #f5f1eb;
        border-radius: 8px;
    }
    
    /* Chat panel */
    .chat-container {
        background: #fffef9;
        border: 1px solid #e8dcc8;
        border-radius: 12px;
        padding: 1rem;
        height: 70vh;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 20px rgba(139, 90, 43, 0.1);
    }
    
    .chat-header {
        font-family: 'Merriweather', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #5c3d1e;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e8dcc8;
        margin-bottom: 0.75rem;
    }
    
    /* Chat messages styling */
    .stChatMessage {
        background: #faf8f5 !important;
        border-radius: 12px !important;
        border: 1px solid #e8dcc8 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #5c3d1e 0%, #3d2914 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f5f1eb;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #d4a574 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5a2b 0%, #5c3d1e 100%);
        color: #f5f1eb;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #a06930 0%, #6b4423 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(139, 90, 43, 0.3);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 254, 249, 0.8);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Status indicators */
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 1px solid #f5c6cb;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f5f1eb;
        border-radius: 8px 8px 0 0;
        color: #5c3d1e;
        font-family: 'Open Sans', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: #fffef9 !important;
        border-bottom: 2px solid #d4a574;
    }
    
    /* Welcome message */
    .welcome-box {
        background: linear-gradient(135deg, #fffef9 0%, #f5f1eb 100%);
        border: 2px dashed #d4a574;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .welcome-title {
        font-family: 'Merriweather', serif;
        font-size: 1.5rem;
        color: #5c3d1e;
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        font-family: 'Open Sans', sans-serif;
        color: #6b5344;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine(
            vector_store=st.session_state.vector_store
        )
    if 'current_book' not in st.session_state:
        st.session_state.current_book = None
    if 'book_loaded' not in st.session_state:
        st.session_state.book_loaded = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'book_text' not in st.session_state:
        st.session_state.book_text = ""
    if 'book_pages' not in st.session_state:
        st.session_state.book_pages = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0


def process_uploaded_file(uploaded_file) -> bool:
    """Process an uploaded book file."""
    try:
        with st.spinner("📖 Reading your book..."):
            # Load document
            text = load_from_bytes(
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name
            )
            
            if not text.strip():
                st.error("Could not extract text from the file.")
                return False
            
            # Store full text for reading
            st.session_state.book_text = text
            
            # Split into pages (roughly 2000 chars per page for readability)
            st.session_state.book_pages = chunk_text(text, chunk_size=2000, chunk_overlap=0)
            st.session_state.current_page = 0
            
            # Create chunks for RAG
            book_name = os.path.splitext(uploaded_file.name)[0]
            chunks = create_chunks_with_metadata(text, source=book_name)
            
            if not chunks:
                st.error("Could not process the book content.")
                return False
            
            # Add to vector store
            with st.spinner(f"🔍 Indexing {len(chunks)} passages..."):
                st.session_state.vector_store.add_documents(chunks, book_name)
            
            # Update state
            st.session_state.current_book = book_name
            st.session_state.rag_engine.set_book(book_name)
            st.session_state.book_loaded = True
            st.session_state.messages = []
            
            return True
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("### 📚 Book Buddy")
        st.markdown("---")
        
        # Ollama status
        ollama_available = st.session_state.rag_engine.is_ollama_available()
        
        if ollama_available:
            st.success("✅ Ollama connected")
            models = st.session_state.rag_engine.get_available_models()
            if models:
                selected = st.selectbox(
                    "AI Model",
                    models,
                    index=0 if not st.session_state.selected_model else 
                          models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
                )
                st.session_state.selected_model = selected
            else:
                st.warning("No models found")
        else:
            st.error("❌ Ollama offline")
            st.code("ollama serve", language="bash")
        
        st.markdown("---")
        st.markdown("### 📄 Load Book")
        
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['pdf', 'epub', 'mobi'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            if st.button("📖 Open Book", use_container_width=True):
                if process_uploaded_file(uploaded_file):
                    st.success(f"✅ Loaded!")
                    st.rerun()
        
        if st.session_state.book_loaded:
            st.markdown("---")
            st.markdown(f"**📖 {st.session_state.current_book}**")
            st.caption(f"{len(st.session_state.book_pages)} pages")
            
            if st.button("🗑️ Close Book", use_container_width=True):
                st.session_state.vector_store.delete_book(st.session_state.current_book)
                st.session_state.current_book = None
                st.session_state.book_loaded = False
                st.session_state.book_text = ""
                st.session_state.book_pages = []
                st.session_state.messages = []
                st.rerun()


def render_book_reader():
    """Render the book reading panel."""
    st.markdown('<div class="book-title">📖 ' + st.session_state.current_book + '</div>', unsafe_allow_html=True)
    
    pages = st.session_state.book_pages
    current = st.session_state.current_page
    total = len(pages)
    
    # Page navigation
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️", disabled=current == 0, use_container_width=True):
            st.session_state.current_page = 0
            st.rerun()
    
    with col2:
        if st.button("◀️", disabled=current == 0, use_container_width=True):
            st.session_state.current_page = max(0, current - 1)
            st.rerun()
    
    with col3:
        # Page selector
        page_num = st.selectbox(
            "Page",
            range(1, total + 1),
            index=current,
            label_visibility="collapsed"
        )
        if page_num - 1 != current:
            st.session_state.current_page = page_num - 1
            st.rerun()
    
    with col4:
        if st.button("▶️", disabled=current >= total - 1, use_container_width=True):
            st.session_state.current_page = min(total - 1, current + 1)
            st.rerun()
    
    with col5:
        if st.button("⏭️", disabled=current >= total - 1, use_container_width=True):
            st.session_state.current_page = total - 1
            st.rerun()
    
    st.caption(f"Page {current + 1} of {total}")
    
    # Book content
    if pages:
        content = pages[current]
        # Format paragraphs
        paragraphs = content.split('\n')
        formatted = '<br><br>'.join(p.strip() for p in paragraphs if p.strip())
        st.markdown(
            f'<div class="book-reader">{formatted}</div>',
            unsafe_allow_html=True
        )


def render_chat_panel():
    """Render the chat panel."""
    st.markdown('<div class="chat-header">💬 Chat with your Book Buddy</div>', unsafe_allow_html=True)
    
    # Check readiness
    if not st.session_state.rag_engine.is_ollama_available():
        st.warning("Start Ollama to chat")
        return
    
    if not st.session_state.selected_model:
        st.warning("Select a model in sidebar")
        return
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about what you're reading..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                response = ""
                
                try:
                    for chunk in st.session_state.rag_engine.query_stream(
                        question=prompt,
                        model=st.session_state.selected_model,
                        chat_history=st.session_state.messages[:-1]
                    ):
                        response += chunk
                        placeholder.markdown(response + "▌")
                    
                    placeholder.markdown(response)
                except Exception as e:
                    response = f"Sorry, an error occurred: {str(e)}"
                    placeholder.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


def render_welcome():
    """Render welcome screen when no book is loaded."""
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">Welcome to Book Buddy!</div>
        <div class="welcome-text">
            Upload a book to start reading and chatting.<br>
            Your AI buddy will help you understand characters, summarize chapters, and discuss the story.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📖 Read")
        st.write("Read your book right in the app with comfortable formatting")
    
    with col2:
        st.markdown("### 💬 Discuss")
        st.write("Ask questions about characters, plot, themes - like having a reading buddy")
    
    with col3:
        st.markdown("### 🔒 No Spoilers")
        st.write("AI only knows what you've read - no plot spoilers ahead")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Header
    st.markdown('<div class="app-header">📖 Book Buddy</div>', unsafe_allow_html=True)
    
    if st.session_state.book_loaded:
        # Split view: Book reader + Chat
        col_book, col_chat = st.columns([3, 2])
        
        with col_book:
            render_book_reader()
        
        with col_chat:
            render_chat_panel()
    else:
        render_welcome()


if __name__ == "__main__":
    main()
