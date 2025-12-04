Book Buddy - Walkthrough
A local Streamlit app for reading books and chatting with an AI buddy.

Project Structure
/home/dslab527/Documents/exper/
├── app.py                    # Main Streamlit UI 
├── requirements.txt          # Dependencies               
└── src/
    ├── document_loader.py    # PDF/EPUB/MOBI parsing
    ├── text_processor.py     # Text chunking
    ├── embeddings.py         # Local embeddings
    ├── vector_store.py       # ChromaDB storage
    ├── ollama_client.py      # Ollama API client
    └── rag_engine.py         # RAG query logic
How to Run
# Start Ollama 
Install ollama and the model you want to run.
# Run the app
Create your environment and install dependencies...  (From requirements.txt)
streamlit run app.py
Open: http://localhost:8501

UI Features
Feature	Description
📖 Book Reader	Left panel with paginated book content
💬 Chat Panel	Right panel for AI discussion
⏮️⏭️ Navigation	Page controls: first, prev, jump-to, next, last
🎨 Warm Theme	Book-friendly cream & brown colors
How It Works
Upload a book (PDF/EPUB/MOBI) via sidebar
Read on the left panel - navigate with page controls
Chat on the right - ask about characters, plot, themes
