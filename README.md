📄 Smart PDF Research Assistant (RAG from Scratch)A lightweight, end-to-end Retrieval-Augmented Generation (RAG) application built from scratch in Python using NumPy, Sentence-Transformers, Google Gemini (gemini-2.5-flash), and Streamlit.
Upload any PDF document, ask both specific detail questions and high-level summary questions, and receive real-time streamed answers backed by verifiable page-level citations and anti-hallucination guardrails.✨ 
Key FeaturesSemantic & Sliding-Window Chunking: Extracts text page-by-page while preserving page metadata (page_number) and splitting text into coherent chunks using sentence boundaries and embedding similarity thresholds.
Pure NumPy Vector Store: Encodes document chunks into 384-dimensional dense vectors (all-MiniLM-L6-v2) and computes Top-$K$ retrieval using matrix dot products, $L_2$ norms, and cosine similarity without heavy external vector database dependencies.
Intent-Based Query Routing: Automatically distinguishes between specific ("needle") queries (routed through Top-$K$ cosine similarity search) and global summary ("haystack") queries (routed to structural document anchors from opening and concluding sections).
Anti-Hallucination Guardrails: Enforces strict prompt boundaries instructing the model to rely only on retrieved excerpts, cite exact page numbers, and explicitly decline to guess when information is absent.Live Token Streaming & Citation Inspector: Streams responses token-by-token in a Streamlit web interface with expandable source drawers displaying page numbers, cosine similarity scores, and text previews.

🏗️ Architecture & Pipeline FlowPDF Document
     │
     ▼
1. Ingestion & Semantic Chunking (p_Ex.py / document_loader.py)
     │  ──► Extracts page text + metadata, splits into semantic chunks
     ▼
2. Vector Store & Cosine Similarity (embedding_creation.py / vector_store.py)
     │  ──► Embeds chunks (384-d) & ranks Top-K matches via NumPy matrix math
     ▼
3. Prompt Engine & Guardrails (rag_engine.py)
     │  ──► Formats source blocks with page tags & anti-hallucination rules
     ▼
4. Query Router & Gemini Streaming (llm_client.py)
     │  ──► Routes global vs. specific queries & streams gemini-2.5-flash output
     ▼
5. Interactive Web UI (app.py)
        ──► Streamlit chat interface with st.session_state & citation expanders
📂 Project Structuresmart-pdf-rag-assistant/
├── app.py                  # Streamlit web interface with session state & citation UI
├── llm_client.py           # CLI client with intent routing & live Gemini streaming
├── rag_engine.py           # Prompt assembly, context formatting & grounding guardrails
├── embedding_creation.py   # NumPy-based dense vector store & cosine similarity search
├── p_Ex.py                 # PDF text extraction & semantic chunking loader
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored cache, temp PDFs, and secret files
└── README.md               # Project documentation
🚀 Quickstart & Setup1. Clone the Repositorygit clone https://github.com/yashubgy/smart-pdf-rag-assistant.git
cd smart-pdf-rag-assistant
2. Install Dependenciespip install -r requirements.txt
3. Set Your Gemini API KeyGet a free API key from Google AI Studio and set it as an environment variable (or enter it directly in the Streamlit sidebar):Windows (PowerShell):$env:GEMINI_API_KEY="your_actual_api_key_here"
Linux / macOS:export GEMINI_API_KEY="your_actual_api_key_here"
4. Run the Streamlit Web Appstreamlit run app.py
Open the local URL shown in your terminal (typically http://localhost:8501), upload a PDF in the sidebar, click Index Document, and start asking questions.🛠️ Tech StackLanguage: Python 3.10+Linear Algebra & Vector Search: NumPyEmbeddings: sentence-transformers (all-MiniLM-L6-v2)PDF Parsing: pypdfLLM & Streaming: Google GenAI SDK (google-genai, gemini-2.5-flash)Frontend UI: Streamlit🔮 Future ImprovementsHybrid Search: Combine dense vector similarity with sparse keyword matching (BM25) for exact acronym and formula retrieval.Multi-PDF Support: Index and query across multiple uploaded documents simultaneously with document-level source attribution.Automated RAG Evaluation: Add retrieval hit-rate and answer faithfulness benchmarks.📄 LicenseThis project is licensed under the MIT License.
