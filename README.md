# Smart PDF Research Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that lets users upload a PDF, ask both specific and high-level questions, receive grounded answers with page citations, and interact through a Streamlit web interface[cite: 3, 4, 6].

## What It Does

* **PDF Extraction & Semantic Chunking:** Extracts text page-by-page with page metadata and splits content into semantic chunks using sentence boundary detection, embedding similarity checks, and word-length guardrails[cite: 3, 5].
* **NumPy Vector Search:** Encodes chunks and user queries into dense embedding vectors and ranks the most relevant chunks using dot products, L2 norms, and cosine similarity (`top_k`)[cite: 4, 5].
* **Query Routing & Grounded Prompts:** Routes high-level overview/summary queries to structural document excerpts while sending specific queries through vector retrieval, packaging context chunks, page numbers, match scores, and anti-hallucination rules[cite: 3, 4].
* **Live Streaming & Citations:** Connects to the Google Gemini API (`gemini-2.5-flash`) with live token streaming and displays page citations and similarity scores[cite: 3, 4].
* **Streamlit Web UI:** Wraps the pipeline in `app.py` using `st.session_state` so users can upload PDFs, chat, and inspect source citations in a browser[cite: 6].

## How the Pipeline Works

PDF -> Page Text & Metadata -> Semantic Chunks -> NumPy Embeddings & Cosine Similarity -> Context + Citations -> Gemini Streaming Answer
```[cite: 1, 3, 4, 5]```

## Setup & Run

1. **Install dependencies:**
   ```bash```
   pip install pypdf numpy sentence-transformers google-genai streamlit
   ```[cite: 1, 3, 5, 6]```

2. **Set your Gemini API key** (or enter it in the Streamlit sidebar):
   ```bash```
   export GEMINI_API_KEY="your_api_key_here"
   ```[cite: 1, 2, 6]```
   *(On Windows PowerShell: `$env:GEMINI_API_KEY="your_api_key_here"`)*[cite: 2]

3. **Launch the Streamlit app:**
   ```bash```
   streamlit run app.py
   ```[cite: 3, 6]```
