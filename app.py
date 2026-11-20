import os
import streamlit as st
from google import genai
from rag_engine import RagEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
if "rag_assistant" not in st.session_state:
    st.session_state.rag_assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Assistant Controller ")
    
    # API Key Input
    api_key_input = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    
    st.divider()
    
    # PDF File Uploader
    uploaded_file = st.file_uploader("PDF Document", type=["pdf"])
    
    if uploaded_file and api_key_input:
        if st.button("🚀 Index Document"):
            with st.spinner("Processing PDF: chunking & embedding..."):
                try:
                    # Save uploaded file locally temporarily
                    temp_pdf_path = f"temp_{uploaded_file.name}"
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Initialize client and RAG Engine
                    os.environ["GEMINI_API_KEY"] = api_key_input
                    client = genai.Client(api_key=api_key_input)
                    
                    rag = RagEngine()
                    rag.index_document(temp_pdf_path)
                    
                    # Store in session state
                    st.session_state.rag = rag
                    st.session_state.client = client
                    st.session_state.is_indexed = True
                    
                    st.success("Document indexed successfully!")
                except Exception as e:
                    st.error(f"Error during indexing: {e}")

    st.divider()
    st.markdown("### 📊 System Status")
    if st.session_state.is_indexed:
        st.markdown("🟢 **Status:** Online & Ready")
        st.markdown(f"📦 **Total Chunks:** {len(st.session_state.rag.store.chunks)}")
    else:
        st.markdown("🔴 **Status:** Awaiting Document Index")

# --- MAIN CHAT INTERFACE ---
st.title("📄 Enterprise AI Document Assistant")
st.markdown("Ask questions about your uploaded document with verifiable page-level citations and real-time streaming.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("🔍 View Verified Sources & Citations"):
                for c in message["citations"]:
                    st.markdown(f"**Page {c.get('Page', c.get('page'))}** (Match Score: `{c.get('score', 0):.4f}`)")
                    st.caption(f"> {c.get('preview', c.get('text', ''))}")

# User input box
if query := st.chat_input("Ask a question about your document..."):
    if not st.session_state.is_indexed:
        st.warning("Please upload and index a PDF document using the sidebar first.")
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 1. Check if global query or specific vector retrieval
                query_lower = query.lower().strip()
                global_triggers = ["what is this pdf about", "what is this document about", "summarize", "summary", "overview", "main topic"]
                is_global = any(t in query_lower for t in global_triggers)
                
                if is_global:
                    raw_chunks = st.session_state.rag.store.chunks
                    intro = raw_chunks[:2]
                    conclusion = raw_chunks[-2:] if len(raw_chunks) > 4 else []
                    selected = intro + [c for c in conclusion if c not in intro]
                    joined_context = "\n\n".join([f"--- [Page {c['page_number']}] ---\n{c['text']}" for c in selected])
                    
                    final_prompt = (
                        "You are an expert document analyst. Provide an executive summary of this document "
                        "based strictly on its structural excerpts below.\n\n"
                        f"=== STRUCTURAL EXCERPTS ===\n{joined_context}\n\n"
                        f"=== USER REQUEST ===\n{query}\n\n=== SUMMARY ==="
                    )
                    citations = []
                else:
                    payload = st.session_state.rag.format_prompt(query, top_k=3)
                    final_prompt = payload["prompt"]
                    citations = payload["citations"]

                # 2. Live Stream from Gemini API
                response_stream = st.session_state.client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=final_prompt
                )

                for chunk in response_stream:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)

                # 3. Render Citations Expanders
                if citations:
                    with st.expander("🔍 View Verified Sources & Citations"):
                        for c in citations:
                            st.markdown(f"**Page {c.get('Page', c.get('page'))}** (Match Score: `{c.get('score', 0):.4f}`)")
                            st.caption(f"> {c.get('preview', c.get('text', ''))}")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations
                })

            except Exception as e:
                st.error(f"Error generating response: {e}")