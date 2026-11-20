import os
from typing import Dict
from google import genai
from rag_engine import RagEngine

class RagAssistant:
    def __init__(self, pdf_path : str , api_key : str  ):

        "initializing the live rag assistant"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini key unavailable !!")

        self.client = genai.Client(api_key = self.api_key)
        self.model_name = "gemini-3.6-flash"

        self.rag = RagEngine()
        self.rag.index_document(pdf_path)

    def _is_global_query(self , query:str) -> bool:
        
        "check if user is asking for sumary or high level query"

        global_triggers = ["what is this pdf about ? "
                           "summary "
                           "summarize this"
                           "overview"
                           "briefly explain "
                           "main topic"
                           "brief me"
                           "what is this document about?"]

        q_lower = query.lower().strip()
        return any(trigger in q_lower for trigger in global_triggers)

    def _assemble_global_prompt(self , query: str):
        raw_chunks = self.rag.store.chunks

        if not raw_chunks:
            raise ValueError("Document is Empty / Not indexed")

        intro_chunks = raw_chunks[:2]
        conclusion_chunks = raw_chunks[-2:] if len(raw_chunks >4) else []
        selected_chunks = intro_chunks + [c for c in conclusion_chunks if c not in intro_chunks]

        context_lines = [f"---Page No. : {c["page_number"]} ---\nText : {c["text"]}---" for c in selected_chunks]
        joined_context = "\n\n".join(context_lines)       

        return (
            "You are an expert document analyst. Provide an executive summary of this document "
            "based strictly on its structural excerpts below.\n\n"
            f"=== STRUCTURAL DOCUMENT EXCERPTS ===\n"
            f"{joined_context}\n\n"
            f"=== USER REQUEST ===\n"
            f"{query}\n\n"
            f"=== SUMMARY ==="
        )

    def ask(self,query , top_k  :int =3 ):
        print(f"Processing the query :  {query}.... ")

        #check query type : 

        if self._is_global_query(query):
            print("-> [Router]: Global summary query detected (using structural anchors).")
            final_prompt = self._assemble_global_prompt(query)
            citations = []
        else:
            print(f"-> [Router]: Specific query detected (running Top-{top_k} vector search).")
            payload = self.rag.format_prompt(query,top_k = top_k)
            final_prompt = payload["prompt"]
            citations = payload["citations"]

        #Stream content generation from the live API:
        print("\n" + "=" * 60)
        print("ASSISTANT RESPONSE:")
        print("=" * 60)

        response_stream = self.client.models.generate_content_stream(model = self.model_name , contents = final_prompt)

        for chunk in response_stream:
            print(chunk.text , end = "" , flush = True)


        print("\n" + "=" * 60)

        if citations:
            print("\nSources Verified & Retrieved:")
            for c in citations:
                print(f"  • Page {c.get('Page', c.get('page'))} (Similarity: {c.get('score', 0):.4f})")
            print("-" * 60)


if __name__ == "__main__":
    
    
    # Run the assistant (make sure GEMINI_API_KEY is exported or paste your key string here)
    assistant = RagAssistant(pdf_path="Enter PDF Path" ,api_key = "Enter Your Api key")

    # Test Specific Question (Vector Retrieval)
    assistant.ask("How many lungs do human have?")

    