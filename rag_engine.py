from typing import List,Dict
import re
from embedding_creation import SimpleVectorStore
from p_Ex import SemanticDocumentLoader

class RagEngine:
    def __init__(self):
        self.loader = SemanticDocumentLoader(similarity_threshold = 0.3 ,min_words = 50 , max_words = 250)
        self.store = SimpleVectorStore()
        self.is_indexed : bool = False

    def index_document(self, file_path : str):
        doc = self.loader.load_pdf(file_path)
        if not doc:
            raise ValueError("No PDF File found to be used! ")
        chunks = self.loader.chunk_documents_semantically(doc)
        self.store.add_chunks(chunks)
        self.is_indexed = True
        print(" Document have been indexed Succesfully " \
        "Total Chunks : ",len(chunks))

    def format_prompt(self , query: str , top_k : int)-> Dict:
        relevant_chunks = self.store.search(query , top_k = top_k)
        context_parts = []
        citations = []

        for i,chunk in enumerate(relevant_chunks):
            page = chunk["page_number"]
            score = chunk["Similarity_score"]
            text = chunk["Text"]

            context_parts.append(f"=== Source Excerpt {i+1} (Page : {page}), Match :{score:.2f} ===\n{text}")
            citations.append({
                "Page" : page,
                "score" :score,
                "preview" : text[:120] + "..." if len(text) > 120 else text

            })

        joined_context = "\n\n".join(context_parts)

        system_instructions = ("You are a strict, expert AI document assistant. Your job is to answer the user's question "
            "using ONLY the provided source excerpts below.\n\n"
            "STRICT RULES:\n"
            "1. Answer strictly based on the provided context excerpts.\n"
            "2. If the answer cannot be found in the context, explicitly say: "
            "'The document does not contain this information.' Never make up an answer.\n"
            "3. Explicitly cite the exact page number(s) for any claim made in your response.")

        final_prompt = (
            f"{system_instructions}\n\n"
            f"=== CONTEXT FROM DOCUMENT ===\n"
            f"{joined_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{query}\n\n"
            f"=== ANSWER (with Page Citations) ==="
        )

        return {
            "prompt": final_prompt,
            "citations": citations,
            "raw_chunks": relevant_chunks
        }


if __name__ == "__main__":
    rag = RagEngine()

    rag.index_document("Enter PDF Path")

    query = " what is respiratory system?"
    payload = rag.format_prompt(query , top_k = 3)

    print("Info : ")
    print(payload["prompt"])
    print("\n\nResponse : ")
    print(payload["citations"])

