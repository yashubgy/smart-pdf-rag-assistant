import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List,Dict

class SimpleVectorStore:
    def __init__(self, model_name :str = "all-MiniLM-L6-v2"):
        print("Loading Model :  ", model_name)
        self.model = SentenceTransformer(model_name)
        self.chunks : List[Dict] = []
        self.vectors : np.ndarray 

    def add_chunks(self , preview_chunks : List[Dict]):
        self.chunks = preview_chunks
        texts = [chunk["Text"] for chunk in preview_chunks]
        print("Vectorizing using model ...... ")
        embeddings = self.model.encode(texts , show_progress_bar= True)
        self.vectors = np.array(embeddings)
        print(f"Embedding Vector shape : {self.vectors.shape}")

    def cosine_similarities(self , vec_a : np.ndarray , matrix_b : np.ndarray ):
        dot_product = np.dot(matrix_b ,vec_a)

        norm_A = np.linalg.norm(vec_a)
        norm_B = np.linalg.norm(matrix_b , axis = 1)

        similarity_score = (dot_product / (norm_A + norm_B + 1e-9))

        return similarity_score

    def search(self , query : str , top_k : int )-> List[Dict]:
        if self.vectors is None or len(self.chunks )==0:
            raise ValueError("No chunks Found !! : Call add_chunks first")

        #Embed user Querry
        query_vector = self.model.encode(query)

        #Compute similarites using Numpy:
        scores = self.cosine_similarities(query_vector , self.vectors)

        #Get highest scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        #return the best mateched chunks
        results = []
        for idx in top_indices:
            result_item = dict(self.chunks[idx])
            result_item["Similarity_score"] = float(scores[idx])
            results.append(result_item)
        return results
        

if __name__ == "__main__":
    from p_Ex import SemanticDocumentLoader
    loader = SemanticDocumentLoader(similarity_threshold = 0.75 ,min_words = 50 , max_words = 250)
    document = loader.load_pdf("Enter PDF Path")
    chunki = loader.chunk_documents_semantically(document)

    store = SimpleVectorStore()
    store.add_chunks(chunki)

    query = "What is this PDF about?"
    matches = store.search(query , top_k = 2)

    for i , match in enumerate(matches):
        print(match["text"][:500] + "...")
    