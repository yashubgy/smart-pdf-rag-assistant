from typing import List , Dict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class SemanticDocumentLoader:
    def __init__(self, model_name : str = "all-MiniLM-L6-v2" , similarity_threshold = 0.75 ,min_words = 50 , max_words = 250):
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.chunk_min_words = min_words
        self.chunk_max_words = max_words

    def load_pdf(self , file_path : str ) -> List[Dict]:
            reader = PdfReader(file_path)
            document = []
            for page_idx , page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    document.append({"Text" : text.strip(),  "page_number" : page_idx + 1})
            return document
    
    def split_into_sentences(self , txt : str)-> List[str]:
        # ?<= : Look Back
        #[.?!] : Ignore whatever is inside it .,?,!
        #\s+ : Remove all with \s[space] + [more than 1 also]
        sentences = re.split(r'(?<=[.?!])\s+' , txt)
        return [s.strip() for s in sentences if s.strip()]

    def cosine_similarities(self, vec_a : np.ndarray , vec_b : np.ndarray)-> float:
         dot_product = np.dot(vec_a,vec_b)
         norm_A = np.linalg.norm(vec_a)
         norm_B = np.linalg.norm(vec_b)

         return float(dot_product / (norm_A*norm_B + 1e-9))

    def chunk_documents_semantically(self , document : List[Dict]) -> List[Dict]:
        all_chunks = []
        chunk_id = 0

        for doc in document:
              page_num = doc["page_number"]
              sentences = self.split_into_sentences(doc['Text'])

              if not sentences :
                   continue

              if len(sentences) ==1 :
                   all_chunks.append({
                        "chunk_id" : chunk_id,
                        "page_number" : page_num,
                        "Text" : sentences[0]
                   })
                   chunk_id+=1
                   continue

              sentence_vectores = self.model.encode(sentences , show_progress_bar= True)
              current_chunk_sentences = [sentences[0]]

              for i in range(len(sentences)-1):
                   score = self.cosine_similarities(sentence_vectores[i],sentence_vectores[i+1])
                   next_sentence = sentences[i+1]
                   word_count = sum(len(s.split()) for s in current_chunk_sentences)
                   
                   is_below_threshold = score < self.threshold
                   is_inLimit = word_count > self.chunk_min_words
                   is_overLimit = word_count > self.chunk_max_words

                   if(is_below_threshold and is_inLimit) or is_overLimit:
                        all_chunks.append({
                        "chunk_id" : chunk_id,
                        "page_number" : page_num,
                        "Text" : " ".join(current_chunk_sentences)
                    })
                        chunk_id+=1
                        current_chunk_sentences = [next_sentence]
                   else:
                        current_chunk_sentences.append(next_sentence)

              if current_chunk_sentences:
                all_chunks.append({
                                        "chunk_id" : chunk_id,
                                        "page_number" : page_num,
                                        "Text" : " ".join(current_chunk_sentences)
                                    })
                chunk_id+=1

        return all_chunks
        

                   
                   

        

    
        
        
