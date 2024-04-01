from modular_rag import ModularRAG
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.packs.corrective_rag import CorrectiveRAGPack
import os

class CorrectiveRAG(ModularRAG):
    def __init__(self):
        super().__init__()
        
    
    def processing(self, file_path):
        tavily_ai_api_key = os.environ["TAVILY_API_KEY"]
        self.corrective_rag_pack = CorrectiveRAGPack(
            self.documents, 
            tavily_ai_apikey=tavily_ai_api_key)
        
    def query(self, user_input):
        return self.corrective_rag_pack.run(user_input)
