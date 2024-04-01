from modular_rag import ModularRAG
from llama_index.packs.self_rag.base import SelfRAGQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


class SelfRAG(ModularRAG):
    def __init__(self):
        super().__init__()

    def processing(self, file_path):
        print('Loading documents')
        self.load_document(file_path)
        print('Chunking documents')
        self.create_nodes()
        print('Creating index')
        self.create_index()
        print('Creating query engine')
        self.create_retriever()
        self.create_query_engine()

    def create_retriever(self):
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        
    def create_query_engine(self, model_path):
        from pathlib import Path
        download_dir = None
        model_path = str(Path(download_dir))
        self.query_engine = SelfRAGQueryEngine(model_path, self.retriever, verbose=True)

