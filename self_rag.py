from modular_rag import ModularRAG
from llama_index.packs.self_rag.base import SelfRAGQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class SelfRAG(ModularRAG):
    def __init__(self):
        super().__init__()
        self.Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


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
        
    def create_query_engine(self):
        from pathlib import Path
        self.download_model()
        download_dir = "/model"
        model_path = str(Path(download_dir)) / "selfrag_llama2_7b.q4_k_m.gguf"
        self.query_engine = SelfRAGQueryEngine(model_path, self.retriever, verbose=True)

    def download_model(self):
        import subprocess
        download_dir = "/model"
        subprocess.run("pip install -q huggingface-hub", shell=True, check=True)
        subprocess.run("huggingface-cli download m4r1/selfrag_llama2_7b-GGUF selfrag_llama2_7b.q4_k_m.gguf --local-dir '/model' --local-dir-use-symlinks False -q huggingface-hub", shell=True, check=True)

