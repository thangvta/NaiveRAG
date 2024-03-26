import os, weaviate
from dotenv import load_dotenv; load_dotenv()
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings

class NaiveRAG:
    def __init__(self):
        self.Settings = Settings
        self.Settings.llm = Gemini(models='gemini-pro', api_key=os.getenv("GOOGLE_API_KEY"))
        self.Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
        self.Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        self.Settings.num_output = 512
        self.Settings.context_window = 3900
        self.WClient = self.connect_weaviate()
        self.vector_store = self.vectorstore()
        self.doc_status = False

    def connect_weaviate(self, mode='local'):
        if mode=='local':
            return weaviate.Client(
                url = "http://localhost:8080",  # Replace with your endpoint
                additional_headers = {
                    "X-Google-Api-Key": os.getenv("GOOGLE_API_KEY") # Replace with your inference API key
                }
            )

    def processing(self, file_path):
        self.load_document(file_path)
        self.create_query_engine()

    def load_document(self, file_path:str="./data"):
        # Load documents
        self.documents=SimpleDirectoryReader(file_path).load_data()
        self.doc_status = True

    def vectorstore(self):
        # Check if Weaviate Service is ready
        if self.WClient.is_ready():
            print('Weaviate is ready!')
            return WeaviateVectorStore(weaviate_client=self.WClient)
        else:
            print(" Not found Weaviate service !!! Pls check service status")
            return None

    def create_query_engine(self):
        if self.vector_store is not None:
        # set up storage for embeddings
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        else:
            print('Pls check vector store stauts')
        if self.doc_status:
            # set up the index 
            index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=storage_context
            )
        else:
            print("Pls load documnets first!")
        # Create indexing
        self.query_engine = index.as_query_engine()

    def query(self, user_input):
        return self.query_engine.query(user_input)

