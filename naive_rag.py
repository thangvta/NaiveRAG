import os, weaviate
from dotenv import load_dotenv; load_dotenv()
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore


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
        self.doc_store = SimpleDocumentStore()
        self.doc_status = False

    def connect_weaviate(self, mode='local'):
        if mode=='local':
            client = weaviate.Client(
                url = "http://localhost:8080",  # Replace with your endpoint
                additional_headers = {
                    "X-Google-Api-Key": os.getenv("GOOGLE_API_KEY") # Replace with your inference API key
                }
            )
            print(client)
            return client

    def processing(self, file_path):
        print('Loading documents')
        self.load_document(file_path)
        print('Chunking documents')
        self.create_nodes()
        print('Creating index')
        self.create_index()
        print('Creating query engine')
        self.create_query_engine()

    def load_document(self, file_path:str="./data"):
        # Load documents
        self.documents=SimpleDirectoryReader(file_path).load_data()
        self.doc_status = True

    def create_nodes(self):
        if self.doc_status:
            nodes = self.Settings.node_parser.get_nodes_from_documents(documents=self.documents)
            self.doc_store.add_documents(nodes)
            return nodes
        else:
            print("Pls load documnets first!")
            return None

    def vectorstore(self):
        # Check if Weaviate Service is ready
        if self.WClient.is_ready():
            print('Weaviate is ready!')
            return WeaviateVectorStore(weaviate_client=self.WClient)
        else:
            print(" Not found Weaviate service !!! Pls check service status")
            return None


    def create_index(self):
        nodes = self.create_nodes()
        self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context
            )
    

    def create_query_engine(self):
        if self.vector_store is not None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store, docstore=self.doc_store)
        else:
            print('Pls check vector store status')

        
        self.query_engine = self.index.as_query_engine()

    def query(self, user_input):
        return self.query_engine.query(user_input)

