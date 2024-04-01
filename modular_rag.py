from naive_rag import NaiveRAG
import os, weaviate
from enums import Node_parser
from dotenv import load_dotenv; load_dotenv()
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore


class ModularRAG(NaiveRAG):
    def __init__(self, llm=None, embed_model=None, node_parser=None):
        super().__init__()
        self.node_parser = self.set_node_parser(node_parser)
        
    
    def connect_weaviate(self, mode='local', api_key_provider:str=None):
        if mode == 'local':
            if api_key_provider is None or api_key_provider=="Google":
                client = weaviate.Client(
                    url = "http://localhost:8080", 
                    additional_headers = {
                        "X-Google-Api-Key": os.getenv("GOOGLE_API_KEY") 
                    }
                )
            elif api_key_provider=="OpenAI":
                client = weaviate.Client(
                    url = "http://localhost:8080", 
                    additional_headers = {
                        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY") 
                    }
                )
            elif api_key_provider not in {"Google", "OpenAI"}:
                raise ValueError("API key type must be 'Google' or 'OpenAI'.")
            
    def set_node_parser(self, parser):
        if parser == Node_parser.SEMANTIC:
            from llama_index.core.node_parser import (
                SemanticSplitterNodeParser,
            )
            self.Settings.node_parser = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95, 
                embed_model=self.Settings.embed_model
            )
            return parser
        elif parser == Node_parser.HIERARCHICAL:
            from llama_index.core.node_parser import (
                HierarchicalNodeParser, 
                TokenTextSplitter
            )
            text_splitter_ids = ["1024", "510"]
            text_splitter_map = {}
            for ids in text_splitter_ids:
                text_splitter_map[ids] = TokenTextSplitter(
                    chunk_size=int(ids),
                    chunk_overlap=200
                )
            
            self.Settings.node_parser = HierarchicalNodeParser.from_defaults(
                node_parser_ids=text_splitter_ids, 
                node_parser_map=text_splitter_map
            )
            return parser
        elif parser == None:
            return parser
        else:
            raise ValueError("Node parser must be one of: 'semantic', 'hierarchical'")
        
    def set_llm(self):
        pass

    def set_embed_model(self):
        pass
        
    def create_nodes(self):
        if self.doc_status:
            nodes = self.Settings.node_parser.get_nodes_from_documents(documents=self.documents)
            self.doc_store.add_documents(nodes)

            if self.node_parser == Node_parser.HIERARCHICAL:
                from llama_index.core.node_parser import (
                    get_leaf_nodes
                )
                return get_leaf_nodes(nodes)
            
            return nodes
        else:
            print("Pls load documnets first!")
            return None
