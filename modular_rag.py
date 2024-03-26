from naive_rag import NaiveRAG
import os, weaviate

class ModularRAG(NaiveRAG):
    def __init__(self, llm=None, embed_model=None, node_parser=None, **kwargs):
        super().__init__(**kwargs)
        if llm is not None:
            self.Settings.llm=llm
        if embed_model is not None:
            self.Settings.embed_model = embed_model
        if node_parser is not None:
            self.Settings.node_parser = node_parser
    
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