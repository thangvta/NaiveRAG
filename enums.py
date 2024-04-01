from enum import Enum

class Node_parser(Enum):
    DEFAULT = 'default'
    HIERARCHICAL = 'hierarchical'
    SEMANTIC = 'semantic'

class Api_key(Enum):
    GOOGLE = 'Google'
    OPENAI = 'OpenAI'

class LLM(Enum):
    pass

class RAG_Framework(Enum):
    DEFAULT = 'default'
    CORRECTIVE = 'corrective'
    SELF = 'self'

