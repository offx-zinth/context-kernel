import logging
from typing import Any, List, Optional, Dict

# Moved from contextkernel.core_logic.llm_listener
class BaseMemorySystem:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{self.__class__.__name__} initialized.")

class RawCacheInterface(BaseMemorySystem):
    async def store(self, doc_id: str, data: Any) -> Optional[str]: raise NotImplementedError
    async def load(self, doc_id: str) -> Optional[Any]: raise NotImplementedError

class STMInterface(BaseMemorySystem):
    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError
    async def load_summary(self, summary_id: str) -> Optional[Any]: raise NotImplementedError

class LTMInterface(BaseMemorySystem):
    async def save_document(self, doc_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError

class GraphDBInterface(BaseMemorySystem):
    async def add_entities(self, entities: List[Any], document_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError
    async def add_relations(self, relations: List[Any], document_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError

# Attempting to import RetrievedItem directly. If this causes issues, will use string literal.
from contextkernel.core_logic.llm_retriever import RetrievedItem

class KeywordSearcherInterface(BaseMemorySystem): # Inheriting from BaseMemorySystem for common logger
    def initialize_searcher(self, config: Any) -> None: # Config type can be a specific Pydantic model later
        """Initializes the searcher with necessary configurations, e.g., sets up index path."""
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[RetrievedItem]:
        """Searches the keyword index for the given query."""
        raise NotImplementedError

    async def add_document(self, doc_id: str, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds a document to the keyword search index."""
        raise NotImplementedError
