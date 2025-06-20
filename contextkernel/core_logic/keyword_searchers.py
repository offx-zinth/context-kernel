import os
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel # For WhooshKeywordSearcherConfig

# Attempt to import Whoosh components
try:
    from whoosh.index import create_in, open_dir, exists_in, Index as WhooshIndex
    from whoosh.fields import Schema, TEXT, ID
    from whoosh.qparser import QueryParser
    from whoosh.writing import AsyncWriter
except ImportError:
    create_in = open_dir = exists_in = WhooshIndex = Schema = TEXT = ID = QueryParser = AsyncWriter = None

from ..interfaces.protocols import KeywordSearcherInterface
from .llm_retriever import RetrievedItem # Assuming RetrievedItem stays here for now
from .exceptions import ConfigurationError, MemoryAccessError

logger = logging.getLogger(__name__)

class WhooshKeywordSearcherConfig(BaseModel):
    whoosh_index_dir: str = "default_whoosh_index/" # Default path, should be configurable

class WhooshKeywordSearcher(KeywordSearcherInterface):
    def __init__(self):
        super().__init__() # For BaseMemorySystem logger via KeywordSearcherInterface
        self.whoosh_ix: Optional[WhooshIndex] = None
        self.schema = Schema(doc_id=ID(stored=True, unique=True), content=TEXT(stored=True))
        self.config: Optional[WhooshKeywordSearcherConfig] = None
        # self.logger is inherited from BaseMemorySystem

    def initialize_searcher(self, config: WhooshKeywordSearcherConfig): # Made synchronous
        self.config = config
        if not all([create_in, open_dir, exists_in, WhooshIndex, Schema, TEXT, ID, QueryParser, AsyncWriter]):
            raise ConfigurationError("Whoosh library not fully available or not installed.")
        if not self.config.whoosh_index_dir:
            raise ConfigurationError("Whoosh index directory not specified in config.")

        try:
            idx_dir = self.config.whoosh_index_dir
            if not os.path.exists(idx_dir):
                os.makedirs(idx_dir)
                self.logger.info(f"Created Whoosh index directory: {idx_dir}")

            if exists_in(idx_dir):
                self.whoosh_ix = open_dir(idx_dir)
                self.logger.info(f"Opened existing Whoosh index at '{idx_dir}'.")
            else:
                self.whoosh_ix = create_in(idx_dir, self.schema)
                self.logger.info(f"Created new Whoosh index at '{idx_dir}'.")
        except Exception as e:
            self.logger.error(f"Whoosh index initialization failed at '{self.config.whoosh_index_dir}': {e}", exc_info=True)
            raise ConfigurationError(f"Whoosh index init failed: {e}") from e

    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[RetrievedItem]:
        if not self.whoosh_ix:
            self.logger.error("Whoosh index not initialized before search.")
            raise MemoryAccessError("Whoosh index not initialized.")
        try:
            # Note: whoosh.searching.Searcher is thread-safe for searching.
            # For fully async, one might use asyncio.to_thread if searcher methods were blocking.
            # However, typical Whoosh searcher usage is synchronous within its context.
            with self.whoosh_ix.searcher() as searcher:
                q_parser = QueryParser("content", schema=self.whoosh_ix.schema)
                parsed_query = q_parser.parse(query)

                hits = searcher.search(parsed_query, limit=top_k)
                results = [
                    RetrievedItem(
                        content=hit.get("content", ""),
                        source="keyword_whoosh",
                        score=hit.score,
                        metadata={"doc_id": hit.get("doc_id")}
                    ) for hit in hits
                ]
                return results
        except Exception as e:
            self.logger.error(f"Whoosh keyword search failed for query '{query}': {e}", exc_info=True)
            raise MemoryAccessError(f"Whoosh keyword search failed: {e}") from e

    async def add_document(self, doc_id: str, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.whoosh_ix:
            self.logger.error("Whoosh index not initialized before adding document.")
            raise MemoryAccessError("Whoosh index not initialized.")
        try:
            # AsyncWriter is suitable for scenarios where multiple coroutines might add documents.
            writer = AsyncWriter(self.whoosh_ix)
            # Ensure doc_id is a string as Whoosh expects for unique ID fields.
            writer.add_document(doc_id=str(doc_id), content=text_content)
            await writer.commit() # Perform the commit asynchronously
            self.logger.debug(f"Document '{doc_id}' added to Whoosh index.")
        except Exception as e:
            self.logger.error(f"Whoosh add_document failed for document ID '{doc_id}': {e}", exc_info=True)
            raise MemoryAccessError(f"Whoosh add_document failed for '{doc_id}': {e}") from e
