import logging
import asyncio
import time
import os
import json
import hashlib # Added for SHA256 hashing
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings # For Config model

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None # type: ignore
    CrossEncoder = None # type: ignore

try:
    import numpy as np
except ImportError:
    np = None # type: ignore

try:
    import faiss # type: ignore
except ImportError:
    faiss = None # type: ignore

try:
    import networkx as nx # type: ignore
except ImportError:
    nx = None # type: ignore

try:
    from whoosh.index import create_in, open_dir, exists_in # type: ignore
    from whoosh.fields import Schema, TEXT, ID # type: ignore
    from whoosh.qparser import QueryParser # type: ignore
except ImportError:
    create_in = open_dir = exists_in = Schema = TEXT = ID = QueryParser = None # type: ignore

from .exceptions import EmbeddingError, ConfigurationError, MemoryAccessError, ExternalServiceError
from ..interfaces.protocols import LTMInterface, GraphDBInterface, KeywordSearcherInterface # Added KeywordSearcherInterface

logger = logging.getLogger(__name__)

# Config Model
class LLMRetrieverConfig(BaseSettings):
    embedding_model_name: Optional[str] = "all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None
    default_top_k: int = 10
    faiss_index_path: Optional[str] = None
    networkx_graph_path: Optional[str] = None
    # whoosh_index_dir: Optional[str] = "whoosh_index_path" # Removed, belongs to WhooshKeywordSearcherConfig
    cross_encoder_model_name: Optional[str] = None
    keyword_search_enabled: bool = True # This can signal if a keyword searcher should be used/injected

    model_config = {'env_prefix': 'LLM_RETRIEVER_'} # Pydantic V2 style for pydantic-settings

# Data Structures
class RetrievedItem(BaseModel):
    content: Any
    source: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RetrievalResponse(BaseModel):
    items: List[RetrievedItem] = Field(default_factory=list)
    retrieval_time_ms: Optional[float] = None
    message: Optional[str] = None

# Embedding Model
class HuggingFaceEmbeddingModel:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.logger = logging.getLogger(__name__ + ".HuggingFaceEmbeddingModel")
        self.model_name = model_name
        self.device = device
        self.model = None
        if SentenceTransformer is None:
            raise ConfigurationError("sentence-transformers library not installed.")
        if not self.model_name:
            raise ConfigurationError("model_name not provided for HuggingFaceEmbeddingModel.")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            raise EmbeddingError(f"Failed to load SentenceTransformer model '{self.model_name}': {e}") from e

    async def generate_embedding(self, text: str) -> List[float]:
        if self.model is None: raise EmbeddingError("Embedding model not available.")
        if not text or not isinstance(text, str): self.logger.warning("Invalid input for embedding."); return []
        try:
            embedding_array = await asyncio.to_thread(self.model.encode, text, convert_to_tensor=False)
            return embedding_array.tolist() if hasattr(embedding_array, 'tolist') else list(map(float, embedding_array))
        except Exception as e: raise EmbeddingError(f"Encoding failed for model '{self.model_name}': {e}") from e

# StubLTM and StubGraphDB definitions moved to contextkernel.tests.mocks.memory_stubs.py


# LLMRetriever
class LLMRetriever:
    def __init__(self,
                 retriever_config: LLMRetrieverConfig,
                 ltm_interface: LTMInterface,
                 stm_interface: Any, # Assuming STMInterface is defined or Any is acceptable
                 graphdb_interface: GraphDBInterface,
                 keyword_searcher: Optional[KeywordSearcherInterface] = None, # Added
                 query_llm: Any = None):
        self.retriever_config = retriever_config
        self.ltm = ltm_interface
        self.stm = stm_interface
        self.graph_db = graphdb_interface
        self.keyword_searcher = keyword_searcher # Added
        self.query_llm = query_llm
        self.logger = logger
        # self.whoosh_ix = None # Removed
        self.cross_encoder = None

        if not self.retriever_config.embedding_model_name and (self.retriever_config.retrieval_strategy in ["all", "vector_only"] if hasattr(self.retriever_config, 'retrieval_strategy') else True) : # check if strategy implies embedding use
             raise ConfigurationError("embedding_model_name is required for retriever if vector search is used.")
        try:
            if self.retriever_config.embedding_model_name:
                 self.embedding_model = HuggingFaceEmbeddingModel(model_name=self.retriever_config.embedding_model_name, device=self.retriever_config.embedding_device)
            else: self.embedding_model = None
        except (EmbeddingError, ConfigurationError) as e:
            self.logger.error(f"Retriever: Embedding model init failed: {e}", exc_info=True); raise

        if self.retriever_config.keyword_search_enabled:
            # Old Whoosh direct initialization logic removed.
            # Keyword searcher should be initialized externally and passed in.
            pass # Placeholder for any future logic if keyword_search_enabled means something else now.
        
        # Removed direct passing of whoosh_ix to ltm. This coupling is removed.
        # if hasattr(self.ltm, 'whoosh_ix'): self.ltm.whoosh_ix = self.whoosh_ix
        # if hasattr(self.ltm, 'retriever_config'): self.ltm.retriever_config = self.retriever_config


        if self.retriever_config.cross_encoder_model_name:
            if CrossEncoder is None: raise ConfigurationError("CrossEncoder configured but not installed.")
            try: self.cross_encoder = CrossEncoder(self.retriever_config.cross_encoder_model_name)
            except Exception as e: raise ConfigurationError(f"Failed to load CrossEncoder '{self.retriever_config.cross_encoder_model_name}': {e}") from e
        self.logger.info("LLMRetriever initialized.")

    async def _preprocess_and_embed_query(self, query: str, task_description: Optional[str]=None) -> List[float]:
        if not self.embedding_model: raise ConfigurationError("Embedding model not available for query embedding.")
        # Query expansion logic (placeholder)
        return await self.embedding_model.generate_embedding(query)

    async def _search_vector_store(self, q_embed: List[float], top_k: int, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        if not hasattr(self.ltm, 'search'): raise ConfigurationError("LTM interface missing 'search'.")
        try: return await self.ltm.search(query_embedding=q_embed, top_k=top_k, filters=filters)
        except Exception as e: raise MemoryAccessError(f"LTM search error: {e}") from e
        
    async def _search_graph_db(self, query: str, task: Optional[str]=None, top_k: int=5, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        if not hasattr(self.graph_db, 'search'): raise ConfigurationError("GraphDB interface missing 'search'.")
        try: return await self.graph_db.search(query=query, top_k=top_k, filters=filters, task_description=task) # Added task_description to call
        except Exception as e: raise MemoryAccessError(f"GraphDB search error: {e}") from e

    async def _search_keyword(self, query: str, top_k: int=5, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        if self.retriever_config.keyword_search_enabled and self.keyword_searcher:
            try:
                return await self.keyword_searcher.search(query, top_k, filters)
            except Exception as e:
                self.logger.error(f"Keyword searcher error: {e}", exc_info=True)
                # Optionally re-raise as MemoryAccessError or return empty list
                # For consistency with previous error handling:
                raise MemoryAccessError(f"Keyword searcher failed: {e}") from e
        return [] # Return empty list if not enabled or no searcher

    async def _consolidate_and_rank_results(self, query:str, results: List[List[RetrievedItem]], top_k_ce: int=20) -> List[RetrievedItem]:
        flat_results = [item for sublist in results for item in sublist]
        unique_results: Dict[str, RetrievedItem] = {}
        for item in flat_results:
            doc_id = item.metadata.get("doc_id")
            if doc_id:
                key = str(doc_id)
            else:
                content_str = str(item.content)
                key = hashlib.sha256(content_str.encode('utf-8')).hexdigest()

            if key not in unique_results or \
               (item.score is not None and (unique_results[key].score is None or item.score > unique_results[key].score)): # type: ignore
                unique_results[key] = item
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score or 0.0, reverse=True)

        if self.cross_encoder and sorted_results:
            to_rerank = sorted_results[:top_k_ce]
            pairs = [[query, str(item.content)] for item in to_rerank]
            try:
                ce_scores = await asyncio.to_thread(self.cross_encoder.predict, pairs, show_progress_bar=False) #type: ignore
                for item, score in zip(to_rerank, ce_scores): item.score = float(score)
                sorted_results = sorted(to_rerank, key=lambda x: x.score or 0.0, reverse=True) + sorted_results[top_k_ce:]
            except Exception as e: self.logger.error(f"CrossEncoder re-ranking failed: {e}", exc_info=True); # Non-critical, proceed with original sort
        return sorted_results

    async def retrieve(self, query: str, task_description: Optional[str]=None, top_k: Optional[int]=None, filters: Optional[Dict]=None, retrieval_strategy: str="all") -> RetrievalResponse:
        start_time, current_top_k = time.time(), top_k or self.retriever_config.default_top_k
        final_items, errors = [], []
        
        try: q_embed = await self._preprocess_and_embed_query(query, task_description) if self.embedding_model else []
        except EmbeddingError as e: q_embed, errors = [], [f"Query embedding failed: {e}"]
        
        task_error_messages: List[str] = [] # Initialize task_error_messages, was missing in original snippet for retrieve

        search_coros = []
        if retrieval_strategy in ["all", "vector_only"] and q_embed: search_coros.append(self._search_vector_store(q_embed, current_top_k, filters))
        if retrieval_strategy in ["all", "graph_only"]: search_coros.append(self._search_graph_db(query, task_description, current_top_k, filters)) # task_description was correctly here
        if retrieval_strategy in ["all", "keyword_only"] and self.retriever_config.keyword_search_enabled: # Added enabled check
            search_coros.append(self._search_keyword(query, current_top_k, filters))

        if search_coros:
            results_from_sources = await asyncio.gather(*search_coros, return_exceptions=True)
            processed_results: List[List[RetrievedItem]] = []
            # Combine errors from asyncio.gather (e.g. direct call failures) 
            # with task_error_messages (e.g. soft errors within search methods if they didn't raise)
            all_errors = errors # Start with embedding errors
            for i, res_or_exc in enumerate(results_from_sources):
                if isinstance(res_or_exc, Exception): 
                    all_errors.append(f"Search source {i} failed: {res_or_exc}")
                elif res_or_exc: 
                    processed_results.append(res_or_exc) # type: ignore
            
            # Add any task_error_messages that might have been populated by individual search methods
            # (though current search methods raise on error, this is for robustness if they change)
            all_errors.extend(task_error_messages)

            if processed_results: 
                final_items = await self._consolidate_and_rank_results(query, processed_results)
        
        msg = f"Retrieved {len(final_items)} items." + (f" Errors: {'; '.join(all_errors)}" if all_errors else "")
        return RetrievalResponse(items=final_items, retrieval_time_ms=(time.time()-start_time)*1000, message=msg)

# Example main (for direct execution if needed)
async def main_test():
    logging.basicConfig(level=logging.INFO)
    # Setup example configs and stubs here for a test run
    # ... (similar to previous main_test but with updated error handling in mind)
if __name__ == '__main__': asyncio.run(main_test())
