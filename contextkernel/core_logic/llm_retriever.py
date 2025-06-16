# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import asyncio
import time
import os
import json
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

logger = logging.getLogger(__name__)

# Config Model
class LLMRetrieverConfig(BaseSettings):
    embedding_model_name: Optional[str] = "all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None
    default_top_k: int = 10
    faiss_index_path: Optional[str] = None
    networkx_graph_path: Optional[str] = None
    whoosh_index_dir: Optional[str] = "whoosh_index_path"
    cross_encoder_model_name: Optional[str] = None
    keyword_search_enabled: bool = True

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

# Stub LTM
class StubLTM:
    def __init__(self, faiss_index_path: Optional[str] = None, whoosh_ix: Any = None, retriever_config: Optional[LLMRetrieverConfig] = None):
        self.logger = logging.getLogger(__name__ + ".StubLTM")
        self.faiss_index_path = faiss_index_path
        self.index: Optional[Any] = None # faiss.Index
        self.doc_id_to_internal_idx: Dict[str, int] = {}
        self.internal_idx_to_doc_item: Dict[int, RetrievedItem] = {}
        self._next_internal_idx = 0
        self.whoosh_ix = whoosh_ix
        self.retriever_config = retriever_config

        if np is None: raise ConfigurationError("Numpy not installed for StubLTM.")
        if faiss is None: self.logger.warning("FAISS library not installed; FAISS features disabled in StubLTM.")
        
        if self.faiss_index_path and os.path.exists(self.faiss_index_path) and faiss:
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                self.logger.info(f"FAISS index loaded from {self.faiss_index_path}, ntotal: {self.index.ntotal}")
                # TODO: Mappings loading logic
            except Exception as e: raise MemoryAccessError(f"Failed to load FAISS index '{self.faiss_index_path}': {e}") from e

    async def add_document(self, doc_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        if np is None: raise ConfigurationError("Numpy not available for StubLTM.add_document.")
        if not doc_id: self.logger.error("doc_id missing."); return
        if str(doc_id) in self.doc_id_to_internal_idx: self.logger.warning(f"Doc ID '{doc_id}' exists."); return

        if faiss and self.index is None and embedding:
            try: self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(np.array(embedding).shape[-1]))
            except Exception as e: raise MemoryAccessError(f"FAISS index creation failed: {e}") from e
        
        if faiss and self.index and embedding:
            try:
                np_embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
                internal_id = self._next_internal_idx
                self.index.add_with_ids(np_embedding, np.array([internal_id]))
                item_meta = metadata or {}; item_meta['doc_id'] = str(doc_id)
                self.internal_idx_to_doc_item[internal_id] = RetrievedItem(content=text_content, source=item_meta.get("source", "ltm_stub"), metadata=item_meta)
                self.doc_id_to_internal_idx[str(doc_id)] = internal_id
                self._next_internal_idx += 1
            except Exception as e: raise MemoryAccessError(f"FAISS add_with_ids failed for '{doc_id}': {e}") from e

        if self.whoosh_ix and self.retriever_config and self.retriever_config.keyword_search_enabled and text_content:
            try:
                writer = self.whoosh_ix.writer()
                writer.add_document(doc_id=str(doc_id), content=text_content)
                writer.commit()
            except Exception as e: self.logger.error(f"Whoosh add failed for '{doc_id}': {e}", exc_info=True)


    async def search(self, query_embedding: List[float], top_k: int, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        if not (faiss and self.index and self.index.ntotal > 0): return []
        if np is None: raise ConfigurationError("Numpy not available for StubLTM.search.")
        try:
            distances, internal_indices = await asyncio.to_thread(self.index.search, np.array(query_embedding, dtype=np.float32).reshape(1, -1), top_k)
            results = []
            for i in range(internal_indices.shape[1]):
                internal_idx, dist = internal_indices[0, i], distances[0, i]
                if internal_idx != -1 and internal_idx in self.internal_idx_to_doc_item:
                    item = self.internal_idx_to_doc_item[internal_idx]
                    results.append(RetrievedItem(content=item.content, source=item.source, score=1.0/(1.0+float(dist)), metadata=item.metadata))
            return results
        except Exception as e: raise MemoryAccessError(f"FAISS search failed: {e}") from e

    async def save_index(self, path: Optional[str]=None):
        save_p = path or self.faiss_index_path
        if not save_p: raise ConfigurationError("No path for FAISS index save.")
        if not (faiss and self.index): raise ConfigurationError("FAISS/index not available for save.")
        try: faiss.write_index(self.index, save_p); self.logger.info(f"FAISS index saved to {save_p}.")
        except Exception as e: raise MemoryAccessError(f"Failed to save FAISS index to '{save_p}': {e}") from e

# Stub GraphDB
class StubGraphDB:
    def __init__(self, networkx_graph_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__ + ".StubGraphDB")
        if nx is None: raise ConfigurationError("NetworkX library not installed.")
        self.graph = nx.Graph()
        if networkx_graph_path and os.path.exists(networkx_graph_path):
            try:
                if networkx_graph_path.endswith(".gml"): self.graph = nx.read_gml(networkx_graph_path)
                elif networkx_graph_path.endswith(".graphml"): self.graph = nx.read_graphml(networkx_graph_path)
                else: self.logger.warning(f"Unsupported graph format: {networkx_graph_path}")
                self.logger.info(f"NetworkX graph loaded from {networkx_graph_path}.")
            except Exception as e: raise MemoryAccessError(f"Failed to load NetworkX graph '{networkx_graph_path}': {e}") from e

    async def add_node(self, node_id: str, **properties: Any):
        try: self.graph.add_node(node_id, **properties)
        except Exception as e: raise MemoryAccessError(f"Failed to add node '{node_id}': {e}") from e
    async def add_relation(self, s_id: str, o_id: str, type: Optional[str]=None, **props: Any):
        attrs = {**props, 'type': type} if type else props
        try: self.graph.add_edge(s_id, o_id, **attrs)
        except Exception as e: raise MemoryAccessError(f"Failed to add relation {s_id}-{o_id}: {e}") from e
    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        # Simplified search, not implementing full GQL or Cypher.
        results: List[RetrievedItem] = []
        try:
            if self.graph.has_node(query):
                results.append(RetrievedItem(content={"id": query, "data": self.graph.nodes[query]}, source="graph_db_node"))
            # Basic property search (very naive)
            for node, data in self.graph.nodes(data=True):
                if query in str(data.values()):
                     results.append(RetrievedItem(content={"id": node, "data": data}, source="graph_db_property"))
            return results[:top_k]
        except Exception as e: raise MemoryAccessError(f"NetworkX search failed for '{query}': {e}") from e
    async def save_graph(self, path: Optional[str]=None):
        save_p = path or self.networkx_graph_path
        if not save_p: raise ConfigurationError("No path for NetworkX graph save.")
        try:
            if save_p.endswith(".gml"): nx.write_gml(self.graph, save_p)
            elif save_p.endswith(".graphml"): nx.write_graphml(self.graph, save_p)
            else: self.logger.warning(f"Unsupported graph save format: {save_p}")
        except Exception as e: raise MemoryAccessError(f"Failed to save NetworkX graph to '{save_p}': {e}") from e


# LLMRetriever
class LLMRetriever:
    def __init__(self, retriever_config: LLMRetrieverConfig, ltm_interface: Any, stm_interface: Any, graphdb_interface: Any, query_llm: Any = None):
        self.retriever_config = retriever_config
        self.ltm = ltm_interface
        self.stm = stm_interface # Not used in current retrieve, but kept for interface
        self.graph_db = graphdb_interface
        self.query_llm = query_llm # For query expansion, etc.
        self.logger = logger
        self.whoosh_ix = None
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
            if not (Schema and QueryParser and ID and TEXT and create_in and open_dir and exists_in):
                raise ConfigurationError("Whoosh library not fully available, but keyword search enabled.")
            if not self.retriever_config.whoosh_index_dir:
                raise ConfigurationError("Whoosh index directory not specified, but keyword search enabled.")
            try:
                schema = Schema(doc_id=ID(stored=True, unique=True), content=TEXT(stored=True))
                idx_dir = self.retriever_config.whoosh_index_dir
                if not os.path.exists(idx_dir): os.makedirs(idx_dir)
                self.whoosh_ix = open_dir(idx_dir) if exists_in(idx_dir) else create_in(idx_dir, schema)
            except Exception as e: raise ConfigurationError(f"Whoosh index init failed at '{idx_dir}': {e}") from e
        
        if hasattr(self.ltm, 'whoosh_ix'): self.ltm.whoosh_ix = self.whoosh_ix
        if hasattr(self.ltm, 'retriever_config'): self.ltm.retriever_config = self.retriever_config

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
        try: return await self.graph_db.search(query=query, task_description=task, top_k=top_k, filters=filters)
        except Exception as e: raise MemoryAccessError(f"GraphDB search error: {e}") from e

    async def _search_keyword(self, query: str, top_k: int=5, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        if not (self.retriever_config.keyword_search_enabled and self.whoosh_ix): return []
        try:
            with self.whoosh_ix.searcher() as searcher:
                q = QueryParser("content", schema=self.whoosh_ix.schema).parse(query)
                hits = searcher.search(q, limit=top_k)
                return [RetrievedItem(content=h.get("content",""), source="keyword", score=h.score, metadata={"doc_id":h.get("doc_id")}) for h in hits]
        except Exception as e: raise MemoryAccessError(f"Whoosh keyword search failed: {e}") from e

    async def _consolidate_and_rank_results(self, query:str, results: List[List[RetrievedItem]], top_k_ce: int=20) -> List[RetrievedItem]:
        flat_results = [item for sublist in results for item in sublist]
        unique_results: Dict[str, RetrievedItem] = {}
        for item in flat_results: # Simple dedupe by content preview or doc_id
            key = str(item.metadata.get("doc_id") or str(item.content)[:50])
            if key not in unique_results or (item.score and (unique_results[key].score is None or item.score > unique_results[key].score)): # type: ignore
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
        if retrieval_strategy in ["all", "graph_only"]: search_coros.append(self._search_graph_db(query, task_description, current_top_k, filters))
        if retrieval_strategy in ["all", "keyword_only"]: search_coros.append(self._search_keyword(query, current_top_k, filters))

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
