# This file will contain stub implementations of memory system interfaces for testing.
import logging
import os
import asyncio
from typing import Any, Dict, Optional, List

try:
    import numpy as np
except ImportError:
    np = None

try:
    import faiss # type: ignore
except ImportError:
    faiss = None

try:
    import networkx as nx # type: ignore
except ImportError:
    nx = None

# Protocol interfaces that these stubs implement
from contextkernel.interfaces.protocols import RawCacheInterface, STMInterface, LTMInterface, GraphDBInterface

# Data structures and configs possibly used by or returned by stubs
from contextkernel.core_logic.llm_retriever import RetrievedItem, LLMRetrieverConfig
from contextkernel.core_logic.exceptions import ConfigurationError, MemoryAccessError


class StubRawCache(RawCacheInterface):
    def __init__(self):
        super().__init__() # For BaseMemorySystem logger
        self.cache: Dict[str, Any] = {}
    async def store(self, doc_id: str, data: Any) -> Optional[str]:
        self.cache[doc_id] = data
        return doc_id
    async def load(self, doc_id: str) -> Optional[Any]:
        return self.cache.get(doc_id)

class StubSTM(STMInterface):
    def __init__(self):
        super().__init__() # For BaseMemorySystem logger
        self.cache: Dict[str, Any] = {}
    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: Optional[Dict[str, Any]]=None) -> None:
        self.cache[summary_id] = {"summary": summary_obj, "metadata": metadata or {}}
    async def load_summary(self, summary_id: str) -> Optional[Any]:
        return self.cache.get(summary_id)

class StubLTM(LTMInterface):
    def __init__(self, faiss_index_path: Optional[str] = None, whoosh_ix: Any = None, retriever_config: Optional[LLMRetrieverConfig] = None):
        super().__init__() # For BaseMemorySystem logger
        self.faiss_index_path = faiss_index_path
        self.index: Optional[Any] = None # faiss.Index
        self.doc_id_to_internal_idx: Dict[str, int] = {}
        self.internal_idx_to_doc_item: Dict[int, RetrievedItem] = {}
        self._next_internal_idx = 0
        # self.whoosh_ix = whoosh_ix # Removed: Whoosh interaction is now decoupled
        # self.retriever_config = retriever_config # Removed: No longer needed for Whoosh in StubLTM

        if np is None: raise ConfigurationError("Numpy not installed for StubLTM.")
        if faiss is None: self.logger.warning("FAISS library not installed; FAISS features disabled in StubLTM.")

        if self.faiss_index_path and os.path.exists(self.faiss_index_path) and faiss:
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                self.logger.info(f"FAISS index loaded from {self.faiss_index_path}, ntotal: {self.index.ntotal}")
            except Exception as e: raise MemoryAccessError(f"Failed to load FAISS index '{self.faiss_index_path}': {e}") from e

    async def save_document(self, doc_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        if np is None: raise ConfigurationError("Numpy not available for StubLTM.save_document.")
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
                # Ensure 'source' is available in metadata if RetrievedItem expects it.
                # Forcing a default source for the stub.
                item_meta.setdefault('source', 'ltm_stub')
                self.internal_idx_to_doc_item[internal_id] = RetrievedItem(content=text_content, source=item_meta["source"], metadata=item_meta)
                self.doc_id_to_internal_idx[str(doc_id)] = internal_id
                self._next_internal_idx += 1
            except Exception as e: raise MemoryAccessError(f"FAISS add_with_ids failed for '{doc_id}': {e}") from e

        # Whoosh indexing logic removed from StubLTM.
        # If documents need to be in a keyword search index, it should be handled by
        # explicitly calling the KeywordSearcherInterface.add_document method.

    async def search(self, query_embedding: List[float], top_k: int, filters: Optional[Dict]=None) -> List[RetrievedItem]:
        # This is a method specific to this stub's current implementation, not directly from LTMInterface.
        # LTMInterface itself does not define a search method.
        # This search method is used by LLMRetriever's _search_vector_store.
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

    async def save_index(self, path: Optional[str]=None): # Specific to this stub
        save_p = path or self.faiss_index_path
        if not save_p: raise ConfigurationError("No path for FAISS index save.")
        if not (faiss and self.index): raise ConfigurationError("FAISS/index not available for save.")
        try: faiss.write_index(self.index, save_p); self.logger.info(f"FAISS index saved to {save_p}.")
        except Exception as e: raise MemoryAccessError(f"Failed to save FAISS index to '{save_p}': {e}") from e

class StubGraphDB(GraphDBInterface):
    def __init__(self, networkx_graph_path: Optional[str] = None):
        super().__init__() # For BaseMemorySystem logger
        if nx is None: raise ConfigurationError("NetworkX library not installed.")
        self.graph = nx.Graph()
        self.networkx_graph_path = networkx_graph_path # Store for saving
        if networkx_graph_path and os.path.exists(networkx_graph_path):
            try:
                if networkx_graph_path.endswith(".gml"): self.graph = nx.read_gml(networkx_graph_path)
                elif networkx_graph_path.endswith(".graphml"): self.graph = nx.read_graphml(networkx_graph_path)
                else: self.logger.warning(f"Unsupported graph format: {networkx_graph_path}")
                self.logger.info(f"NetworkX graph loaded from {networkx_graph_path}.")
            except Exception as e: raise MemoryAccessError(f"Failed to load NetworkX graph '{networkx_graph_path}': {e}") from e

    async def add_entities(self, entities: List[Any], document_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> None:
        try:
            for entity_data in entities:
                if not isinstance(entity_data, dict):
                    self.logger.warning(f"Skipping non-dict entity: {entity_data}"); continue
                node_id = entity_data.get('id')
                if not node_id: self.logger.warning(f"Skipping entity with no id: {entity_data}"); continue
                properties = entity_data.get('properties', {})
                # Include common metadata if provided and not overridden by entity properties
                if metadata: properties = {**metadata, **properties}
                if document_id: properties['document_id'] = document_id
                self.graph.add_node(node_id, **properties)
        except Exception as e: raise MemoryAccessError(f"Failed to add entities: {e}") from e

    async def add_relations(self, relations: List[Any], document_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> None:
        try:
            for relation_data in relations:
                if not isinstance(relation_data, dict):
                    self.logger.warning(f"Skipping non-dict relation: {relation_data}"); continue

                s_id = relation_data.get('subject_id', relation_data.get('s_id'))
                o_id = relation_data.get('object_id', relation_data.get('o_id'))
                rel_type = relation_data.get('type', relation_data.get('verb'))
                props = relation_data.get('properties', {})

                if not (s_id and o_id and rel_type):
                    self.logger.warning(f"Skipping incomplete relation: {relation_data}"); continue

                attrs = {**props, 'type': rel_type}
                if metadata: attrs = {**metadata, **attrs}
                if document_id: attrs['document_id'] = document_id

                self.graph.add_edge(s_id, o_id, **attrs)
        except Exception as e: raise MemoryAccessError(f"Failed to add relations: {e}") from e

    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict]=None, task_description: Optional[str]=None) -> List[RetrievedItem]:
        # This search method is specific to this stub, used by LLMRetriever's _search_graph_db.
        # GraphDBInterface itself does not define a search method.
        results: List[RetrievedItem] = []
        try:
            if self.graph.has_node(query): # Prioritize direct node match
                results.append(RetrievedItem(content={"id": query, "data": self.graph.nodes[query]}, source="graph_db_node_direct_match", metadata=self.graph.nodes[query]))

            # Basic property search (very naive) - search node attributes
            for node, data in self.graph.nodes(data=True):
                if len(results) >= top_k: break
                if query in str(data.values()): # Check if query string is in any attribute value
                    if not any(r.content["id"] == node for r in results): # Avoid duplicates from direct match
                         results.append(RetrievedItem(content={"id": node, "data": data}, source="graph_db_node_property_match", metadata=data))

            # Search edge attributes if still haven't reached top_k
            if len(results) < top_k:
                for u, v, data in self.graph.edges(data=True):
                    if len(results) >= top_k: break
                    if query in str(data.values()): # Check if query string is in any attribute value of an edge
                        # Content could be the edge itself or related nodes
                        edge_content = {"source_node": u, "target_node": v, "relation_data": data}
                        results.append(RetrievedItem(content=edge_content, source="graph_db_edge_property_match", metadata=data))

            return results[:top_k]
        except Exception as e: raise MemoryAccessError(f"NetworkX search failed for '{query}': {e}") from e

    async def save_graph(self, path: Optional[str]=None): # Specific to this stub
        save_p = path or self.networkx_graph_path
        if not save_p: raise ConfigurationError("No path for NetworkX graph save.")
        try:
            if save_p.endswith(".gml"): nx.write_gml(self.graph, save_p)
            elif save_p.endswith(".graphml"): nx.write_graphml(self.graph, save_p)
            else: self.logger.warning(f"Unsupported graph save format: {save_p}")
        except Exception as e: raise MemoryAccessError(f"Failed to save NetworkX graph to '{save_p}': {e}") from e
