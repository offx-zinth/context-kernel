import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None # type: ignore

try:
    import numpy as np
except ImportError:
    np = None # type: ignore

from pydantic import BaseSettings # Added for LLMRetrieverConfig

# Configuration Model for LLMRetriever
class LLMRetrieverConfig(BaseSettings):
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None
    default_top_k: int = 10

    class Config:
        env_prefix = 'LLM_RETRIEVER_' # e.g., LLM_RETRIEVER_EMBEDDING_MODEL_NAME

# --- Data Structures ---

class RetrievedItem(BaseModel):
    """
    Represents a single item retrieved from a memory source.
    """
    content: Any  # The actual retrieved data (e.g., text chunk, graph snippet, document)
    source: str   # e.g., "ltm", "graph_db", "stm_cache", "keyword_search"
    score: Optional[float] = None  # Relevance score, if applicable
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict) # Timestamps, doc ID, etc.

class RetrievalResponse(BaseModel):
    """
    Represents the overall response from a retrieval operation.
    """
    items: List[RetrievedItem] = Field(default_factory=list)
    retrieval_time_ms: Optional[float] = None
    message: Optional[str] = None # For warnings or additional info

# --- LLMRetriever Class ---

class LLMRetriever:
    """
    Retrieves relevant information from long-term memory (LTM), short-term memory (STM),
    and a graph database (GraphDB) based on a given query.
    """

    def __init__(self,
                 retriever_config: LLMRetrieverConfig, # New config parameter
                 ltm_interface,
                 stm_interface,
                 graphdb_interface,
                 query_llm=None):
        """
        Initializes the LLMRetriever.

        Args:
            retriever_config: Configuration object for the LLMRetriever.
            ltm_interface: An interface to the Long-Term Memory store.
            stm_interface: An interface to the Short-Term Memory store.
            graphdb_interface: An interface to the Graph Database.
            query_llm (optional): An LLM used for query manipulation (e.g., expansion, rewriting).
                                  Defaults to None.
        """
        self.retriever_config = retriever_config # Store the config object
        self.ltm = ltm_interface
        self.stm = stm_interface
        self.graph_db = graphdb_interface
        self.query_llm = query_llm
        self.logger = logging.getLogger(__name__)

        if SentenceTransformer is None:
            self.logger.error(
                "sentence-transformers library not installed. Embedding features will be unavailable. "
                "Please install with: pip install sentence-transformers"
            )
            self.embedding_model = None
        else:
            try:
                self.embedding_model = HuggingFaceEmbeddingModel(
                    model_name=self.retriever_config.embedding_model_name,
                    device=self.retriever_config.embedding_device
                )
                self.logger.info(f"LLMRetriever initialized with HuggingFaceEmbeddingModel: {self.retriever_config.embedding_model_name} on device: {self.retriever_config.embedding_device or 'default'}.")
            except Exception as e:
                self.logger.error(f"Failed to initialize HuggingFaceEmbeddingModel: {e}", exc_info=True)
                self.embedding_model = None

        self.logger.info(f"LLMRetriever initialized. Embedding model status noted above. Config: {self.retriever_config.model_dump_json(indent=2)}")


    async def _preprocess_and_embed_query(self, query: str, task_description: str = None):
        """
        Preprocesses the query (optional) and generates its vector embedding.

        Args:
            query (str): The input query string.
            task_description (str, optional): Additional context about the task. Defaults to None.

        Returns:
            list: The vector embedding of the query, or None if an error occurs.
        """
        self.logger.info(f"Received query: '{query}'")
        if task_description:
            self.logger.info(f"Task description: '{task_description}'")

        # Optional: Query expansion/rewriting using self.query_llm
        # if self.query_llm and task_description:
        #     # This is a placeholder for more sophisticated query manipulation
        #     try:
        #         # processed_query = await self.query_llm.refine_query(query, task_description)
        #         # self.logger.info(f"Refined query to: '{processed_query}'")
        #         # query = processed_query
        #         pass # Replace with actual call
        #     except Exception as e:
        #         self.logger.warning(f"Error during query refinement: {e}. Using original query.")
        # elif self.query_llm:
        #      # processed_query = await self.query_llm.refine_query(query)
        #      # self.logger.info(f"Refined query to: '{processed_query}'")
        #      # query = processed_query
        #      pass # Replace with actual call

        if not self.embedding_model:
            self.logger.error("Embedding model is not available (failed to initialize or sentence-transformers not installed).")
            return None

        if not hasattr(self.embedding_model, 'generate_embedding'):
            self.logger.error("Loaded embedding model does not have a 'generate_embedding' method.")
            # This case should ideally not happen if HuggingFaceEmbeddingModel is instantiated correctly
            # and an error wasn't caught, or if a different type of model was somehow assigned.
            return None

        try:
            query_embedding = await self.embedding_model.generate_embedding(query)
            if query_embedding:
                self.logger.info(f"Successfully generated embedding for query: '{query}'")
            else: # Should not happen if generate_embedding raises on error, but as a safeguard
                self.logger.error(f"generate_embedding returned None for query: '{query}'")
            return query_embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding for query '{query}': {e}", exc_info=True)
            return None

    async def _search_vector_store(self, query_embedding: list[float], top_k: int = 5, filters: dict = None) -> List[RetrievedItem]:
        """
        Searches the vector store (LTM) for relevant documents.
        (Adapts results to List[RetrievedItem] conceptually)

        Args:
            query_embedding (list[float]): The vector embedding of the query.
            top_k (int): The number of top results to retrieve.
            filters (dict, optional): Metadata filters to apply to the search. Defaults to None.

        Returns:
            List[RetrievedItem]: A list of search results from the LTM, adapted to RetrievedItem.
        """
        self.logger.info(f"Searching vector store (LTM) with top_k={top_k}, filters={filters}")

        if not hasattr(self.ltm, 'search'):
            self.logger.error("LTM interface does not have a 'search' method.")
            raise AttributeError("LTM interface must have an async method 'search'.")

        try:
            # Assuming self.ltm.search returns data that can be mapped to RetrievedItem
            # For now, we'll return a placeholder if actual conversion isn't done here
            raw_ltm_results = await self.ltm.search(query_embedding=query_embedding, top_k=top_k, filters=filters)
            self.logger.info(f"LTM search completed. Found {len(raw_ltm_results)} raw results.")

            # Placeholder: Convert raw_ltm_results to List[RetrievedItem]
            # This would involve knowing the structure of raw_ltm_results
            processed_results = []
            for res in raw_ltm_results:
                # Example: res could be a dict {'content': ..., 'score': ..., 'meta': ...}
                # Or it could be an object with attributes.
                # This is a conceptual mapping.
                processed_results.append(RetrievedItem(
                    content=res.get('content', res), # adapt as needed
                    source="ltm",
                    score=res.get('score'),
                    metadata=res.get('metadata', {})
                ))
            return processed_results
        except Exception as e:
            self.logger.error(f"Error during LTM search: {e}")
            # Depending on strategy, might want to return [] or raise
            # raise # Re-raising might be too harsh if other sources can compensate
            return [] # Return empty list on error


    async def _search_graph_db(self, query: str, task_description: str = None, top_k: int = 5, filters: dict = None) -> List[RetrievedItem]:
        """
        Searches the graph database for relevant entities and relationships.
        (Adapts results to List[RetrievedItem] conceptually)

        Args:
            query (str): The original query string (can be used for entity extraction or direct querying).
            task_description (str, optional): Additional context.
            top_k (int): The number of top results/paths to retrieve.
            filters (dict, optional): Filters to apply to the graph search. Defaults to None.

        Returns:
            List[RetrievedItem]: A list of search results from the GraphDB, adapted to RetrievedItem.
        """
        self.logger.info(f"Searching GraphDB with query: '{query}', top_k={top_k}, filters={filters}")

        if not hasattr(self.graph_db, 'search'):
            self.logger.error("GraphDB interface does not have a 'search' method.")
            raise AttributeError("GraphDB interface must have an async method 'search'.")

        try:
            # Assuming self.graph_db.search returns data that can be mapped
            raw_graph_results = await self.graph_db.search(query=query, task_description=task_description, top_k=top_k, filters=filters)
            self.logger.info(f"GraphDB search completed. Found {len(raw_graph_results)} raw results.")

            # Placeholder: Convert raw_graph_results to List[RetrievedItem]
            processed_results = []
            for res in raw_graph_results:
                processed_results.append(RetrievedItem(
                    content=res.get('content', res), # adapt as needed
                    source="graph_db",
                    score=res.get('score'),
                    metadata=res.get('metadata', {})
                ))
            return processed_results
        except Exception as e:
            self.logger.error(f"Error during GraphDB search: {e}")
            return [] # Return empty list on error

    async def _search_keyword(self, query: str, top_k: int = 5, filters: dict = None) -> List[RetrievedItem]:
        """
        Performs a keyword-based search (e.g., full-text search) across available memory.
        NOTE: This is a placeholder for future implementation.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to retrieve.
            filters (dict, optional): Filters to apply. Defaults to None.

        Returns:
            List[RetrievedItem]: An empty list, as this is not yet implemented.
        """
        self.logger.info(f"Keyword search (placeholder) for query: '{query}', top_k={top_k}, filters={filters}")
        # TODO: Implement keyword search logic. When implemented, ensure it returns List[RetrievedItem].
        # For now, returns an empty list.
        return []

    async def _consolidate_and_rank_results(self, results_collection: List[List[RetrievedItem]], strategy: str = "simple_aggregation") -> List[RetrievedItem]:
        """
        Consolidates results from different sources and ranks them.

        Args:
            results_collection (List[List[RetrievedItem]]): A list where each sublist contains
                                                            RetrievedItem objects from a source.
            strategy (str): The consolidation and ranking strategy to use.
                            Defaults to "simple_aggregation".

        Returns:
            List[RetrievedItem]: A single, consolidated (and potentially ranked) list of RetrievedItem.
        """
        self.logger.info(f"Consolidating and ranking results using strategy: '{strategy}'")

        raw_aggregated_results: List[RetrievedItem] = []
        if strategy == "simple_aggregation":
            for source_results_list in results_collection:
                if source_results_list:
                    raw_aggregated_results.extend(source_results_list)

            self.logger.info(f"Aggregated {len(raw_aggregated_results)} items before deduplication and sorting.")

            # Deduplication
            # Prefers items with higher scores. If scores are equal, keeps the first encountered.
            # Uses 'doc_id' or 'node_id' from metadata for identity.
            # Items without these specific IDs and without content matching (not implemented here) will be treated as unique.
            deduplicated_results_map: Dict[str, RetrievedItem] = {}
            items_without_identifiable_id: List[RetrievedItem] = []

            id_keys_to_check = ['doc_id', 'node_id'] # Common ID keys in metadata

            for item in raw_aggregated_results:
                item_id = None
                if item.metadata:
                    for key in id_keys_to_check:
                        if key in item.metadata:
                            item_id = str(item.metadata[key]) # Ensure ID is string for dict key
                            break

                if item_id:
                    if item_id not in deduplicated_results_map:
                        deduplicated_results_map[item_id] = item
                    else:
                        existing_item = deduplicated_results_map[item_id]
                        # Prioritize item with higher score
                        if item.score is not None and (existing_item.score is None or item.score > existing_item.score):
                            deduplicated_results_map[item_id] = item
                        # If scores are equal or new item score is None, keep existing (first encountered)
                else:
                    # For items without a standard ID, collect them separately.
                    # More advanced content-based deduplication could be a TODO here.
                    items_without_identifiable_id.append(item)

            consolidated_results = list(deduplicated_results_map.values()) + items_without_identifiable_id
            self.logger.info(f"Reduced to {len(consolidated_results)} items after deduplication based on metadata IDs.")
            self.logger.debug("TODO: Implement more advanced content-based deduplication if needed.")

            # Sorting
            # Sort by score (descending), items with None score come last.
            # Python's sort is stable, so relative order of items with same score (or None score) is preserved.
            consolidated_results.sort(key=lambda x: (x.score is None, -(x.score or float('-inf'))))

            self.logger.info("Results sorted by score (descending, None scores last).")
            self.logger.debug("TODO: Implement more sophisticated re-ranking (e.g., cross-encoder) if needed.")

        else:
            self.logger.warning(f"Unknown consolidation strategy: '{strategy}'. Returning raw aggregated (unprocessed) results.")
            # Fallback to simple aggregation without deduplication or specific sorting for unknown strategies
            for source_results_list in results_collection:
                 if source_results_list:
                    raw_aggregated_results.extend(source_results_list)
            consolidated_results = raw_aggregated_results # No processing for unknown strategy

        return consolidated_results

    async def retrieve(self, query: str, task_description: str = None, top_k: Optional[int] = None, filters: dict = None, retrieval_strategy: str = "all") -> RetrievalResponse:
        """
        Main method to retrieve relevant information based on a query.

        Args:
            query (str): The user's query.
            task_description (str, optional): Context about the task for which information is being retrieved.
            top_k (int): The desired number of results from each source.
            filters (dict, optional): Filters to apply during search (e.g., metadata, date ranges).
            retrieval_strategy (str): Defines which sources to query ("all", "vector_only", "graph_only", "keyword_only").

        Returns:
            RetrievalResponse: An object containing the retrieved items and metadata.
        """
        start_time = time.time()

        # Use default_top_k from config if top_k is not provided
        current_top_k = top_k if top_k is not None else self.retriever_config.default_top_k

        self.logger.info(
            f"Starting retrieval for query='{query}', task_description='{task_description}', "
            f"top_k={current_top_k} (using config default if not specified), filters={filters}, strategy='{retrieval_strategy}'"
        )
        final_message = None

        try:
            # 1. Preprocessing & Embedding
            query_embedding = await self._preprocess_and_embed_query(query, task_description)
            if query_embedding is None:
                self.logger.error("Failed to generate query embedding. Aborting retrieval.")
                return RetrievalResponse(items=[], message="Failed to generate query embedding.")

            # 2. Memory Interaction
            vector_store_results: List[RetrievedItem] = []
            graph_db_results: List[RetrievedItem] = []
            keyword_results: List[RetrievedItem] = []

            search_tasks = []

            if retrieval_strategy in ["all", "vector_only"]:
                search_tasks.append(self._search_vector_store(query_embedding, top_k=current_top_k, filters=filters))

            if retrieval_strategy in ["all", "graph_only"]:
                search_tasks.append(self._search_graph_db(query, task_description=task_description, top_k=current_top_k, filters=filters))

            if retrieval_strategy in ["all", "keyword_only"]: # Assuming keyword search is part of 'all'
                search_tasks.append(self._search_keyword(query, top_k=current_top_k, filters=filters))

            if not search_tasks:
                self.logger.warning(f"No search tasks identified for strategy '{retrieval_strategy}'. Returning empty response.")
                return RetrievalResponse(items=[], message=f"No search tasks for strategy '{retrieval_strategy}'.")

            # Execute searches. If 'all', potentially concurrently.
            # For other strategies, it will be a single task.
            self.logger.info(f"Executing {len(search_tasks)} search tasks based on strategy '{retrieval_strategy}'.")

            # asyncio.gather will run them concurrently if there are multiple
            all_source_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results from asyncio.gather, handling potential exceptions
            # Each result in all_source_results should be List[RetrievedItem] or an Exception
            processed_source_results_temp: List[Union[List[RetrievedItem], Exception]] = await asyncio.gather(*search_tasks, return_exceptions=True)

            processed_source_results: List[List[RetrievedItem]] = []
            task_error_messages = []
            for i, res_or_exc in enumerate(processed_source_results_temp):
                if isinstance(res_or_exc, Exception):
                    self.logger.error(f"Search task {i} (type: {search_tasks[i].__name__ if hasattr(search_tasks[i], '__name__') else 'unknown'}) failed: {res_or_exc}")
                    processed_source_results.append([]) # Add empty list for failed task
                    task_error_messages.append(f"Task {i} failed: {str(res_or_exc)[:100]}.") # Truncate long errors
                elif res_or_exc is None: # Should not happen if search methods return [] on error
                     self.logger.warning(f"Search task {i} returned None. Using empty list.")
                     processed_source_results.append([])
                else: # Should be List[RetrievedItem]
                    processed_source_results.append(res_or_exc)

            if task_error_messages:
                final_message = "Some search tasks failed. " + " ".join(task_error_messages)

            # Assign results based on the strategy and order of tasks added
            current_idx = 0
            if retrieval_strategy in ["all", "vector_only"]:
                if current_idx < len(processed_source_results): vector_store_results = processed_source_results[current_idx]
                current_idx +=1
            if retrieval_strategy in ["all", "graph_only"]:
                if current_idx < len(processed_source_results): graph_db_results = processed_source_results[current_idx]
                current_idx += 1
            if retrieval_strategy in ["all", "keyword_only"]:
                if current_idx < len(processed_source_results): keyword_results = processed_source_results[current_idx]
                # current_idx += 1 # No more consumers after this one


            # 3. Consolidation and Ranking
            all_results_collection: List[List[RetrievedItem]] = [
                vector_store_results, graph_db_results, keyword_results
            ]
            # Filter out any sub-lists that might be None if a search wasn't run (though current logic populates with [])
            all_results_collection = [res_list for res_list in all_results_collection if res_list is not None]

            final_items = await self._consolidate_and_rank_results(all_results_collection)

            # 4. Output Formatting & Logging
            retrieval_duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Retrieval completed in {retrieval_duration_ms:.2f} ms. Returning {len(final_items)} items.")

            return RetrievalResponse(
                items=final_items,
                retrieval_time_ms=retrieval_duration_ms,
                message=final_message if final_message else f"Successfully retrieved {len(final_items)} items."
            )

        except AttributeError as ae: # Specific error for missing methods on interfaces
            self.logger.error(f"AttributeError during retrieval: {ae}. This might indicate a misconfigured component.")
            return RetrievalResponse(items=[], retrieval_time_ms=(time.time() - start_time) * 1000, message=f"AttributeError: {ae}")
        except Exception as e:
            self.logger.error(f"Unexpected error during retrieval: {e}", exc_info=True)
            return RetrievalResponse(items=[], retrieval_time_ms=(time.time() - start_time) * 1000, message=f"Unexpected error: {e}")


# Example Usage (Illustrative - to be removed or moved to tests/examples)
if __name__ == '__main__':
    # This is placeholder code for demonstration.
    # Actual interfaces and models would be needed.

    # Mocking interfaces and models for the sake of example
    class MockInterface:
        def __init__(self, name):
            self.name = name
            self.logger = logging.getLogger(f"MockInterface.{name}")
            self.logger.info(f"MockInterface {name} initialized.")

        def search(self, query_embedding, top_k):
            self.logger.info(f"{self.name} received search request.")
            return [f"Result from {self.name} for query_embedding"]

    class MockEmbeddingModel:
        def __init__(self):
            self.logger = logging.getLogger("MockEmbeddingModel")
            self.logger.info("MockEmbeddingModel initialized.")

        def embed(self, text):
            self.logger.info(f"Embedding text: '{text}'")
            return [0.1, 0.2, 0.3] # Dummy embedding

    logging.basicConfig(level=logging.INFO)

    # Instantiate mock components
    mock_ltm = MockInterface("LTM")
    mock_stm = MockInterface("STM")
    mock_graphdb = MockInterface("GraphDB")
    mock_embed_model = MockEmbeddingModel()

    # Instantiate the retriever
    retriever = LLMRetriever(
        ltm_interface=mock_ltm,
        stm_interface=mock_stm,
        graphdb_interface=mock_graphdb,
        embedding_model=mock_embed_model
    )

    retriever.logger.info("LLMRetriever instance created for example.")

    # Example of how a search might be initiated (actual search method not yet implemented)
    # query_text = "What is context kernel?"
    # query_embedding = retriever.embedding_model.embed(query_text) # This would need an async call if using the new stubs
    # results = retriever.search(query_embedding, top_k=5) # This would need an async call
    # print(f"Search results: {results}")
    print("LLMRetriever structure created. Example usage (if run directly) would show logger messages. Stubs are defined below.")

# --- Stub Implementations for Dependencies (StubEmbeddingModel is removed, others enhanced) ---

class HuggingFaceEmbeddingModel:
    """
    Generates embeddings using a Hugging Face Sentence Transformer model.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.logger = logging.getLogger(__name__ + ".HuggingFaceEmbeddingModel")
        self.model_name = model_name
        self.device = device
        self.model = None

        if SentenceTransformer is None:
            self.logger.error(
                "sentence-transformers library not installed. "
                "HuggingFaceEmbeddingModel cannot function."
            )
            return # Model remains None

        try:
            self.logger.info(f"Loading SentenceTransformer model: {self.model_name} on device: {self.device or 'default'}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
            self.model = None # Ensure model is None on failure

    async def generate_embedding(self, text: str) -> List[float]:
        if self.model is None:
            self.logger.error("Embedding model not loaded. Cannot generate embedding.")
            return []

        self.logger.debug(f"Generating embedding for text (first 50 chars): '{text[:50]}...'")
        try:
            # SentenceTransformer.encode is synchronous and CPU/GPU-bound.
            # Run it in a separate thread to avoid blocking the asyncio event loop.
            embedding = await asyncio.to_thread(
                self.model.encode, text, convert_to_tensor=False
            )
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            self.logger.error(f"Error during text encoding with SentenceTransformer: {e}", exc_info=True)
            return []


class StubLTM:
    """
    Enhanced stub implementation for a Long-Term Memory interface with in-memory vector search.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".StubLTM")
        self.documents: List[RetrievedItem] = []
        self.embeddings: Optional[np.ndarray] = None
        if np is None:
            self.logger.warning("Numpy not installed. StubLTM vector search functionality will be limited/unavailable.")
        self.logger.info("StubLTM initialized (enhanced with in-memory store).")

    async def add_document(self, item: RetrievedItem, embedding: List[float]):
        """Adds a document and its embedding to the in-memory store."""
        if np is None:
            self.logger.error("Numpy not available, cannot add document with embedding to StubLTM.")
            return

        self.documents.append(item)
        np_embedding = np.array(embedding, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = np_embedding.reshape(1, -1)
        else:
            self.embeddings = np.concatenate((self.embeddings, np_embedding.reshape(1, -1)), axis=0)
        self.logger.info(f"Added document (source: {item.source}) to StubLTM. Total docs: {len(self.documents)}")

    def _calculate_cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        # Ensure query_embedding is 1D for dot product with 2D doc_embeddings
        query_embedding_1d = query_embedding.flatten()

        # Normalize embeddings to unit vectors
        query_norm = query_embedding_1d / np.linalg.norm(query_embedding_1d)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Calculate cosine similarity (dot product of normalized vectors)
        return np.dot(doc_norms, query_norm.T).flatten()

    async def search(self, query_embedding: List[float], top_k: int, filters: Optional[Dict] = None) -> List[RetrievedItem]:
        self.logger.info(f"Performing LTM search. Query embedding (first 3): {query_embedding[:3]}, top_k={top_k}, filters={filters}")
        if filters:
            self.logger.info(f"StubLTM received filters but does not currently apply them: {filters}")

        if np is None or self.embeddings is None or len(self.documents) == 0:
            self.logger.warning("Numpy not available or no documents/embeddings in StubLTM. Returning empty list.")
            return []

        query_emb_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        try:
            similarities = await asyncio.to_thread(
                self._calculate_cosine_similarity, query_emb_np, self.embeddings
            )
        except Exception as e:
            self.logger.error(f"Error during similarity calculation in StubLTM: {e}", exc_info=True)
            return []

        actual_top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:actual_top_k]

        results = []
        for i in top_indices:
            item = self.documents[i]
            results.append(RetrievedItem(
                content=item.content,
                source=item.source, # Keep original source
                score=float(similarities[i]),
                metadata=item.metadata
            ))

        self.logger.info(f"StubLTM search found {len(results)} items.")
        return results

class StubGraphDB:
    """
    Enhanced stub implementation for a Graph Database interface with in-memory storage.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".StubGraphDB")
        self.nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> properties
        self.relations: List[Dict[str, Any]] = [] # {'subject_id': 'id1', 'object_id': 'id2', 'type': 'REL_TYPE', 'properties': {}}
        self.logger.info("StubGraphDB initialized (enhanced with in-memory store).")

    async def add_node(self, node_id: str, properties: Dict[str, Any]):
        """Adds a node to the in-memory graph."""
        if node_id in self.nodes:
            self.logger.warning(f"Node '{node_id}' already exists in StubGraphDB. Updating properties.")
        self.nodes[node_id] = properties
        self.logger.info(f"Added/Updated node '{node_id}' in StubGraphDB with properties: {properties}")

    async def add_relation(self, subject_id: str, object_id: str, type: str, properties: Optional[Dict[str, Any]] = None):
        """Adds a relation between two nodes."""
        if subject_id not in self.nodes or object_id not in self.nodes:
            self.logger.error(f"Cannot add relation in StubGraphDB: Subject '{subject_id}' or Object '{object_id}' does not exist.")
            return

        relation_to_add = {
            "subject_id": subject_id,
            "object_id": object_id,
            "type": type,
            "properties": properties or {}
        }
        # Avoid duplicate relations for simplicity in this stub
        if relation_to_add not in self.relations:
            self.relations.append(relation_to_add)
            self.logger.info(f"Added relation in StubGraphDB: {subject_id} -[{type}]-> {object_id}")
        else:
            self.logger.info(f"Relation {subject_id} -[{type}]-> {object_id} already exists in StubGraphDB.")


    async def search(self, query: str, task_description: Optional[str] = None, top_k: int = 5, filters: Optional[Dict] = None) -> List[RetrievedItem]:
        self.logger.info(f"Performing GraphDB search. Query: '{query}', Task: {task_description}, Top_k: {top_k}, Filters: {filters}")
        if filters:
            self.logger.info(f"StubGraphDB received filters but does not currently apply them: {filters}")

        results: List[RetrievedItem] = []

        # 1. Check if query is a node_id
        if query in self.nodes:
            node_data = self.nodes[query]
            results.append(RetrievedItem(
                content={"node_id": query, "properties": node_data},
                source="graph_db_stub_node_id_match",
                score=1.0,
                metadata={"query_type": "node_id_lookup"}
            ))
            if len(results) >= top_k: return results[:top_k]

        # 2. Check if query matches a node property (e.g., "name:SomeName")
        if ":" in query:
            try:
                prop_key, prop_value = query.split(":", 1)
                prop_key = prop_key.strip()
                prop_value = prop_value.strip() # Basic parsing, could be more robust
                for node_id, properties in self.nodes.items():
                    if str(properties.get(prop_key)) == prop_value: # Compare as string for simplicity
                        results.append(RetrievedItem(
                            content={"node_id": node_id, "properties": properties},
                            source="graph_db_stub_property_match",
                            score=0.9,
                            metadata={"matched_property": prop_key}
                        ))
                        if len(results) >= top_k: return results[:top_k]
            except ValueError:
                self.logger.debug(f"Query '{query}' in StubGraphDB contains ':' but not in key:value format for property search.")

        # 3. Check for relations involving the query as a subject or object ID
        related_info_for_node = []
        for rel in self.relations:
            if rel["subject_id"] == query or rel["object_id"] == query:
                related_info_for_node.append(rel)

        if related_info_for_node:
             results.append(RetrievedItem(
                content={"node_id_queried": query, "relations_found": related_info_for_node},
                source="graph_db_stub_relations_match",
                score=0.8,
                metadata={"query_type": "node_relations_lookup"}
            ))

        self.logger.info(f"StubGraphDB search found {len(results)} items matching query '{query}'.")
        return results[:top_k]


class StubQueryLLM:
    """
    Stub implementation for an LLM used for query manipulation and result re-ranking.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".StubQueryLLM")
        self.logger.info("StubQueryLLM initialized.")

    async def expand_query(self, query: str, task_description: Optional[str] = None) -> str:
        self.logger.info(f"Performing stub query expansion for: '{query}', task: {task_description}")
        return f"{query} (expanded_stub)"

    async def rerank_results(self, query:str, results: List[RetrievedItem]) -> List[RetrievedItem]:
        self.logger.info(f"Performing stub re-ranking for query '{query}' on {len(results)} results.")
        # Simple re-ranking: reverse the list and slightly adjust scores
        reranked_results = []
        for i, item in enumerate(reversed(results)):
            new_score = item.score * 1.1 if item.score is not None else 0.5 # Give some score if None
            reranked_results.append(RetrievedItem(
                content=item.content,
                source=item.source,
                score=round(min(new_score, 1.0), 2), # Cap score at 1.0
                metadata={**item.metadata, "reranked_by": "StubQueryLLM"}
            ))
        return reranked_results

# Example of how to use the stubs (can be run with `python -m contextkernel.core_logic.llm_retriever` if structure allows)
async def main_test():
    # Configure basic logging for the test run
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__) # Get a logger for the main_test function itself
    logger.info("--- Running LLMRetriever with Stubs ---")

    # Instantiate stubs
    stub_embedding_model = StubEmbeddingModel()
    stub_ltm = StubLTM()
    stub_graph_db = StubGraphDB()
    stub_query_llm = StubQueryLLM() # Optional, can be None

    # Instantiate the retriever with stubs
    # For STM, we can pass None or a simple mock if its interface is also defined/needed.
    # For this test, assuming STM is not critically needed or its methods aren't called by current strategies.

    # Create a default config for the test
    default_config = LLMRetrieverConfig()

    retriever = LLMRetriever(
        retriever_config=default_config,
        ltm_interface=stub_ltm,
        stm_interface=None,
        graphdb_interface=stub_graph_db,
        # embedding_model_name and embedding_device are now part of retriever_config
        query_llm=stub_query_llm
    )

    logger.info("--- Test Case 1: 'all' strategy ---")
    response_all = await retriever.retrieve(
        query="What is the Context Kernel?",
        task_description="User is asking a general question.",
        top_k=3, # This will override default_top_k from config for this call
        retrieval_strategy="all"
    )
    logger.info(f"Response (all): {response_all.model_dump_json(indent=2)}") # Pydantic v2

    logger.info("--- Test Case 2: 'vector_only' strategy, using default top_k ---")
    response_vector = await retriever.retrieve(
        query="Find similar documents to X.",
        task_description="User wants semantic search.",
        # top_k not specified, should use default_top_k from config
        retrieval_strategy="vector_only"
    )
    logger.info(f"Response (vector_only): {response_vector.model_dump_json(indent=2)}")

    logger.info("--- Test Case 3: 'graph_only' strategy ---")
    response_graph = await retriever.retrieve(
        query="Who is connected to Node Y?",
        task_description="User is exploring connections.",
        top_k=2,
        retrieval_strategy="graph_only"
    )
    logger.info(f"Response (graph_only): {response_graph.model_dump_json(indent=2)}")

    logger.info("--- Test Case 4: Query that might not generate embedding (testing error path) ---")
    # To test embedding failure, the stub would need a specific trigger.
    # For now, we assume it always succeeds. If generate_embedding returned None:
    # query_embedding = await retriever._preprocess_and_embed_query("force_error_in_embed", "")
    # assert query_embedding is None , "Embedding should fail for this special query"
    # response_embed_fail = await retriever.retrieve(query="force_error_in_embed", retrieval_strategy="vector_only")
    # logger.info(f"Response (embed_fail): {response_embed_fail.json(indent=2)}")


    # Example of using query_llm for query expansion (not directly in retrieve yet)
    if retriever.query_llm:
        original_query = "original query"
        expanded = await retriever.query_llm.expand_query(original_query)
        logger.info(f"Query expansion example: '{original_query}' -> '{expanded}'")

        # Example of reranking (not directly in retrieve's main path yet for re-ranking)
        if response_all.items:
            reranked_items = await retriever.query_llm.rerank_results(query="What is the Context Kernel?", results=response_all.items)
            logger.info(f"Reranking example (first item score before: {response_all.items[0].score}, after: {reranked_items[0].score if reranked_items else 'N/A'})")


if __name__ == '__main__':
    # Remove old mock classes as they are replaced by stubs
    # class MockInterface: ... (removed)
    # class MockEmbeddingModel: ... (removed)

    # The old main part was just for basic instantiation.
    # The new async main_test() provides a more comprehensive test.
    # Commenting out for now as it requires a full setup or more robust mocking.
    # asyncio.run(main_test())
    pass
