import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from contextkernel.core_logic.llm_retriever import LLMRetriever, LLMRetrieverConfig, RetrievedItem, RetrievalResponse
from contextkernel.core_logic.exceptions import ConfigurationError, EmbeddingError, MemoryAccessError

# Minimal stubs for interfaces if not wanting to mock every method
class MinimalLTMStub:
    async def search(self, query_embedding: list, top_k: int, filters: dict = None):
        return [] # Default empty result
    async def add_document(self, *args, **kwargs): pass

class MinimalGraphDBStub:
    async def search(self, query: str, top_k: int = 5, filters: dict = None, task_description: str = None):
        return [] # Default empty result
    async def add_node(self, *args, **kwargs): pass


class TestLLMRetrieverStrategies(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_ltm_interface = AsyncMock(spec=MinimalLTMStub)
        self.mock_graphdb_interface = AsyncMock(spec=MinimalGraphDBStub)
        self.mock_query_llm = AsyncMock() # Not heavily used in current retrieve, but part of constructor

        # Mock Embedding Model
        self.mock_embedding_model_instance = AsyncMock()
        self.mock_embedding_model_instance.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        self.patcher_embedding_model = patch('contextkernel.core_logic.llm_retriever.HuggingFaceEmbeddingModel', return_value=self.mock_embedding_model_instance)
        self.mock_hf_embedding_model_cls = self.patcher_embedding_model.start()

        # Mock Whoosh components if keyword search is enabled in config
        self.patcher_whoosh_open_dir = patch('contextkernel.core_logic.llm_retriever.open_dir')
        self.patcher_whoosh_exists_in = patch('contextkernel.core_logic.llm_retriever.exists_in', return_value=True) # Assume index exists
        self.patcher_whoosh_schema = patch('contextkernel.core_logic.llm_retriever.Schema')

        self.mock_whoosh_open_dir = self.patcher_whoosh_open_dir.start()
        self.mock_whoosh_exists_in = self.patcher_whoosh_exists_in.start()
        self.mock_whoosh_schema_cls = self.patcher_whoosh_schema.start()

        # Mock for QueryParser and searcher within _search_keyword
        self.mock_query_parser_instance = MagicMock()
        self.mock_query_parser_instance.parse = MagicMock()
        self.patcher_query_parser = patch('contextkernel.core_logic.llm_retriever.QueryParser', return_value=self.mock_query_parser_instance)
        self.mock_query_parser_cls = self.patcher_query_parser.start()

        self.mock_whoosh_searcher = MagicMock()
        self.mock_whoosh_searcher.search = MagicMock(return_value=[]) # Default empty keyword results

        # Ensure the Whoosh index mock can be used as a context manager
        mock_index_obj = MagicMock()
        mock_index_obj.searcher.return_value.__enter__.return_value = self.mock_whoosh_searcher
        self.mock_whoosh_open_dir.return_value = mock_index_obj


        # Default config enables all sources and graph_first strategy
        self.base_config = LLMRetrieverConfig(
            embedding_model_name="test-embed-model", # Needs a name to load embedding model
            whoosh_index_dir="dummy_whoosh_path" # Needs a path for Whoosh setup
        )

    def tearDown(self):
        self.patcher_embedding_model.stop()
        self.patcher_whoosh_open_dir.stop()
        self.patcher_whoosh_exists_in.stop()
        self.patcher_whoosh_schema.stop()
        self.patcher_query_parser.stop()

    async def test_retrieve_graph_only_strategy(self):
        config = self.base_config.model_copy(update={"default_retrieval_strategy": "graph_only"})
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        mock_graph_results = [RetrievedItem(content="graph data", source="graph_db_node", score=0.9)]
        self.mock_graphdb_interface.search = AsyncMock(return_value=mock_graph_results)

        response = await retriever.retrieve("test query", top_k=3)

        self.mock_graphdb_interface.search.assert_called_once_with(query="test query", top_k=3, filters=None) # task_description removed from call in graph_db.py
        self.mock_ltm_interface.search.assert_not_called()
        self.mock_whoosh_searcher.search.assert_not_called()
        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].content, "graph data")

    async def test_retrieve_graph_first_graph_has_enough_results(self):
        config = self.base_config.model_copy(update={
            "default_retrieval_strategy": "graph_first",
            "graph_search_top_k": 5,
            "default_top_k": 5 # Overall top_k
        })
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        mock_graph_results = [RetrievedItem(content=f"graph_data_{i}", source="graph", score=0.9-i*0.1) for i in range(5)]
        self.mock_graphdb_interface.search = AsyncMock(return_value=mock_graph_results)

        response = await retriever.retrieve("test query", top_k=5)

        self.mock_graphdb_interface.search.assert_called_once_with(query="test query", top_k=config.graph_search_top_k, filters=None) # task_description removed
        self.mock_ltm_interface.search.assert_not_called()
        self.mock_whoosh_searcher.search.assert_not_called()
        self.assertEqual(len(response.items), 5)

    async def test_retrieve_graph_first_fallback_to_vector_and_keyword(self):
        config = self.base_config.model_copy(update={
            "default_retrieval_strategy": "graph_first",
            "graph_search_top_k": 2, # Graph returns 1, needs more
            "default_top_k": 5,      # Overall top_k
            "enable_vector_search": True,
            "enable_keyword_search": True
        })
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        mock_graph_results = [RetrievedItem(content="graph_data_0", source="graph", score=0.9)]
        self.mock_graphdb_interface.search = AsyncMock(return_value=mock_graph_results)

        mock_vector_results = [RetrievedItem(content="vector_data_0", source="vector", score=0.8)]
        self.mock_ltm_interface.search = AsyncMock(return_value=mock_vector_results)

        # Mock Whoosh searcher for keyword results
        mock_keyword_hit = MagicMock()
        mock_keyword_hit.get.side_effect = lambda key: "keyword_data_0" if key == "content" else ("kw_doc_1" if key == "doc_id" else None)
        mock_keyword_hit.score = 0.7
        self.mock_whoosh_searcher.search.return_value = [mock_keyword_hit]


        response = await retriever.retrieve("test query", top_k=5)
        remaining_k = config.default_top_k - len(mock_graph_results) # 5 - 1 = 4

        self.mock_graphdb_interface.search.assert_called_once_with(query="test query", top_k=config.graph_search_top_k, filters=None)
        self.mock_embedding_model_instance.generate_embedding.assert_called_once_with("test query") # Called for vector search
        self.mock_ltm_interface.search.assert_called_once_with(query_embedding=[0.1,0.2,0.3], top_k=remaining_k, filters=None)

        self.mock_query_parser_instance.parse.assert_called_once_with("test query")
        self.mock_whoosh_searcher.search.assert_called_once() # Check if called, specific args depend on QueryParser output

        self.assertEqual(len(response.items), 3) # 1 graph, 1 vector, 1 keyword
        contents = [item.content for item in response.items]
        self.assertIn("graph_data_0", contents)
        self.assertIn("vector_data_0", contents)
        self.assertIn("keyword_data_0", contents)

    async def test_retrieve_vector_only_strategy(self):
        config = self.base_config.model_copy(update={"default_retrieval_strategy": "vector_only"})
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        mock_vector_results = [RetrievedItem(content="vector data", source="ltm_stub", score=0.85)]
        self.mock_ltm_interface.search = AsyncMock(return_value=mock_vector_results)

        response = await retriever.retrieve("test query", top_k=3)

        self.mock_ltm_interface.search.assert_called_once_with(query_embedding=[0.1,0.2,0.3], top_k=3, filters=None)
        self.mock_graphdb_interface.search.assert_not_called()
        self.mock_whoosh_searcher.search.assert_not_called()
        self.assertEqual(response.items[0].content, "vector data")

    async def test_retrieve_keyword_only_strategy(self):
        config = self.base_config.model_copy(update={"default_retrieval_strategy": "keyword_only"})
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        mock_keyword_hit = MagicMock()
        mock_keyword_hit.get.side_effect = lambda key: "keyword_data" if key == "content" else ("kw_doc_1" if key == "doc_id" else None)
        mock_keyword_hit.score = 0.75
        self.mock_whoosh_searcher.search.return_value = [mock_keyword_hit]

        response = await retriever.retrieve("test query", top_k=3)

        self.mock_query_parser_instance.parse.assert_called_once_with("test query")
        self.mock_whoosh_searcher.search.assert_called_once()
        self.mock_ltm_interface.search.assert_not_called()
        self.mock_graphdb_interface.search.assert_not_called()
        self.assertEqual(response.items[0].content, "keyword_data")

    async def test_retrieve_all_strategy_all_enabled(self):
        config = self.base_config.model_copy(update={
            "default_retrieval_strategy": "all",
            "enable_graph_search": True,
            "enable_vector_search": True,
            "enable_keyword_search": True
        })
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        self.mock_graphdb_interface.search = AsyncMock(return_value=[RetrievedItem(content="g",source="g")])
        self.mock_ltm_interface.search = AsyncMock(return_value=[RetrievedItem(content="v",source="v")])
        mock_keyword_hit = MagicMock(); mock_keyword_hit.get.return_value = "k"; mock_keyword_hit.score = 0.1
        self.mock_whoosh_searcher.search.return_value = [mock_keyword_hit]


        await retriever.retrieve("test query", top_k=5)

        self.mock_graphdb_interface.search.assert_called_once()
        self.mock_ltm_interface.search.assert_called_once()
        self.mock_whoosh_searcher.search.assert_called_once()

    async def test_retrieve_all_strategy_some_disabled(self):
        config = self.base_config.model_copy(update={
            "default_retrieval_strategy": "all",
            "enable_graph_search": True,
            "enable_vector_search": False, # Disable vector
            "enable_keyword_search": True
        })
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)
        # Reset mocks for this specific test
        self.mock_graphdb_interface.search = AsyncMock(return_value=[])
        self.mock_ltm_interface.search = AsyncMock(return_value=[])
        self.mock_whoosh_searcher.search.return_value = []


        await retriever.retrieve("test query", top_k=5)

        self.mock_graphdb_interface.search.assert_called_once()
        self.mock_ltm_interface.search.assert_not_called() # Vector is disabled
        self.mock_whoosh_searcher.search.assert_called_once()

    async def test_embedding_error_handling(self):
        config = self.base_config.model_copy(default_retrieval_strategy="vector_only")
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        self.mock_embedding_model_instance.generate_embedding = AsyncMock(side_effect=EmbeddingError("Embedding failed"))

        response = await retriever.retrieve("test query")
        self.assertEqual(len(response.items), 0)
        self.assertIn("Embedding failed", response.message)

    async def test_retrieve_consolidation_and_ranking(self):
        # Using "all" strategy for simplicity to get results from multiple sources
        config = self.base_config.model_copy(default_retrieval_strategy="all")
        retriever = LLMRetriever(config, self.mock_ltm_interface, None, self.mock_graphdb_interface, self.mock_query_llm)

        # Mock results - note scores are important for ranking
        self.mock_graphdb_interface.search = AsyncMock(return_value=[
            RetrievedItem(content="graph_item1", source="graph", score=0.9, metadata={"doc_id": "g1"}),
            RetrievedItem(content="common_item", source="graph", score=0.8, metadata={"doc_id": "common"})
        ])
        self.mock_ltm_interface.search = AsyncMock(return_value=[
            RetrievedItem(content="vector_item1", source="vector", score=0.85, metadata={"doc_id": "v1"}),
            RetrievedItem(content="common_item", source="vector", score=0.7, metadata={"doc_id": "common"}) # Lower score for common
        ])

        mock_kw_hit = MagicMock()
        mock_kw_hit.get.side_effect = lambda key: "keyword_item1" if key == "content" else ("kw1" if key == "doc_id" else None)
        mock_kw_hit.score = 0.77
        self.mock_whoosh_searcher.search.return_value = [mock_kw_hit]


        response = await retriever.retrieve("test query", top_k=3)

        self.assertEqual(len(response.items), 3) # Should be limited by top_k

        # Check order and deduplication (common_item from graph should win due to higher score)
        self.assertEqual(response.items[0].content, "graph_item1") # score 0.9
        self.assertEqual(response.items[1].content, "vector_item1") # score 0.85
        self.assertEqual(response.items[2].content, "common_item") # score 0.8 (from graph)
        self.assertEqual(response.items[2].source, "graph")


if __name__ == '__main__':
    unittest.main()
