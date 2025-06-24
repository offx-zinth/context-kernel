import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from contextkernel.memory_system import MemoryKernel
from contextkernel.utils.config import AppSettings, RedisConfig, Neo4jConfig, VectorDBConfig, EmbeddingConfig, NLPServiceConfig, FileSystemConfig, S3Config

# Helper to create a default AppSettings for mocking
def create_mock_appsettings():
    settings = AppSettings(
        redis_config=RedisConfig(),
        neo4j_config=Neo4jConfig(),
        vector_db_config=VectorDBConfig(params={"index_path": "dummy_faiss.index", "dimension": 384}), # LTM needs params
        embedding_config=EmbeddingConfig(model_name="all-MiniLM-L6-v2"), # For LTM/GraphIndexer
        nlp_service_config=NLPServiceConfig(model="en_core_web_sm"), # For GraphIndexer/STM
        filesystem_config=FileSystemConfig(base_path="dummy_fs_path"), # For LTM
        s3_config=S3Config(bucket_name="dummy_s3_bucket") # For LTM (if configured)
    )
    return settings

class TestMemoryKernel(unittest.IsolatedAsyncioTestCase):

    @patch('contextkernel.memory_system.RedisClient', new_callable=AsyncMock)
    @patch('contextkernel.memory_system.AsyncGraphDatabase.driver', new_callable=MagicMock)
    @patch('contextkernel.memory_system.LTM', new_callable=AsyncMock)
    @patch('contextkernel.memory_system.STM', new_callable=AsyncMock)
    @patch('contextkernel.memory_system.GraphDB', new_callable=AsyncMock)
    @patch('contextkernel.memory_system.GraphIndexer', new_callable=AsyncMock)
    @patch('contextkernel.memory_system.RawCache', new_callable=AsyncMock)
    async def asyncSetUp(self, MockRawCache, MockGraphIndexer, MockGraphDB, MockSTM, MockLTM, mock_neo4j_driver_static, mock_redis_client_static):
        # Prevent _initialize_clients from trying to create real clients
        # by patching the client classes themselves at the source.
        # The mock_redis_client_static and mock_neo4j_driver_static are for the MemoryKernel._initialize_clients part

        self.app_settings = create_mock_appsettings()

        # Mock clients that MemoryKernel._initialize_clients would create
        self.mock_redis_client_instance = AsyncMock()
        mock_redis_client_static.return_value = self.mock_redis_client_instance

        self.mock_neo4j_driver_instance = AsyncMock()
        mock_neo4j_driver_static.return_value = self.mock_neo4j_driver_instance

        # Mock placeholder clients created by _create_placeholder_client
        # We can patch _create_placeholder_client itself or ensure components get mocks.
        # For these tests, the components (LTM, GraphIndexer, etc.) are already fully mocked.

        # Mock component instances
        self.mock_raw_cache = MockRawCache.return_value
        self.mock_graph_db = MockGraphDB.return_value
        self.mock_ltm = MockLTM.return_value
        self.mock_graph_indexer = MockGraphIndexer.return_value
        self.mock_stm = MockSTM.return_value

        # Ensure GraphIndexer has a mock nlp_processor for get_context entity extraction
        self.mock_graph_indexer.nlp_processor = MagicMock() # spaCy model mock

        # Reset singleton instance for each test
        MemoryKernel._instance = None
        MemoryKernel._clients_initialized = False # Reset this flag too
        self.kernel = MemoryKernel.get_instance(app_settings=self.app_settings)

        # Ensure that the mocked components are assigned to the kernel instance
        self.kernel.raw_cache = self.mock_raw_cache
        self.kernel.graph_db = self.mock_graph_db
        self.kernel.ltm = self.mock_ltm
        self.kernel.graph_indexer = self.mock_graph_indexer
        self.kernel.stm = self.mock_stm


    async def test_get_context_flow(self):
        query = "test query about AI"
        session_id = "test_session_123"
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_ltm_results = [{"memory_id": "ltm1", "text_content": "LTM content about AI", "score": 0.9}]
        mock_stm_results = [{"role": "user", "content": "Previous turn about AI"}]

        # Mocking entity extraction from query
        mock_spacy_doc = MagicMock()
        mock_spacy_doc.ents = [MagicMock(text="AI", label_="ORG")]
        self.kernel.graph_indexer.nlp_processor.return_value = mock_spacy_doc

        mock_graph_db_results = [{"e": {"name": "AI", "type": "Concept"}, "relations": []}]

        self.mock_ltm.generate_embedding.return_value = mock_query_embedding
        self.mock_ltm.retrieve_relevant_memories.return_value = mock_ltm_results
        self.mock_stm.get_recent_turns.return_value = mock_stm_results
        self.mock_graph_db.cypher_query.return_value = mock_graph_db_results

        context = await self.kernel.get_context(query, session_id=session_id)

        self.mock_ltm.generate_embedding.assert_called_once_with(query)
        self.mock_ltm.retrieve_relevant_memories.assert_called_once_with(query_embedding=mock_query_embedding, top_k=5)
        self.mock_stm.get_recent_turns.assert_called_once_with(session_id=session_id, num_turns=10)

        # Assert spaCy processor was called via run_in_executor (harder to directly assert run_in_executor usage)
        self.kernel.graph_indexer.nlp_processor.assert_called_once_with(query)
        self.mock_graph_db.cypher_query.assert_called_once() # Or more specific if query is stable

        self.assertEqual(context["query"], query)
        self.assertEqual(context["retrieved_ltm_items"], mock_ltm_results)
        self.assertEqual(context["recent_stm_turns"], mock_stm_results)
        self.assertEqual(context["related_graph_entities"], mock_graph_db_results)
        self.assertIn("Query embedding generated successfully", context["synthesis_log"][0])

    async def test_get_context_partial_failures(self):
        query = "test query"
        self.mock_ltm.generate_embedding.return_value = [0.1, 0.2]
        self.mock_ltm.retrieve_relevant_memories.side_effect = Exception("LTM retrieval failed")
        self.mock_stm.get_recent_turns.return_value = [] # Successful empty return
        self.kernel.graph_indexer.nlp_processor.return_value = MagicMock(ents=[]) # No entities
        self.mock_graph_db.cypher_query.return_value = []

        context = await self.kernel.get_context(query, session_id="s1")

        self.assertIn("Error retrieving from LTM: LTM retrieval failed", context["synthesis_log"])
        self.assertEqual(context["retrieved_ltm_items"], []) # Should be empty on failure
        self.assertEqual(context["recent_stm_turns"], []) # STM part should still work

    async def test_store_context_flow(self):
        session_id = "test_session_store"
        test_uuid = str(uuid.uuid4())
        data_to_store = {
            "chunk_id": "store_chunk1",
            "text_content": "This is important text content.",
            "metadata": {"source": "test_source"},
            "ephemeral_data": {"temp_key": "temp_value"},
            "turn_data": {"role": "user", "content": "User turn for STM"}
        }
        mock_embedding = [0.3, 0.4, 0.5]

        self.mock_graph_indexer.process_memory_chunk.return_value = {"status": "success", "chunk_id": "store_chunk1"}
        self.mock_ltm.generate_embedding.return_value = mock_embedding
        self.mock_ltm.store_memory_chunk.return_value = f"ltm_{data_to_store['chunk_id']}" # LTM returns its own ID

        with patch('uuid.uuid4', return_value=test_uuid): # Mock uuid for predictable ephemeral key
            success = await self.kernel.store_context(data_to_store, session_id=session_id)

        self.assertTrue(success)
        self.mock_raw_cache.set.assert_called_once_with(
            key=f"ephemeral_{data_to_store['chunk_id']}",
            value=data_to_store["ephemeral_data"],
            ttl_seconds=3600
        )
        self.mock_stm.add_turn.assert_called_once_with(session_id=session_id, turn_data=data_to_store["turn_data"])
        self.mock_graph_indexer.process_memory_chunk.assert_called_once_with(
            chunk_id=data_to_store["chunk_id"],
            chunk_data={"text_content": data_to_store["text_content"], **data_to_store["metadata"]}
        )
        self.mock_ltm.generate_embedding.assert_called_once_with(data_to_store["text_content"])
        self.mock_ltm.store_memory_chunk.assert_called_once_with(
            chunk_id=data_to_store["chunk_id"],
            text_content=data_to_store["text_content"],
            embedding=mock_embedding,
            metadata=data_to_store["metadata"]
        )

    async def test_store_context_error_handling(self):
        self.mock_graph_indexer.process_memory_chunk.side_effect = Exception("GraphIndexer failed")

        data_to_store = {"text_content": "Error case content"}
        success = await self.kernel.store_context(data_to_store)

        self.assertFalse(success)
        # Check logs or specific error messages if store_context provides more detailed status

    async def test_boot_shutdown_calls_dependencies(self):
        # Boot
        await self.kernel.boot()
        self.mock_raw_cache.boot.assert_called_once()
        self.mock_graph_db.boot.assert_called_once()
        self.mock_ltm.boot.assert_called_once()
        self.mock_graph_indexer.boot.assert_called_once()
        self.mock_stm.boot.assert_called_once()
        # Also check Redis/Neo4j client pings if they are exposed or part of component boot
        self.mock_redis_client_instance.ping.assert_called_once()
        self.mock_neo4j_driver_instance.verify_connectivity.assert_called_once()


        # Shutdown
        await self.kernel.shutdown()
        self.mock_raw_cache.shutdown.assert_called_once()
        self.mock_graph_db.shutdown.assert_called_once()
        self.mock_ltm.shutdown.assert_called_once()
        self.mock_graph_indexer.shutdown.assert_called_once()
        self.mock_stm.shutdown.assert_called_once()
        # Check client close calls
        self.mock_redis_client_instance.close.assert_called_once()
        self.mock_neo4j_driver_instance.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
