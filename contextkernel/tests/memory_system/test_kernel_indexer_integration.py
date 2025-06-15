import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from contextkernel.memory_system import MemoryKernel, GraphIndexer
from contextkernel.memory_system.graph_db import GraphDB # For type hinting and mocking GraphDB methods
from contextkernel.memory_system.raw_cache import RawCache # For type hinting and mocking RawCache
from contextkernel.utils.config import AppSettings, RedisConfig, Neo4jConfig, VectorDBConfig, EmbeddingConfig, NLPServiceConfig, FileSystemConfig, S3Config

# Helper to create AppSettings for this integration test
def create_integration_appsettings():
    return AppSettings(
        redis_config=RedisConfig(db=9), # Use a different DB for test isolation if real Redis is hit
        neo4j_config=Neo4jConfig(uri="bolt://mockhost:7687"), # Will be mocked
        vector_db_config=VectorDBConfig(params={"index_path": "dummy_ltm_faiss.index", "dimension": 384}),
        embedding_config=EmbeddingConfig(model_name='all-MiniLM-L6-v2'), # Real model for GraphIndexer
        nlp_service_config=NLPServiceConfig(model='en_core_web_sm'), # Real model for GraphIndexer
        filesystem_config=FileSystemConfig(base_path="dummy_ltm_fs_path"),
        s3_config=S3Config()
    )

class TestKernelIndexerIntegration(unittest.IsolatedAsyncioTestCase):

    @patch('contextkernel.memory_system.RedisClient', new_callable=AsyncMock) # Mocks Redis for Kernel's own RawCache
    @patch('contextkernel.memory_system.AsyncGraphDatabase.driver') # Mocks Neo4j driver for Kernel's GraphDB
    @patch('contextkernel.memory_system.LTM', new_callable=AsyncMock) # Mock LTM
    @patch('contextkernel.memory_system.STM', new_callable=AsyncMock) # Mock STM
    # Note: We are NOT mocking GraphIndexer here at the MemoryKernel dependency level initially
    # We will also not mock GraphDB for GraphIndexer, but pass a mock GraphDB instance to it.
    async def asyncSetUp(self, MockSTM, MockLTM, mock_neo4j_driver_static, mock_redis_client_static):
        self.app_settings = create_integration_appsettings()

        # Setup mocks for clients MemoryKernel would create
        mock_redis_client_static.return_value = AsyncMock()
        mock_neo4j_driver_static.return_value = AsyncMock()

        # Mock GraphDB that will be passed to the real GraphIndexer
        self.mock_graph_db_for_indexer = AsyncMock(spec=GraphDB)
        # If create_node isn't directly on AsyncMock, need to add it
        # For spec'd mocks, methods from the spec are available.
        # self.mock_graph_db_for_indexer.create_node = AsyncMock(return_value=True)
        # self.mock_graph_db_for_indexer.create_edge = AsyncMock(return_value=True)
        # self.mock_graph_db_for_indexer.get_node = AsyncMock(return_value=None) # Important for idempotency checks
        # self.mock_graph_db_for_indexer.get_edge = AsyncMock(return_value=None)


        # Mock RawCache for GraphIndexer's embedding cache
        self.mock_raw_cache_for_indexer = AsyncMock(spec=RawCache)
        # self.mock_raw_cache_for_indexer.get = AsyncMock(return_value=None) # Simulate cache miss
        # self.mock_raw_cache_for_indexer.set = AsyncMock()
        # self.mock_raw_cache_for_indexer.boot = AsyncMock(return_value=True)
        # self.mock_raw_cache_for_indexer.shutdown = AsyncMock(return_value=True)


        # Mock other components for MemoryKernel
        self.mock_ltm_instance = MockLTM.return_value
        self.mock_stm_instance = MockSTM.return_value

        # This RawCache is for MemoryKernel's own direct use if any (not used in store_context path for GraphIndexer)
        # Or, it's the one that gets passed to LTM and GraphIndexer.
        # The setup for GraphIndexer below will ensure it gets the mock_raw_cache_for_indexer.
        self.mock_kernel_raw_cache_instance = AsyncMock(spec=RawCache)


        # Patch _initialize_clients to control client creation within MemoryKernel
        # And to inject our real GraphIndexer with its mocked DB/Cache

        # Store original method
        self.original_initialize_clients = MemoryKernel._initialize_clients

        # Define what our patched _initialize_clients should do
        def patched_initialize_clients(mk_instance, app_settings_param):
            # Setup Redis and Neo4j clients as normal (they are mocked at class level)
            mk_instance.redis_client = mock_redis_client_static()
            mk_instance.neo4j_driver = mock_neo4j_driver_static()

            # Create REAL GraphIndexer but with MOCKED GraphDB and MOCKED RawCache
            # Ensure GraphIndexer's dependencies (NLP/Embedding models) can load
            try {
                mk_instance.graph_indexer = GraphIndexer(
                    graph_db=self.mock_graph_db_for_indexer, # Pass the mock DB here
                    nlp_config=app_settings_param.nlp_service_config, # Real config
                    embedding_config=app_settings_param.embedding_config, # Real config
                    embedding_cache=self.mock_raw_cache_for_indexer # Pass the mock Cache here
                )
            } except Exception as e: {
                print(f"ERROR during GraphIndexer instantiation in test: {e}")
                # Log this properly or raise to fail test if GraphIndexer can't init
                # For now, this might happen if models aren't downloadable in test env.
                # A real test setup would ensure models are pre-cached or use very small test models.
                raise e # Fail fast if GraphIndexer cannot be created
            }


            # Other components are mocked as per the class-level patches
            mk_instance.raw_cache = self.mock_kernel_raw_cache_instance # Kernel's own, or passed to LTM
            mk_instance.graph_db = AsyncMock(spec=GraphDB) # Kernel's GraphDB, separate from GIs
            mk_instance.ltm = self.mock_ltm_instance
            mk_instance.stm = self.mock_stm_instance

            # Placeholders for other clients MemoryKernel might create
            mk_instance.shared_nlp_client = AsyncMock()
            mk_instance.shared_embedding_client = AsyncMock()
            mk_instance.vector_db_client = AsyncMock()
            mk_instance.raw_content_store_client = AsyncMock()
            MemoryKernel._clients_initialized = True


        self.init_clients_patcher = patch.object(MemoryKernel, '_initialize_clients', autospec=True)
        self.mock_initialize_clients = self.init_clients_patcher.start()
        self.mock_initialize_clients.side_effect = patched_initialize_clients

        MemoryKernel._instance = None # Reset singleton
        MemoryKernel._clients_initialized = False
        self.kernel = MemoryKernel.get_instance(app_settings=self.app_settings)
        # At this point, self.kernel.graph_indexer should be our real GraphIndexer
        # and self.kernel.graph_indexer.graph_db should be self.mock_graph_db_for_indexer

    async def asyncTearDown(self):
        self.init_clients_patcher.stop()
        MemoryKernel._initialize_clients = self.original_initialize_clients # Restore original
        MemoryKernel._instance = None
        MemoryKernel._clients_initialized = False

    async def test_store_context_triggers_graph_indexer_processing(self):
        """
        Tests that MemoryKernel.store_context correctly calls GraphIndexer,
        and GraphIndexer performs NLP and attempts to store graph elements.
        """
        logger.info("Ensure spaCy model 'en_core_web_sm' and SentenceTransformer 'all-MiniLM-L6-v2' are available/downloadable.")

        text_content = "Alice Alpha works at Wonderland Inc. Bob Beta is her colleague."
        metadata = {"source": "integration_test_doc"}
        data_to_store = {
            "text_content": text_content,
            "metadata": metadata,
            "chunk_id": "integ_test_chunk1"
        }

        # Mock the methods of the mocked GraphDB instance that GraphIndexer uses
        self.mock_graph_db_for_indexer.get_node.return_value = None # Simulate nodes don't exist
        self.mock_graph_db_for_indexer.create_node.return_value = True
        self.mock_graph_db_for_indexer.get_edge.return_value = None
        self.mock_graph_db_for_indexer.create_edge.return_value = True

        # Mock LTM calls made by MemoryKernel.store_context after GraphIndexer
        self.mock_ltm_instance.generate_embedding.return_value = [0.01] * 384 # Dummy embedding
        self.mock_ltm_instance.store_memory_chunk.return_value = "ltm_dummy_id_123"

        await self.kernel.boot() # Boot kernel and its components (including real GraphIndexer)

        success = await self.kernel.store_context(data_to_store)
        self.assertTrue(success)

        # Assertions:
        # 1. GraphIndexer's process_memory_chunk was called (implicitly tested by checking DB calls)
        # 2. GraphDB's create_node was called by the real GraphIndexer
        self.mock_graph_db_for_indexer.create_node.assert_called()

        # Check some calls to create_node for expected entities
        # Entity node IDs are like: f"entity_{entity_type.lower()}_{entity_text_clean}"
        # spaCy for "Alice Alpha" (PERSON), "Wonderland Inc" (ORG), "Bob Beta" (PERSON)

        # Extract call arguments to inspect them
        create_node_calls = self.mock_graph_db_for_indexer.create_node.call_args_list
        created_node_ids = [call[1]['node_id'] for call in create_node_calls] # call[1] is kwargs

        # Expected entity node IDs (simplified, actual cleaning might differ slightly)
        # GraphIndexer cleans entity text: .replace(' ', '_').replace('.', '').lower()
        expected_entity_ids = [
            "entity_person_alice_alpha",
            "entity_organization_wonderland_inc",
            "entity_person_bob_beta"
        ]

        logger.debug(f"Created node IDs by GraphIndexer: {created_node_ids}")

        for expected_id in expected_entity_ids:
            self.assertTrue(any(expected_id in called_id for called_id in created_node_ids), f"Expected entity node {expected_id} not created.")

        # Check if a chunk node was created (GraphIndexer creates one)
        self.assertTrue(any(data_to_store["chunk_id"] in called_id for called_id in created_node_ids), "Chunk node not created.")

        # Check edge creation (MENTIONS_ENTITY)
        self.mock_graph_db_for_indexer.create_edge.assert_called()
        create_edge_calls = self.mock_graph_db_for_indexer.create_edge.call_args_list
        logger.debug(f"Create_edge calls: {create_edge_calls}")

        # Ensure edges were created from the chunk node to the entity nodes
        chunk_node_id_in_graph = next(id for id in created_node_ids if data_to_store["chunk_id"] in id) # Get the actual chunk node ID

        for entity_id_part in expected_entity_ids:
             entity_node_id_in_graph = next(id for id in created_node_ids if entity_id_part in id)
             self.assertTrue(
                 any(
                     call[1]['source_node_id'] == chunk_node_id_in_graph and
                     call[1]['target_node_id'] == entity_node_id_in_graph and
                     call[1]['relationship_type'] == "MENTIONS_ENTITY"
                     for call in create_edge_calls
                 ),
                 f"Edge from chunk to entity {entity_id_part} not created."
             )

        # 3. LTM storage was called by MemoryKernel after GraphIndexer
        self.mock_ltm_instance.generate_embedding.assert_called_once_with(text_content)
        self.mock_ltm_instance.store_memory_chunk.assert_called_once()

        await self.kernel.shutdown()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
