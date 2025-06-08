import unittest
from unittest.mock import patch, MagicMock, call # Added call for checking multiple calls
import uuid

# Adjust import paths as necessary
from context_kernel.memory_llm_roles import MemoryAccessorAgent, SummarizerUpdaterAgent, DEFAULT_EMBEDDING_SIZE
from context_kernel.vector_db_manager import TIER_NAMES # Import TIER_NAMES for tests
# Assuming qdrant_models are part of VectorDBManager or accessible for type hints/mocking
from qdrant_client import models as qdrant_models # For qdrant_models.UpdateStatus

# If these classes are not directly in context_kernel, adjust path.
# E.g., from context_kernel.vector_db_manager import VectorDBManager
# from context_kernel.graph_db_layer import GraphDBLayer
# from context_kernel.working_memory_system import WorkingMemorySystem


class TestMemoryAccessorAgent(unittest.TestCase):

    def setUp(self):
        self.mock_vdb_manager = MagicMock()
        self.mock_graph_db = MagicMock()
        self.mock_wms = MagicMock()

        # Mock the placeholder functions within the module where the agent will call them
        # Patching them 'within' the module where they are defined and used by the agents.
        self.patcher_generate_embedding = patch('context_kernel.memory_llm_roles._placeholder_generate_embedding')
        self.patcher_llm_for_graph_query = patch('context_kernel.memory_llm_roles._placeholder_llm_for_graph_query')

        self.mock_generate_embedding = self.patcher_generate_embedding.start()
        self.mock_llm_for_graph_query = self.patcher_llm_for_graph_query.start()
        
        # Default mock returns for placeholder functions
        self.mock_embedding_vector = [0.1] * DEFAULT_EMBEDDING_SIZE
        self.mock_generate_embedding.return_value = self.mock_embedding_vector
        self.mock_llm_for_graph_query.return_value = "MATCH (n) RETURN n.name as content, id(n) as id"


        self.agent = MemoryAccessorAgent(
            vector_db_manager=self.mock_vdb_manager,
            graph_db_layer=self.mock_graph_db,
            working_memory=self.mock_wms,
            embedding_model_name="test_embed_model"
        )

    def tearDown(self):
        self.patcher_generate_embedding.stop()
        self.patcher_llm_for_graph_query.stop()

    def test_init(self):
        self.assertEqual(self.agent.vector_db_manager, self.mock_vdb_manager)
        self.assertEqual(self.agent.graph_db_layer, self.mock_graph_db)
        self.assertEqual(self.agent.working_memory, self.mock_wms)
        self.assertEqual(self.agent.embedding_model_name, "test_embed_model")

    def test_fetch_memory_all_sources(self):
        query_text = "test query"
        
        # Setup mock responses
        self.mock_wms.get_recent_notes.return_value = [{'id': 'wms1', 'content': 'WMS note', 'origin': 'test', 'timestamp': 123}]
        
        # Mock Qdrant ScoredPoint object
        mock_qdrant_hit = MagicMock(spec=qdrant_models.ScoredPoint)
        mock_qdrant_hit.id = "vdb1"
        mock_qdrant_hit.score = 0.9
        mock_qdrant_hit.payload = {"text": "Vector DB hit"}
        self.mock_vdb_manager.query_tier.return_value = [mock_qdrant_hit]
        
        self.mock_graph_db.query_graph.return_value = [{"content": "Graph DB record", "id": "gdb1"}]

        results = self.agent.fetch_memory(
            query_text,
            top_k_vector=3,
            search_working_memory=True,
            search_vector_dbs=["RawThoughtsDB"], # Test with a specific tier
            search_graph=True
        )

        self.mock_generate_embedding.assert_called_once_with(query_text) # From the agent's _generate_embedding wrapper
        self.mock_wms.get_recent_notes.assert_called_once_with(count=10) # Default count in WMS search
        self.mock_vdb_manager.query_tier.assert_called_once_with(
            tier_name="RawThoughtsDB",
            query_vector=self.mock_embedding_vector,
            top_k=3
        )
        self.mock_llm_for_graph_query.assert_called_once_with(query_text)
        self.mock_graph_db.query_graph.assert_called_once_with(self.mock_llm_for_graph_query.return_value, parameters={})
        
        self.assertEqual(len(results), 3) # One from each source as mocked
        # Check if formatting is roughly as expected
        self.assertTrue(any(r['source'] == 'working_memory' for r in results))
        self.assertTrue(any(r['source'] == 'vector_db_RawThoughtsDB' for r in results))
        self.assertTrue(any(r['source'] == 'graph_db' for r in results))


    def test_fetch_memory_only_wms(self):
        query_text = "wms only query"
        self.mock_wms.get_recent_notes.return_value = [{'id': 'wms1', 'content': 'WMS note for wms only'}]
        
        results = self.agent.fetch_memory(
            query_text,
            search_working_memory=True,
            search_vector_dbs=[], # Disable VDB
            search_graph=False     # Disable Graph
        )
        self.mock_generate_embedding.assert_called_once_with(query_text)
        self.mock_wms.get_recent_notes.assert_called_once()
        self.mock_vdb_manager.query_tier.assert_not_called()
        self.mock_graph_db.query_graph.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['source'], 'working_memory')

    def test_fetch_memory_no_sources_enabled(self):
        query_text = "no sources query"
        results = self.agent.fetch_memory(
            query_text,
            search_working_memory=False,
            search_vector_dbs=[], 
            search_graph=False
        )
        self.mock_generate_embedding.assert_called_once_with(query_text)
        self.mock_wms.get_recent_notes.assert_not_called()
        self.mock_vdb_manager.query_tier.assert_not_called()
        self.mock_graph_db.query_graph.assert_not_called()
        self.assertEqual(len(results), 0)

    def test_fetch_memory_all_vector_dbs_if_none_specified(self):
        query_text = "all vdb query"
        # Mock TIER_NAMES if it's not directly available or to control it for test
        with patch('context_kernel.memory_llm_roles.TIER_NAMES', ["TestTier1", "TestTier2"]):
            self.agent.fetch_memory(query_text, search_vector_dbs=None, search_working_memory=False, search_graph=False)
            
            self.assertEqual(self.mock_vdb_manager.query_tier.call_count, 2)
            self.mock_vdb_manager.query_tier.assert_any_call(tier_name="TestTier1", query_vector=self.mock_embedding_vector, top_k=5) # default top_k
            self.mock_vdb_manager.query_tier.assert_any_call(tier_name="TestTier2", query_vector=self.mock_embedding_vector, top_k=5)


    def test_anticipate_memory_needs(self):
        """Test the placeholder anticipate_memory_needs method."""
        activity_vector = [0.5] * DEFAULT_EMBEDDING_SIZE
        # The method prints and returns a mock response. We just check it's callable.
        with patch('builtins.print') as mock_print:
            result = self.agent.anticipate_memory_needs(activity_vector)
        
        self.assertIsInstance(result, list)
        self.assertTrue(any("anticipated_ltm" in item.get("source", "") for item in result))
        mock_print.assert_any_call(unittest.mock.ANY) # Check that it printed something


class TestSummarizerUpdaterAgent(unittest.TestCase):

    def setUp(self):
        self.mock_vdb_manager = MagicMock()
        self.mock_graph_db = MagicMock()

        self.patcher_generate_embedding = patch('context_kernel.memory_llm_roles._placeholder_generate_embedding')
        self.patcher_call_llm_summary = patch('context_kernel.memory_llm_roles._placeholder_call_llm_for_summary')
        
        self.mock_generate_embedding = self.patcher_generate_embedding.start()
        self.mock_call_llm_summary = self.patcher_call_llm_summary.start()

        self.mock_embedding_vector = [0.2] * DEFAULT_EMBEDDING_SIZE
        self.mock_generate_embedding.return_value = self.mock_embedding_vector
        self.mock_summary_text = "Mocked summary."
        self.mock_call_llm_summary.return_value = self.mock_summary_text
        
        # Mock the UpdateResult status for VDB store_embedding
        mock_update_result = MagicMock(spec=qdrant_models.UpdateResult)
        mock_update_result.status = qdrant_models.UpdateStatus.COMPLETED
        self.mock_vdb_manager.store_embedding.return_value = mock_update_result

        self.mock_graph_db.add_node.return_value = 123 # Mock node ID
        self.mock_graph_db.add_relationship.return_value = "RELATED_TO" # Mock rel type string


        self.agent = SummarizerUpdaterAgent(
            vector_db_manager=self.mock_vdb_manager,
            graph_db_layer=self.mock_graph_db,
            embedding_model_name="test_summarizer_embed_model"
        )

    def tearDown(self):
        self.patcher_generate_embedding.stop()
        self.patcher_call_llm_summary.stop()

    def test_init(self):
        self.assertEqual(self.agent.vector_db_manager, self.mock_vdb_manager)
        self.assertEqual(self.agent.graph_db_layer, self.mock_graph_db)
        self.assertEqual(self.agent.embedding_model_name, "test_summarizer_embed_model")

    def test_process_input_all_enabled(self):
        input_text = "This is a test input for summarization and storage."
        source_module = "test_source"
        existing_node_id = 789

        results = self.agent.process_input(
            input_text, source_module,
            existing_node_id=existing_node_id,
            store_raw=True, create_summary=True, store_in_graph=True
        )

        # Check embedding calls
        self.mock_generate_embedding.assert_any_call(input_text, model_name="test_summarizer_embed_model")
        self.mock_generate_embedding.assert_any_call(self.mock_summary_text, model_name="test_summarizer_embed_model")
        self.assertEqual(self.mock_generate_embedding.call_count, 2)
        
        # Check summary call
        self.mock_call_llm_summary.assert_called_once_with(input_text, max_length=150, context=f"Source: {source_module}")

        # Check VDB storage
        self.mock_vdb_manager.store_embedding.assert_any_call(
            tier_name="RawThoughtsDB", points=unittest.mock.ANY
        )
        self.mock_vdb_manager.store_embedding.assert_any_call(
            tier_name="ChunkSummaryDB", points=unittest.mock.ANY
        )
        self.assertEqual(self.mock_vdb_manager.store_embedding.call_count, 2)
        
        # Verify payload of VDB calls (checking one for example)
        raw_store_call = self.mock_vdb_manager.store_embedding.call_args_list[0] # Assuming RawThoughtsDB is first
        self.assertEqual(raw_store_call[1]['tier_name'], "RawThoughtsDB")
        point_raw = raw_store_call[1]['points'][0]
        self.assertEqual(point_raw.payload['text'], input_text)
        self.assertEqual(point_raw.payload['source'], source_module)
        self.assertEqual(point_raw.vector, self.mock_embedding_vector)


        # Check GraphDB storage
        self.mock_graph_db.add_node.assert_called_once()
        # Properties will include text_snippet, source, timestamp, summary
        self.assertEqual(self.mock_graph_db.add_node.call_args[1]['properties']['source'], source_module)
        self.assertIn(input_text[:50], self.mock_graph_db.add_node.call_args[1]['properties']['text_snippet'])
        self.assertEqual(self.mock_graph_db.add_node.call_args[1]['properties']['summary'], self.mock_summary_text)


        self.mock_graph_db.add_relationship.assert_called_once_with(
            start_node_id=123, # from add_node mock
            end_node_id=existing_node_id,
            relationship_type=unittest.mock.ANY, # Type could be DERIVED_FROM or RELATES_TO
            properties=unittest.mock.ANY
        )

        # Check results structure
        self.assertIsNotNone(results.get("raw_stored_ids"))
        self.assertTrue(len(results["raw_stored_ids"]) == 1)
        self.assertIsNotNone(results.get("summary_stored_ids"))
        self.assertTrue(len(results["summary_stored_ids"]) == 1)
        self.assertEqual(results.get("graph_node_id"), 123)
        self.assertTrue(len(results.get("graph_relationship_ids")) == 1)


    def test_process_input_raw_only_no_graph_link(self):
        input_text = "Raw only input."
        source_module = "raw_test_source"

        results = self.agent.process_input(
            input_text, source_module,
            existing_node_id=None, # No link
            store_raw=True, create_summary=False, store_in_graph=False # Disable summary and graph
        )

        self.mock_generate_embedding.assert_called_once_with(input_text, model_name="test_summarizer_embed_model")
        self.mock_call_llm_summary.assert_not_called()
        
        self.mock_vdb_manager.store_embedding.assert_called_once_with(
            tier_name="RawThoughtsDB", points=unittest.mock.ANY
        )
        
        self.mock_graph_db.add_node.assert_not_called()
        self.mock_graph_db.add_relationship.assert_not_called()

        self.assertTrue(len(results["raw_stored_ids"]) == 1)
        self.assertTrue(len(results["summary_stored_ids"]) == 0)
        self.assertIsNone(results["graph_node_id"])
        self.assertTrue(len(results["graph_relationship_ids"]) == 0)


    def test_maintain_executive_summary(self):
        """Test the placeholder maintain_executive_summary method."""
        chunk_summaries = ["summary 1", "summary 2"]
        
        # Mock the store_embedding specifically for ExecutiveSummaryDB
        mock_exec_update_result = MagicMock(spec=qdrant_models.UpdateResult)
        mock_exec_update_result.status = qdrant_models.UpdateStatus.COMPLETED
        
        # Ensure store_embedding returns the specific mock when called for ExecutiveSummaryDB
        def store_embedding_side_effect(tier_name, points):
            if tier_name == "ExecutiveSummaryDB":
                return mock_exec_update_result
            return MagicMock() # Default for other tiers if any were called

        self.mock_vdb_manager.store_embedding.side_effect = store_embedding_side_effect

        with patch('builtins.print') as mock_print:
            self.agent.maintain_executive_summary(chunk_summaries)

        self.mock_call_llm_summary.assert_called_once_with(
            "\n".join(chunk_summaries), max_length=500, context="Create a high-level executive summary."
        )
        self.mock_generate_embedding.assert_called_once_with(self.mock_summary_text, model_name="test_summarizer_embed_model")
        
        # Check that store_embedding was called for ExecutiveSummaryDB
        found_exec_db_call = False
        for vdb_call in self.mock_vdb_manager.store_embedding.call_args_list:
            if vdb_call[1]['tier_name'] == "ExecutiveSummaryDB":
                found_exec_db_call = True
                point_exec = vdb_call[1]['points'][0]
                self.assertEqual(point_exec.payload['summary_text'], self.mock_summary_text)
                self.assertEqual(point_exec.payload['source_chunks'], len(chunk_summaries))
                self.assertEqual(point_exec.vector, self.mock_embedding_vector)
                break
        self.assertTrue(found_exec_db_call, "store_embedding was not called for ExecutiveSummaryDB")
        
        mock_print.assert_any_call(unittest.mock.ANY) # Check it printed something


if __name__ == '__main__':
    unittest.main()
