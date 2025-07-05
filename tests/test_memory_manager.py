import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Imports from the actual project
# Adjust paths as necessary if these are located elsewhere
from contextkernel.memory_system.memory_manager import MemoryManager, MemoryAccessError
from contextkernel.core_logic.llm_listener import StructuredInsight, Summary, Entity, Relation # Assuming these types

# Mock Interfaces for DBs
class MockGraphDBInterface:
    ensure_source_document_node = AsyncMock()
    add_memory_fragment_link = AsyncMock()
    add_entities_to_document = AsyncMock()
    add_relations_to_document = AsyncMock()
    update_node = AsyncMock()
    delete_node = AsyncMock()
    # Add other methods if MemoryManager calls them directly

class MockLTMInterface:
    save_document = AsyncMock()
    update_document = AsyncMock() # Or update_embedding if that's the chosen name
    delete_embedding = AsyncMock()
    # Add other methods if MemoryManager calls them directly

class MockSTMInterface:
    save_summary = AsyncMock()
    update_summary = AsyncMock()
    delete_summary = AsyncMock()
    # Add other methods if MemoryManager calls them directly

class MockRawCacheInterface:
    # RawCache is not directly managed by MemoryManager's store/edit/forget
    # in terms of writing/deleting its primary content, but it's a dependency.
    # Its methods might be called if, e.g., MemoryManager had to fetch raw data.
    # For the current scope of store/edit/forget, direct calls seem minimal beyond graph linking.
    delete_raw_data = AsyncMock() # For the forget method
    pass


class TestMemoryManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_graph_db = MockGraphDBInterface()
        self.mock_ltm = MockLTMInterface()
        self.mock_stm = MockSTMInterface()
        self.mock_raw_cache = MockRawCacheInterface()

        self.memory_manager = MemoryManager(
            graph_db=self.mock_graph_db,
            ltm=self.mock_ltm,
            stm=self.mock_stm,
            raw_cache=self.mock_raw_cache
        )

        # Reset mocks before each test
        self.mock_graph_db.reset_mock()
        self.mock_ltm.reset_mock()
        self.mock_stm.reset_mock()
        self.mock_raw_cache.reset_mock()

    async def test_store_successful(self):
        # Create a dummy StructuredInsight object
        summary_obj = Summary(text="Test summary")
        entity_obj = Entity(text="Test Entity", type="PERSON")
        relation_obj = Relation(subject="Test Entity", verb="is", object="testing")

        structured_data = StructuredInsight(
            raw_data_id="raw_doc_123",
            source_data_preview="This is a test.",
            original_data_type="text",
            summary=summary_obj,
            entities=[entity_obj],
            relations=[relation_obj],
            content_embedding=[0.1, 0.2, 0.3]
        )

        await self.memory_manager.store(structured_data)

        # Assert that the correct DB methods were called
        self.mock_stm.save_summary.assert_called_once()
        self.mock_ltm.save_document.assert_called_once()
        self.mock_graph_db.ensure_source_document_node.assert_called_once()
        self.mock_graph_db.add_memory_fragment_link.assert_any_call(
            document_id="raw_doc_123",
            fragment_id="raw_doc_123_summary", # Based on MemoryManager's current ID generation
            fragment_main_label="STMEntry",
            relationship_type="HAS_STM_REPRESENTATION",
            fragment_properties=unittest.mock.ANY
        )
        self.mock_graph_db.add_memory_fragment_link.assert_any_call(
            document_id="raw_doc_123",
            fragment_id="raw_doc_123_ltm_doc", # Based on MemoryManager's current ID generation
            fragment_main_label="LTMLogEntry",
            relationship_type="HAS_LTM_REPRESENTATION",
            fragment_properties=unittest.mock.ANY
        )
        # Check if raw_cache linking was called (it should if raw_data_id is present)
        self.mock_graph_db.add_memory_fragment_link.assert_any_call(
            document_id="raw_doc_123",
            fragment_id="raw_doc_123",
            fragment_main_label="RawCacheEntry",
            relationship_type="REFERENCES_RAW_CACHE",
            fragment_properties=unittest.mock.ANY
        )
        self.mock_graph_db.add_entities_to_document.assert_called_once()
        self.mock_graph_db.add_relations_to_document.assert_called_once()

    async def test_store_graph_db_initial_failure(self):
        self.mock_graph_db.ensure_source_document_node.side_effect = MemoryAccessError("Graph init failed")
        structured_data = StructuredInsight(raw_data_id="fail_doc_001", source_data_preview="Test")

        with self.assertRaises(MemoryAccessError) as context:
            await self.memory_manager.store(structured_data)

        self.assertIn("GraphDB initial node setup failed", str(context.exception))
        self.mock_stm.save_summary.assert_not_called() # Should not proceed if graph init fails early

    async def test_store_one_sub_operation_fails(self):
        # Example: STM save_summary fails
        self.mock_stm.save_summary.side_effect = MemoryAccessError("STM save failed")
        structured_data = StructuredInsight(
            raw_data_id="doc_stm_fail",
            source_data_preview="Test",
            summary=Summary(text="A summary") # Ensure summary is present to trigger STM call
        )

        with self.assertRaises(MemoryAccessError) as context:
            await self.memory_manager.store(structured_data)

        self.assertIn("At least one memory operation failed: STM save failed", str(context.exception))
        # ensure_source_document_node would have been called before the gather
        self.mock_graph_db.ensure_source_document_node.assert_called_once()


    async def test_edit_stm_summary(self):
        memory_id = "doc123_summary"
        updates = {"summary_content": "Updated summary text."}

        # Mock STM's update_summary to simulate success
        self.mock_stm.update_summary = AsyncMock(return_value=None)

        result = await self.memory_manager.edit(memory_id, updates)

        self.assertTrue(result)
        self.mock_stm.update_summary.assert_called_once_with(summary_id=memory_id, updates=updates)
        self.mock_ltm.update_document.assert_not_called()
        self.mock_graph_db.update_node.assert_not_called()

    async def test_edit_ltm_document(self):
        memory_id = "doc123_ltm_doc"
        updates = {"text_content": "Updated LTM content.", "metadata": {"tag": "updated"}}

        self.mock_ltm.update_document = AsyncMock(return_value=None) # Assuming this is the method name

        result = await self.memory_manager.edit(memory_id, updates)

        self.assertTrue(result)
        self.mock_ltm.update_document.assert_called_once_with(doc_id=memory_id, updates=updates)
        self.mock_stm.update_summary.assert_not_called()
        self.mock_graph_db.update_node.assert_not_called()

    async def test_edit_graph_node(self):
        memory_id = "graph_node_entity_abc" # Generic graph node ID
        updates = {"property1": "new_value", "status": "archived"}

        self.mock_graph_db.update_node = AsyncMock(return_value=None)

        result = await self.memory_manager.edit(memory_id, updates)

        self.assertTrue(result)
        self.mock_graph_db.update_node.assert_called_once_with(node_id=memory_id, data=updates)
        self.mock_stm.update_summary.assert_not_called()
        self.mock_ltm.update_document.assert_not_called()

    async def test_edit_unknown_id_format_or_system_unavailable(self):
        # Test with an ID format that doesn't match known patterns and no graph_db (or graph_db update fails)
        memory_id = "unknown_format_id_123"
        updates = {"some_update": "value"}

        # Simulate graph_db not being available or its update_node failing to match
        # If graph_db is None, edit would return False if no other system matches.
        # If graph_db.update_node is mocked to simulate "not found" or error:
        self.mock_graph_db.update_node = AsyncMock(side_effect=MemoryAccessError("Node not found or generic error"))
        # Or, if MemoryManager's logic returns False if no specific system is targeted and graph_db fails:
        # For now, let's assume it tries graph_db and if that fails, it raises MemoryAccessError.

        with self.assertRaises(MemoryAccessError): # Expecting error as graph_db is the fallback
            await self.memory_manager.edit(memory_id, updates)

        # If we want to test the "return False" path for unknown ID when graph_db is also None:
        mm_no_graph = MemoryManager(graph_db=None, ltm=self.mock_ltm, stm=self.mock_stm, raw_cache=self.mock_raw_cache)
        result_no_graph = await mm_no_graph.edit(memory_id, updates)
        self.assertFalse(result_no_graph)


    async def test_forget_successful_cascading_delete(self):
        memory_id = "doc_to_forget_123" # This ID is assumed to be the base for others

        # Mock DB delete methods to indicate success
        self.mock_graph_db.delete_node = AsyncMock(return_value=None)
        self.mock_ltm.delete_embedding = AsyncMock(return_value=None)
        self.mock_stm.delete_summary = AsyncMock(return_value=None)
        self.mock_raw_cache.delete_raw_data = AsyncMock(return_value=None)

        result = await self.memory_manager.forget(memory_id)

        self.assertTrue(result)
        self.mock_graph_db.delete_node.assert_called_once_with(node_id=memory_id)
        self.mock_ltm.delete_embedding.assert_called_once_with(memory_id=f"{memory_id}_ltm_doc")
        self.mock_stm.delete_summary.assert_called_once_with(summary_id=f"{memory_id}_summary")
        self.mock_raw_cache.delete_raw_data.assert_called_once_with(doc_id=memory_id)

    async def test_forget_one_db_fails(self):
        memory_id = "doc_partial_forget_456"

        self.mock_graph_db.delete_node = AsyncMock(return_value=None) # Graph succeeds
        self.mock_ltm.delete_embedding.side_effect = MemoryAccessError("LTM delete failed") # LTM fails
        self.mock_stm.delete_summary = AsyncMock(return_value=None) # STM succeeds
        self.mock_raw_cache.delete_raw_data = AsyncMock(return_value=None) # RawCache succeeds

        result = await self.memory_manager.forget(memory_id)

        # Current forget logic returns True if any operation succeeded ("best effort")
        # and logs errors for failures.
        self.assertTrue(result)
        self.mock_graph_db.delete_node.assert_called_once()
        self.mock_ltm.delete_embedding.assert_called_once() # Called, but raised error
        self.mock_stm.delete_summary.assert_called_once()
        self.mock_raw_cache.delete_raw_data.assert_called_once()

    async def test_forget_no_relevant_systems(self):
        # Test case where all DB interfaces are None (or their delete methods are missing)
        mm_no_dbs = MemoryManager(graph_db=None, ltm=None, stm=None, raw_cache=None)
        memory_id = "doc_no_systems_789"

        result = await mm_no_dbs.forget(memory_id)
        self.assertFalse(result) # Should be False if no operations could be performed

    async def test_edit_stm_summary_missing_method_on_interface(self):
        # Simulate STM interface not having update_summary (e.g., before it was added)
        del self.mock_stm.update_summary # Remove the mock method

        memory_id = "doc123_summary"
        updates = {"summary_content": "Updated summary text."}

        result = await self.memory_manager.edit(memory_id, updates)
        self.assertFalse(result) # Should log error and return False

if __name__ == '__main__':
    unittest.main()
