import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call

from contextkernel.memory_system.memory_manager import MemoryManager
from contextkernel.core_logic.llm_listener import StructuredInsight, Summary, Entity, Relation # Assuming these are the types
import datetime

# Helper to run async tests
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        self.mock_graph_db = AsyncMock()
        self.mock_ltm = AsyncMock()
        self.mock_stm = AsyncMock()
        self.mock_raw_cache = AsyncMock()

        self.memory_manager = MemoryManager(
            graph_db=self.mock_graph_db,
            ltm=self.mock_ltm,
            stm=self.mock_stm,
            raw_cache=self.mock_raw_cache
        )

        # Sample StructuredInsight data for testing
        self.utc_now = datetime.datetime.now(datetime.timezone.utc)
        self.sample_summary = Summary(text="This is a summary.", created_at=self.utc_now, updated_at=self.utc_now)
        self.sample_entities = [Entity(text="Entity1", type="PERSON", created_at=self.utc_now, updated_at=self.utc_now)]
        self.sample_relations = [Relation(subject="Entity1", verb="IS_A", object="PERSON", created_at=self.utc_now, updated_at=self.utc_now)]
        self.sample_embedding = [0.1, 0.2, 0.3]

        self.sample_insight = StructuredInsight(
            original_data_type="text",
            source_data_preview="Original data preview...",
            summary=self.sample_summary,
            entities=self.sample_entities,
            relations=self.sample_relations,
            raw_data_id="raw_doc_123",
            content_embedding=self.sample_embedding,
            created_at=self.utc_now, # Main insight timestamp
            updated_at=self.utc_now
        )
        self.doc_id_base = self.sample_insight.raw_data_id

    @async_test
    async def test_store_successful(self):
        self.mock_stm.save_summary.return_value = None
        self.mock_ltm.save_document.return_value = None
        self.mock_graph_db.ensure_source_document_node.return_value = True
        self.mock_graph_db.add_memory_fragment_link.return_value = True
        self.mock_graph_db.add_entities_to_document.return_value = True
        self.mock_graph_db.add_relations_to_document.return_value = True

        await self.memory_manager.store(self.sample_insight)

        self.mock_stm.save_summary.assert_called_once()
        args_stm, kwargs_stm = self.mock_stm.save_summary.call_args
        self.assertEqual(kwargs_stm['summary_id'], f"{self.doc_id_base}_summary")
        self.assertEqual(kwargs_stm['summary_obj'], self.sample_summary.text)

        self.mock_ltm.save_document.assert_called_once()
        args_ltm, kwargs_ltm = self.mock_ltm.save_document.call_args
        self.assertEqual(kwargs_ltm['doc_id'], f"{self.doc_id_base}_ltm_doc")
        self.assertEqual(kwargs_ltm['text_content'], self.sample_summary.text)
        self.assertEqual(kwargs_ltm['embedding'], self.sample_embedding)

        self.mock_graph_db.ensure_source_document_node.assert_called_once_with(
            document_id=self.doc_id_base,
            properties={
                "raw_data_id": self.sample_insight.raw_data_id,
                "preview": self.sample_insight.source_data_preview,
                "original_data_type": self.sample_insight.original_data_type,
                "created_at": self.sample_insight.created_at.isoformat(),
                "updated_at": self.sample_insight.updated_at.isoformat()
            }
        )

        expected_stm_frag_props = {
            "id": f"{self.doc_id_base}_summary",
            "text": self.sample_summary.text,
            "source_document_id": self.doc_id_base,
            "type": "summary",
            "created_at": self.sample_summary.created_at.isoformat()
        }
        expected_ltm_frag_props = {
            "id": f"{self.doc_id_base}_ltm_doc",
            "text_preview": self.sample_summary.text[:255],
            "has_embedding": bool(self.sample_embedding),
            "source_document_id": self.doc_id_base,
            "type": "ltm_document_content",
            "created_at": self.sample_insight.created_at.isoformat()
        }
        expected_raw_frag_props = {
            "id": self.sample_insight.raw_data_id,
            "type": "raw_data_log",
            "source_document_id": self.doc_id_base,
            "created_at": self.sample_insight.created_at.isoformat()
        }

        calls = [
            call(document_id=self.doc_id_base, fragment_id=f"{self.doc_id_base}_summary", fragment_main_label="STMEntry", relationship_type="HAS_STM_REPRESENTATION", fragment_properties=expected_stm_frag_props),
            call(document_id=self.doc_id_base, fragment_id=f"{self.doc_id_base}_ltm_doc", fragment_main_label="LTMLogEntry", relationship_type="HAS_LTM_REPRESENTATION", fragment_properties=expected_ltm_frag_props),
            call(document_id=self.doc_id_base, fragment_id=self.sample_insight.raw_data_id, fragment_main_label="RawCacheEntry", relationship_type="REFERENCES_RAW_CACHE", fragment_properties=expected_raw_frag_props)
        ]
        self.mock_graph_db.add_memory_fragment_link.assert_has_calls(calls, any_order=True)
        self.assertEqual(self.mock_graph_db.add_memory_fragment_link.call_count, 3)


        self.mock_graph_db.add_entities_to_document.assert_called_once()
        args_entities_call, kwargs_entities_call = self.mock_graph_db.add_entities_to_document.call_args
        self.assertEqual(kwargs_entities_call['document_id'], self.doc_id_base)
        self.assertEqual(len(kwargs_entities_call['entities']), len(self.sample_entities))
        self.assertEqual(kwargs_entities_call['entities'][0]['text'], self.sample_entities[0].text)


        self.mock_graph_db.add_relations_to_document.assert_called_once()
        args_relations_call, kwargs_relations_call = self.mock_graph_db.add_relations_to_document.call_args
        self.assertEqual(kwargs_relations_call['document_id'], self.doc_id_base)
        self.assertEqual(len(kwargs_relations_call['relations']), len(self.sample_relations))
        self.assertEqual(kwargs_relations_call['relations'][0]['subject'], self.sample_relations[0].subject)

    @async_test
    async def test_store_no_summary(self):
        insight_no_summary = self.sample_insight.model_copy(deep=True)
        insight_no_summary.summary = None
        expected_ltm_content = insight_no_summary.source_data_preview

        await self.memory_manager.store(insight_no_summary)

        self.mock_stm.save_summary.assert_not_called()

        self.mock_ltm.save_document.assert_called_once()
        _, kwargs_ltm = self.mock_ltm.save_document.call_args
        self.assertEqual(kwargs_ltm['text_content'], expected_ltm_content)

        self.assertEqual(self.mock_graph_db.add_memory_fragment_link.call_count, 2) # LTM, RawCache
        for call_args_tuple in self.mock_graph_db.add_memory_fragment_link.call_args_list:
            _, c_kwargs = call_args_tuple
            self.assertNotEqual(c_kwargs['fragment_main_label'], "STMEntry")

    @async_test
    async def test_store_graph_failure_on_ensure_node(self):
        self.mock_graph_db.ensure_source_document_node.side_effect = Exception("GraphDB connection failed")

        with self.assertRaisesRegex(Exception, "GraphDB initial node setup failed"): # MemoryAccessError
            await self.memory_manager.store(self.sample_insight)

        self.mock_stm.save_summary.assert_not_called()

    @async_test
    async def test_store_one_memory_op_fails(self):
        self.mock_graph_db.ensure_source_document_node.return_value = True
        self.mock_graph_db.add_memory_fragment_link.return_value = True
        self.mock_stm.save_summary.return_value = None
        self.mock_ltm.save_document.return_value = None
        self.mock_graph_db.add_entities_to_document.side_effect = Exception("Failed to add entities")

        with self.assertRaisesRegex(Exception, "At least one memory operation failed"): # MemoryAccessError
            await self.memory_manager.store(self.sample_insight)

        self.mock_stm.save_summary.assert_called_once()
        self.mock_ltm.save_document.assert_called_once()
        self.mock_graph_db.ensure_source_document_node.assert_called_once()
        self.assertTrue(self.mock_graph_db.add_memory_fragment_link.call_count >= 1)


    @async_test
    async def test_update_placeholder(self):
        with patch('logging.Logger.warning') as mock_log_warning:
            result = await self.memory_manager.update("doc_id_test_update", self.sample_insight)
            self.assertTrue(result)
            mock_log_warning.assert_any_call("MemoryManager.update called for document_id: doc_id_test_update. Not fully implemented.")
            self.mock_graph_db.ensure_source_document_node.assert_called_once()

    @async_test
    async def test_edit_placeholder(self):
        with patch('logging.Logger.warning') as mock_log_warning:
            result = await self.memory_manager.edit("doc_id_test_edit", {"field": "new_value"})
            self.assertFalse(result)
            mock_log_warning.assert_called_once_with(
                "MemoryManager.edit called for document_id: doc_id_test_edit with edits: {'field': 'new_value'}. Not fully implemented."
            )

    @async_test
    async def test_delete_placeholder(self):
        self.mock_graph_db.delete_node.return_value = True
        with patch('logging.Logger.warning') as mock_log_warning, \
             patch('logging.Logger.info') as mock_log_info:
            # Patch the hasattr checks for stm, ltm, raw_cache to simulate they don't have delete methods
            with patch.object(self.memory_manager.stm, 'delete_summary', create=False, side_effect=AttributeError), \
                 patch.object(self.memory_manager.ltm, 'delete_document', create=False, side_effect=AttributeError), \
                 patch.object(self.memory_manager.raw_cache, 'delete', create=False, side_effect=AttributeError):

                result = await self.memory_manager.delete("doc_id_test_delete")
                self.assertTrue(result)

            mock_log_warning.assert_any_call("MemoryManager.delete called for document_id: doc_id_test_delete. Not fully implemented.")
            self.mock_graph_db.delete_node.assert_called_once_with("doc_id_test_delete")
            mock_log_info.assert_any_call("GraphDB SourceDocument node doc_id_test_delete deleted (associated fragments/entities may need separate cleanup if not handled by DETACH DELETE or cascading logic).")
            # Check that warnings for missing delete methods on sub-components are logged
            mock_log_warning.assert_any_call("STM delete for doc_id_test_delete_summary not implemented on STM component yet.")


    @async_test
    async def test_delete_placeholder_graph_fail(self):
        self.mock_graph_db.delete_node.return_value = False
        with patch('logging.Logger.warning') as mock_log_warning:
            # Patch hasattr checks for stm, ltm, raw_cache
            with patch.object(self.memory_manager.stm, 'delete_summary', create=False, side_effect=AttributeError), \
                 patch.object(self.memory_manager.ltm, 'delete_document', create=False, side_effect=AttributeError), \
                 patch.object(self.memory_manager.raw_cache, 'delete', create=False, side_effect=AttributeError):
                result = await self.memory_manager.delete("doc_id_test_delete_fail")
                self.assertFalse(result)
            mock_log_warning.assert_any_call("MemoryManager.delete called for document_id: doc_id_test_delete_fail. Not fully implemented.")
            mock_log_warning.assert_any_call("GraphDB SourceDocument node doc_id_test_delete_fail not found or delete failed.")


if __name__ == '__main__':
    unittest.main()
