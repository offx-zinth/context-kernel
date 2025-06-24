import unittest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import asyncio
import json # For comparing properties

from contextkernel.memory_system.graph_db import GraphDB
from contextkernel.utils.config import Neo4jConfig

class TestGraphDBMethods(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_config = Neo4jConfig(uri="bolt://mockdb:7687", user="neo4j", password="password", database="testdb")

        self.mock_driver = AsyncMock()
        self.mock_session = AsyncMock()
        self.mock_transaction = AsyncMock() # For execute_write/read logic
        self.mock_cursor = AsyncMock() # For results of tx.run

        # Setup driver and session mocks
        self.mock_driver.session.return_value = self.mock_session
        self.mock_session.__aenter__.return_value = self.mock_session # For async with
        self.mock_session.__aexit__.return_value = None

        # Setup execute_write and execute_read to use our mock transaction logic
        async def mock_tx_logic_wrapper(tx_logic, query, params):
            # Simulate the transaction logic being passed and called
            return await tx_logic(self.mock_transaction, query, params)

        self.mock_session.execute_write = mock_tx_logic_wrapper
        self.mock_session.execute_read = mock_tx_logic_wrapper

        self.mock_transaction.run = AsyncMock(return_value=self.mock_cursor)

        self.graph_db = GraphDB(config=self.mock_config, driver=self.mock_driver)

        # Patch _execute_query_neo4j to make assertions on query and params easier for some tests,
        # or to control its direct output for others (like search).
        # For methods like create_node, create_edge, we want to inspect the query.
        # For search, we might want to mock its return value.
        self.patcher_execute_query = patch.object(self.graph_db, '_execute_query_neo4j', new_callable=AsyncMock)
        self.mock_execute_query = self.patcher_execute_query.start()

    def tearDown(self):
        self.patcher_execute_query.stop()

    async def test_ensure_source_document_node(self):
        doc_id = "doc123"
        props = {"preview": "Test document...", "original_data_type": "text"}

        self.mock_execute_query.return_value = [{"created_node_id": doc_id}] # Simulate successful MERGE

        await self.graph_db.ensure_source_document_node(doc_id, props.copy()) # Pass copy

        self.mock_execute_query.assert_called_once()
        args, kwargs = self.mock_execute_query.call_args
        query = args[0]
        params = args[1]

        self.assertIn(f"MERGE (n:`SourceDocument` {{node_id: $node_id}})", query)
        self.assertIn("ON CREATE SET n = $props", query)
        self.assertEqual(params["node_id"], doc_id)
        self.assertEqual(params["props"]["preview"], props["preview"])
        self.assertIn("updated_at", params["props"]) # Added by the method

    async def test_add_memory_fragment_link(self):
        doc_id = "doc123"
        fragment_id = "frag789"
        main_label = "STMEntry"
        rel_type = "HAS_STM_REPRESENTATION"
        fragment_props = {"text": "Summary text", "type": "summary"}

        # Mock return for create_node (called twice by add_memory_fragment_link indirectly via self.create_node)
        # and create_edge
        self.mock_execute_query.side_effect = [
            [{"created_node_id": fragment_id}],  # For fragment node creation
            [{"relationship_type": rel_type}]    # For edge creation
        ]

        await self.graph_db.add_memory_fragment_link(doc_id, fragment_id, main_label, rel_type, fragment_props.copy())

        self.assertEqual(self.mock_execute_query.call_count, 2)

        # Call 1: Create Fragment Node
        call_frag_node_args, _ = self.mock_execute_query.call_args_list[0]
        query_frag = call_frag_node_args[0]
        params_frag = call_frag_node_args[1]
        self.assertIn(f"MERGE (n:`{main_label}`:`MemoryFragment` {{node_id: $node_id}})", query_frag)
        self.assertEqual(params_frag["node_id"], fragment_id)
        self.assertEqual(params_frag["props"]["text"], fragment_props["text"])
        self.assertIn("updated_at", params_frag["props"])

        # Call 2: Create Edge
        call_edge_args, _ = self.mock_execute_query.call_args_list[1]
        query_edge = call_edge_args[0]
        params_edge = call_edge_args[1]
        self.assertIn(f"MATCH (a {{node_id: $source_node_id}}), (b {{node_id: $target_node_id}})", query_edge)
        self.assertIn(f"MERGE (a)-[r:`{rel_type}`]->(b)", query_edge)
        self.assertEqual(params_edge["source_node_id"], doc_id)
        self.assertEqual(params_edge["target_node_id"], fragment_id)
        self.assertIn("created_at", params_edge["properties"])


    async def test_add_entities_to_document(self):
        doc_id = "doc456"
        entities = [
            {"text": "Alice", "type": "Person", "metadata": {"source": "ner"}},
            {"text": "Google", "type": "ORG"} # No metadata
        ]

        # Mock get_node for SourceDocument check
        # self.graph_db.get_node = AsyncMock(return_value={"node_id": doc_id}) # No, get_node uses _execute_query too.
        # So, first call to _execute_query is from get_node. Then 2 for each entity (node, edge).

        # Let's assume get_node is successful (or we don't strictly check its call if it's too complex for this test focus)
        # We focus on the calls made for entity creation and linking.
        # Expecting 4 calls to _execute_query: 2 entities * (1 for node + 1 for edge)
        # Plus one for the initial get_node call
        expected_entity1_id = "entity_person_alice"
        expected_entity2_id = "entity_org_google"

        self.mock_execute_query.side_effect = [
            [{"n": {"node_id": doc_id}}], # For the self.get_node(document_id) call
            [{"created_node_id": expected_entity1_id}],  # Alice node
            [{"relationship_type": "CONTAINS_ENTITY"}], # Alice edge
            [{"created_node_id": expected_entity2_id}], # Google node
            [{"relationship_type": "CONTAINS_ENTITY"}]  # Google edge
        ]

        await self.graph_db.add_entities_to_document(doc_id, entities)

        self.assertEqual(self.mock_execute_query.call_count, 1 + len(entities) * 2) # 1 for get_node, 2 per entity

        # Check Alice entity node creation
        call_alice_node_args, _ = self.mock_execute_query.call_args_list[1]
        self.assertIn(f"MERGE (n:`Entity`:`Person` {{node_id: $node_id}})", call_alice_node_args[0])
        self.assertEqual(call_alice_node_args[1]["node_id"], expected_entity1_id)
        self.assertEqual(call_alice_node_args[1]["props"]["text"], "Alice")
        self.assertEqual(call_alice_node_args[1]["props"]["source"], "ner")

        # Check Alice link to document
        call_alice_edge_args, _ = self.mock_execute_query.call_args_list[2]
        self.assertIn(f"MERGE (a)-[r:`CONTAINS_ENTITY`]->(b)", call_alice_edge_args[0])
        self.assertEqual(call_alice_edge_args[1]["source_node_id"], doc_id)
        self.assertEqual(call_alice_edge_args[1]["target_node_id"], expected_entity1_id)


    async def test_add_relations_to_document(self):
        doc_id = "doc789"
        relations = [
            {"subject": "Alice", "verb": "WORKS_AT", "object": "Google", "context": "She is a dev."}
        ]

        # Expected node IDs for entities involved in relation
        subj_id = "entity_unknownentity_alice" # Default type guess
        obj_id = "entity_unknownentity_google"

        # Mock calls:
        # 1. get_node for doc_id (not explicitly tested here, assume it works or covered by add_entities)
        # 2. create_node for subject
        # 3. create_node for object
        # 4. create_edge for the relation
        self.mock_execute_query.side_effect = [
            [{"created_node_id": subj_id}],  # Subject node
            [{"created_node_id": obj_id}],   # Object node
            [{"relationship_type": "WORKS_AT"}] # Relation edge
        ]

        await self.graph_db.add_relations_to_document(doc_id, relations)
        self.assertEqual(self.mock_execute_query.call_count, 3)

        # Check relation edge creation
        call_rel_edge_args, _ = self.mock_execute_query.call_args_list[2]
        self.assertIn(f"MERGE (a)-[r:`WORKS_AT`]->(b)", call_rel_edge_args[0])
        self.assertEqual(call_rel_edge_args[1]["source_node_id"], subj_id)
        self.assertEqual(call_rel_edge_args[1]["target_node_id"], obj_id)
        self.assertEqual(call_rel_edge_args[1]["properties"]["context"], "She is a dev.")


    async def test_search_transforms_results_correctly(self):
        query_text = "find this"
        top_k = 2

        # Mock raw Neo4j response for the UNION query
        mock_neo4j_response = [
            { # Simulating an Entity match
                "id": "entity_person_alice", "content": "Alice is a person",
                "source_description": "Entity: Person", "document_id": "doc1",
                "document_preview": "Doc about Alice...", "summary_text": "Alice summary.",
                "ltm_text": None, "matched_node_properties": {"text": "Alice is a person", "type": "Person", "node_id": "entity_person_alice"},
                "score": 1.0
            },
            { # Simulating an STMEntry match
                "id": "doc1_summary", "content": "Summary of Alice's document",
                "source_description": "STMEntry", "document_id": "doc1",
                "document_preview": "Doc about Alice...", "summary_text": None, # summary_text is null when STMEntry is the source
                "ltm_text": None, "matched_node_properties": {"text": "Summary of Alice's document", "type": "summary", "node_id": "doc1_summary"},
                "score": 0.9
            }
        ]
        self.mock_execute_query.return_value = mock_neo4j_response

        results = await self.graph_db.search(query_text, top_k)

        self.mock_execute_query.assert_called_once()
        args, _ = self.mock_execute_query.call_args
        self.assertIn("UNION ALL", args[0]) # Check it's the unified query
        self.assertEqual(args[1]["query_text"], query_text)
        self.assertEqual(args[1]["limit"], top_k)

        self.assertEqual(len(results), 2)

        # Check first item (Entity match)
        item1 = results[0]
        self.assertEqual(item1["id"], "entity_person_alice")
        self.assertEqual(item1["content"], "Alice is a person")
        self.assertEqual(item1["source"], "graph_db_entity_person") # Transformed source
        self.assertEqual(item1["score"], 1.0)
        self.assertEqual(item1["metadata"]["document_id"], "doc1")
        self.assertEqual(item1["metadata"]["summary_text"], "Alice summary.")
        self.assertEqual(item1["metadata"]["text"], "Alice is a person") # from matched_node_properties

        # Check second item (STMEntry match)
        item2 = results[1]
        self.assertEqual(item2["id"], "doc1_summary")
        self.assertEqual(item2["content"], "Summary of Alice's document")
        self.assertEqual(item2["source"], "graph_db_stmentry")
        self.assertEqual(item2["score"], 0.9)
        self.assertIsNone(item2["metadata"].get("summary_text")) # Should not be there if STM is source
        self.assertEqual(item2["metadata"]["type"], "summary")


if __name__ == '__main__':
    unittest.main()
