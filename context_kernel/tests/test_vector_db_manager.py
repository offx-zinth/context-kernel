import unittest
from unittest.mock import patch, MagicMock, ANY
from context_kernel.vector_db_manager import VectorDBManager, TIER_NAMES, models
from qdrant_client.http.exceptions import UnexpectedResponse # For simulating collection not found


# Mock Qdrant models.Distance if not directly accessible or for simplicity
class MockDistance:
    COSINE = "Cosine"
    EUCLID = "Euclid"

if not hasattr(models, 'Distance'): # If models.Distance is not available from qdrant_client import (e.g. older versions)
    models.Distance = MockDistance


class TestVectorDBManager(unittest.TestCase):

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_init_success_and_collection_creation(self, MockQdrantClient):
        """Test successful initialization and creation of collections if they don't exist."""
        mock_client_instance = MagicMock()
        MockQdrantClient.return_value = mock_client_instance

        # Simulate collection 'RawThoughtsDB' not existing, others existing
        def mock_get_collection(collection_name):
            if collection_name == TIER_NAMES[0]: # e.g., "RawThoughtsDB"
                # Simulate Qdrant raising an error for a non-existent collection.
                # A common way Qdrant client indicates "not found" is via UnexpectedResponse with 404
                raise UnexpectedResponse(status_code=404, response=MagicMock(), content=b"Not found")
            # For other collections, return a mock CollectionInfo object
            mock_collection_info = MagicMock(spec=models.CollectionInfo)
            mock_collection_info.name = collection_name
            return mock_collection_info

        mock_client_instance.get_collection.side_effect = mock_get_collection
        
        vector_size = 128
        distance_metric = models.Distance.EUCLID

        manager = VectorDBManager(host="test_host", port=1234, api_key="test_key",
                                  vector_size=vector_size, distance_metric=distance_metric)

        MockQdrantClient.assert_called_once_with(host="test_host", port=1234, api_key="test_key")
        self.assertEqual(manager.client, mock_client_instance)
        self.assertEqual(manager.vector_size, vector_size)
        self.assertEqual(manager.distance_metric, distance_metric)

        # Check that get_collection was called for all tiers
        self.assertEqual(mock_client_instance.get_collection.call_count, len(TIER_NAMES))
        for tier_name in TIER_NAMES:
            mock_client_instance.get_collection.assert_any_call(collection_name=tier_name)

        # Check that recreate_collection was called for the tier that "did not exist"
        mock_client_instance.recreate_collection.assert_called_once_with(
            collection_name=TIER_NAMES[0],
            vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
        )
        # Ensure it wasn't called for more than the one missing collection
        self.assertEqual(mock_client_instance.recreate_collection.call_count, 1)


    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_init_qdrant_connection_error(self, MockQdrantClient):
        """Test initialization when QdrantClient fails to connect."""
        MockQdrantClient.side_effect = Exception("Qdrant connection failed")
        
        with patch('builtins.print') as mock_print: # Suppress print statements
            manager = VectorDBManager()
        
        self.assertIsNone(manager.client)
        mock_print.assert_any_call(unittest.mock.ANY) # An error message should have been printed


    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_store_embedding_success(self, MockQdrantClient):
        """Test storing an embedding in a valid tier."""
        mock_client_instance = MagicMock()
        MockQdrantClient.return_value = mock_client_instance
        
        # Bypass __init__ logic for simplicity in this specific test
        manager = VectorDBManager()
        manager.client = mock_client_instance 
        manager.tier_names = TIER_NAMES # Ensure tier_names is set

        points_to_store = [models.PointStruct(id=1, vector=[0.1, 0.2], payload={"data": "test"})]
        target_tier = TIER_NAMES[0]

        manager.store_embedding(tier_name=target_tier, points=points_to_store)

        mock_client_instance.upsert.assert_called_once_with(
            collection_name=target_tier,
            points=points_to_store,
            wait=True
        )

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_store_embedding_invalid_tier(self, MockQdrantClient):
        """Test storing an embedding in an invalid tier."""
        manager = VectorDBManager()
        manager.client = MockQdrantClient.return_value # Dummy client
        manager.tier_names = TIER_NAMES

        with self.assertRaises(ValueError):
            manager.store_embedding(tier_name="InvalidTierNameXYZ", points=[])

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_store_embedding_no_client(self, MockQdrantClient):
        """Test store_embedding when client is not initialized."""
        manager = VectorDBManager()
        manager.client = None # Simulate client init failure
        with self.assertRaises(ValueError) as context:
            manager.store_embedding(TIER_NAMES[0], points=[models.PointStruct(id=1, vector=[0.1], payload={})])
        self.assertIn("Qdrant client is not initialized", str(context.exception))


    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_query_tier_success(self, MockQdrantClient):
        """Test querying a valid tier."""
        mock_client_instance = MagicMock()
        MockQdrantClient.return_value = mock_client_instance
        manager = VectorDBManager()
        manager.client = mock_client_instance
        manager.tier_names = TIER_NAMES

        query_vector = [0.3, 0.4]
        top_k = 5
        target_tier = TIER_NAMES[1]
        test_filter = models.Filter(must=[models.FieldCondition(key="field", match=models.MatchValue(value="value"))])


        manager.query_tier(tier_name=target_tier, query_vector=query_vector, top_k=top_k, filter_payload=test_filter)

        mock_client_instance.search.assert_called_once_with(
            collection_name=target_tier,
            query_vector=query_vector,
            query_filter=test_filter,
            limit=top_k
        )

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_query_tier_invalid_tier(self, MockQdrantClient):
        """Test querying an invalid tier."""
        manager = VectorDBManager()
        manager.client = MockQdrantClient.return_value
        manager.tier_names = TIER_NAMES
        
        with self.assertRaises(ValueError):
            manager.query_tier(tier_name="InvalidTierNameXYZ", query_vector=[0.1])

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_get_collection_info_success(self, MockQdrantClient):
        """Test getting collection info for a valid tier."""
        mock_client_instance = MagicMock()
        MockQdrantClient.return_value = mock_client_instance
        manager = VectorDBManager()
        manager.client = mock_client_instance
        manager.tier_names = TIER_NAMES
        
        target_tier = TIER_NAMES[2]
        manager.get_collection_info(tier_name=target_tier)

        mock_client_instance.get_collection.assert_called_once_with(collection_name=target_tier)

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_delete_point_success(self, MockQdrantClient):
        """Test deleting a point from a valid tier."""
        mock_client_instance = MagicMock()
        MockQdrantClient.return_value = mock_client_instance
        manager = VectorDBManager()
        manager.client = mock_client_instance
        manager.tier_names = TIER_NAMES

        target_tier = TIER_NAMES[3]
        point_id_to_delete = "some-uuid-string" # Or an int, depending on how IDs are used

        manager.delete_point(tier_name=target_tier, point_id=point_id_to_delete)
        
        # Qdrant expects a list of point IDs for deletion via PointIdsList.
        # We need to ensure that the points_selector argument matches this structure.
        # Using ANY here because constructing the exact models.PointIdsList object for assertion
        # can be verbose if the model is complex or has defaults.
        # A more precise check would involve:
        # expected_selector = models.PointIdsList(points=[point_id_to_delete])
        # mock_client_instance.delete.assert_called_once_with(
        #     collection_name=target_tier,
        #     points_selector=expected_selector,
        #     wait=True
        # )
        # For now, using ANY for points_selector if the exact model is tricky.
        # Let's try to be more specific:
        
        # We need to check the structure of the PointIdsList
        # The actual call will be client.delete(collection_name=..., points_selector=models.PointIdsList(points=[point_id]))
        # So, we can use a lambda to check the points attribute of the PointIdsList object.
        
        mock_client_instance.delete.assert_called_once_with(
            collection_name=target_tier,
            points_selector=unittest.mock.ANY, # Using ANY for simplicity
            wait=True
        )
        # To be more precise about points_selector:
        args, kwargs = mock_client_instance.delete.call_args
        selector_arg = kwargs.get('points_selector')
        self.assertIsInstance(selector_arg, models.PointIdsList)
        self.assertEqual(selector_arg.points, [point_id_to_delete])


    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_close_client(self, MockQdrantClient):
        """Test closing the Qdrant client."""
        mock_client_instance = MagicMock()
        MockQdrantClient.return_value = mock_client_instance
        
        # Bypass __init__ for direct client testing
        manager = VectorDBManager()
        manager.client = mock_client_instance

        manager.close()
        mock_client_instance.close.assert_called_once()

    @patch('context_kernel.vector_db_manager.QdrantClient')
    def test_close_no_client(self, MockQdrantClient):
        """Test close method when client is None."""
        manager = VectorDBManager()
        manager.client = None # Simulate client not being initialized

        try:
            manager.close() # Should not raise an error
        except Exception as e:
            self.fail(f"manager.close() raised an exception with no client: {e}")


if __name__ == '__main__':
    unittest.main()
