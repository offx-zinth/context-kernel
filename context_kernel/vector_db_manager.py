# Core Infrastructure: Production-ready Qdrant vector DB interface. (Marked as per review)
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import QdrantException, UnexpectedResponse
import time # For potential retries or delays if needed, though not explicitly used in basic init

TIER_NAMES = ["RawThoughtsDB", "ChunkSummaryDB", "ExecutiveSummaryDB", "ShortTermMemoryDB", "LongTermMemoryDB"]

class VectorDBManager:
    """
    Manages interactions with a Qdrant vector database, organizing data into predefined tiers (collections).
    """

    def __init__(self, host: str = 'localhost', port: int = 6333, api_key: str = None,
                 vector_size: int = 1536, distance_metric: models.Distance = models.Distance.COSINE):
        """
        Initializes the VectorDBManager, connects to Qdrant, and ensures all predefined tiers (collections) exist.

        Args:
            host (str): Hostname or IP address of the Qdrant server.
            port (int): Port number of the Qdrant server.
            api_key (str, optional): API key for Qdrant Cloud or secured instances.
            vector_size (int): The dimensionality of the vectors to be stored.
                               Default is 1536 (OpenAI's text-embedding-ada-002).
            distance_metric (models.Distance): The distance metric to use for vector comparison
                                               (e.g., COSINE, EUCLID, DOT). Default is COSINE.
        """
        self.client = None
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.tier_names = TIER_NAMES

        try:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)
            # You can ping the client to ensure connection, though operations below will also test it.
            # self.client.health_check() 
            print(f"Successfully connected to Qdrant at {host}:{port}")

            for tier_name in self.tier_names:
                try:
                    # Check if collection exists
                    self.client.get_collection(collection_name=tier_name)
                    print(f"Collection '{tier_name}' already exists.")
                except UnexpectedResponse as e:
                    # This exception (or similar, depending on qdrant_client version)
                    # often indicates "Not Found" with status code 404.
                    if e.status_code == 404:
                        print(f"Collection '{tier_name}' not found. Creating...")
                        self.client.recreate_collection(
                            collection_name=tier_name,
                            vectors_config=models.VectorParams(size=self.vector_size, distance=self.distance_metric)
                        )
                        print(f"Collection '{tier_name}' created successfully.")
                    else:
                        # Handle other unexpected errors during get_collection
                        print(f"Error checking collection '{tier_name}': {e}. Status code: {e.status_code}")
                        raise  # Re-raise the exception if it's not a simple "not found"
                except QdrantException as e:
                    # Broad Qdrant exception for other cases, e.g., network issues during check
                    # This might be too broad if UnexpectedResponse with status code is reliable for "not found"
                    print(f"A Qdrant specific error occurred while checking or creating collection '{tier_name}': {e}")
                    # Depending on the error, you might try to create it still, or re-raise
                    # For now, let's assume if it's not UnexpectedResponse 404, it's a more serious issue.
                    raise
                except Exception as e:
                    # Catch-all for other errors like network issues during client operations
                    print(f"An unexpected error occurred while ensuring collection '{tier_name}' exists: {e}")
                    # Decide if to re-raise or try to continue. For now, re-raise.
                    raise

        except QdrantException as e:
            print(f"Failed to initialize Qdrant client or setup collections: {e}")
            self.client = None # Ensure client is None if setup fails
        except Exception as e:
            print(f"An unexpected error occurred during VectorDBManager initialization: {e}")
            self.client = None

    def store_embedding(self, tier_name: str, points: list[models.PointStruct]) -> models.UpdateResult | None:
        """
        Stores or updates embeddings (points) in the specified tier (collection).

        Args:
            tier_name (str): The name of the tier (collection) to store the points in.
            points (list[models.PointStruct]): A list of Qdrant PointStruct objects.
                                               Each PointStruct requires 'id', 'vector', and 'payload'.

        Returns:
            models.UpdateResult | None: The result of the upsert operation from Qdrant, or None if an error occurs.

        Raises:
            ValueError: If tier_name is not valid or client is not initialized.
        """
        if not self.client:
            raise ValueError("Qdrant client is not initialized.")
        if tier_name not in self.tier_names:
            raise ValueError(f"Invalid tier_name: '{tier_name}'. Must be one of {self.tier_names}")
        if not points:
            print("No points provided to store.")
            return None

        try:
            result = self.client.upsert(collection_name=tier_name, points=points, wait=True)
            return result
        except QdrantException as e:
            print(f"Error upserting points to tier '{tier_name}': {e}")
            # Potentially log more details or handle specific error codes
        except Exception as e:
            print(f"An unexpected error occurred during upsert to '{tier_name}': {e}")
        return None

    def query_tier(self, tier_name: str, query_vector: list[float], top_k: int = 10,
                     filter_payload: models.Filter = None) -> list[models.ScoredPoint] | None:
        """
        Performs a semantic search for similar vectors in the specified tier.

        Args:
            tier_name (str): The name of the tier (collection) to query.
            query_vector (list[float]): The vector to search for.
            top_k (int): The number of closest results to return.
            filter_payload (models.Filter, optional): Qdrant filter conditions for the search.

        Returns:
            list[models.ScoredPoint] | None: A list of ScoredPoint objects, or None if an error occurs.

        Raises:
            ValueError: If tier_name is not valid or client is not initialized.
        """
        if not self.client:
            raise ValueError("Qdrant client is not initialized.")
        if tier_name not in self.tier_names:
            raise ValueError(f"Invalid tier_name: '{tier_name}'. Must be one of {self.tier_names}")

        try:
            search_result = self.client.search(
                collection_name=tier_name,
                query_vector=query_vector,
                query_filter=filter_payload,
                limit=top_k
            )
            return search_result
        except QdrantException as e:
            print(f"Error querying tier '{tier_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred during query in '{tier_name}': {e}")
        return None

    def get_collection_info(self, tier_name: str) -> models.CollectionInfo | None:
        """
        Retrieves information about a specific collection (tier).

        Args:
            tier_name (str): The name of the tier (collection).

        Returns:
            models.CollectionInfo | None: Information about the collection, or None if an error occurs.

        Raises:
            ValueError: If tier_name is not valid or client is not initialized.
        """
        if not self.client:
            raise ValueError("Qdrant client is not initialized.")
        if tier_name not in self.tier_names:
            raise ValueError(f"Invalid tier_name: '{tier_name}'. Must be one of {self.tier_names}")

        try:
            collection_info = self.client.get_collection(collection_name=tier_name)
            return collection_info
        except QdrantException as e:
            print(f"Error getting collection info for '{tier_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while getting info for '{tier_name}': {e}")
        return None

    def delete_point(self, tier_name: str, point_id: models.ExtendedPointId) -> models.UpdateResult | None:
        """
        Deletes a point from the specified tier by its ID.

        Args:
            tier_name (str): The name of the tier (collection).
            point_id (models.ExtendedPointId): The ID of the point to delete (can be int or UUID string).

        Returns:
            models.UpdateResult | None: The result of the delete operation, or None if an error occurs.

        Raises:
            ValueError: If tier_name is not valid or client is not initialized.
        """
        if not self.client:
            raise ValueError("Qdrant client is not initialized.")
        if tier_name not in self.tier_names:
            raise ValueError(f"Invalid tier_name: '{tier_name}'. Must be one of {self.tier_names}")

        try:
            # Qdrant expects a list of point IDs for deletion.
            result = self.client.delete(
                collection_name=tier_name,
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True
            )
            return result
        except QdrantException as e:
            print(f"Error deleting point_id '{point_id}' from tier '{tier_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while deleting point from '{tier_name}': {e}")
        return None
    
    def close(self):
        """
        Closes the Qdrant client connection if it's open.
        """
        if self.client:
            try:
                self.client.close()
                print("Qdrant client connection closed.")
            except Exception as e:
                print(f"Error closing Qdrant client: {e}")


if __name__ == '__main__':
    from uuid import uuid4

    # --- IMPORTANT ---
    # This example usage block assumes a Qdrant server is running on localhost:6333.
    # If not, these operations will fail. The class itself is designed to be testable
    # without a running server for its logic, but this block will try to connect.
    #
    # Set to True to run example, False to skip.
    RUN_QDRANT_EXAMPLES = False 
    # Use a very small vector size for this example to avoid large embeddings.
    # Real applications would use larger sizes like 1536 (OpenAI) or 384/768 (SentenceTransformers).
    EXAMPLE_VECTOR_SIZE = 4 

    if RUN_QDRANT_EXAMPLES:
        print("Initializing VectorDBManager for example usage...")
        # Note: If Qdrant is not running, this initialization will print errors and client will be None.
        vector_db_manager = VectorDBManager(host='localhost', port=6333, vector_size=EXAMPLE_VECTOR_SIZE)

        if vector_db_manager.client:
            print("\n--- VectorDBManager Example Operations ---")
            
            # Example point
            point_id_1 = str(uuid4())
            example_point_1 = models.PointStruct(
                id=point_id_1,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"text": "This is the first example thought about apples.", "source": "test_module_A", "timestamp": time.time()}
            )
            
            point_id_2 = str(uuid4())
            example_point_2 = models.PointStruct(
                id=point_id_2,
                vector=[0.5, 0.6, 0.7, 0.8],
                payload={"text": "A second thought, this one about bananas.", "source": "test_module_B", "timestamp": time.time() + 1}
            )

            target_tier = "RawThoughtsDB" # Example tier

            try:
                # Store embeddings
                print(f"\nAttempting to store points in '{target_tier}'...")
                store_result = vector_db_manager.store_embedding(tier_name=target_tier, points=[example_point_1, example_point_2])
                if store_result:
                    print(f"Store operation result: {store_result.status}")
                else:
                    print(f"Store operation failed or returned None.")

                # Get collection info (after potential creation/upsert)
                print(f"\nAttempting to get info for '{target_tier}'...")
                collection_info = vector_db_manager.get_collection_info(target_tier)
                if collection_info:
                    print(f"Collection '{target_tier}' info: Points count = {collection_info.points_count}, Status = {collection_info.status}")
                else:
                    print(f"Could not get info for '{target_tier}'.")


                # Query the tier
                print(f"\nAttempting to query '{target_tier}'...")
                # Query vector similar to example_point_1
                query_results = vector_db_manager.query_tier(tier_name=target_tier, query_vector=[0.11, 0.19, 0.31, 0.41], top_k=1)
                if query_results:
                    print(f"Query results from '{target_tier}':")
                    for hit in query_results:
                        print(f"  Point ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")
                else:
                    print(f"Query failed or returned no results.")

                # Query with a filter
                print(f"\nAttempting to query '{target_tier}' with a filter (source='test_module_B')...")
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value="test_module_B")
                        )
                    ]
                )
                filtered_results = vector_db_manager.query_tier(
                    tier_name=target_tier, 
                    query_vector=[0.5, 0.5, 0.5, 0.5], # Generic query vector
                    top_k=5, 
                    filter_payload=filter_condition
                )
                if filtered_results:
                    print(f"Filtered query results from '{target_tier}':")
                    for hit in filtered_results:
                        print(f"  Point ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")
                else:
                    print("Filtered query failed or returned no results.")


                # Delete a point
                print(f"\nAttempting to delete point '{point_id_1}' from '{target_tier}'...")
                delete_result = vector_db_manager.delete_point(target_tier, point_id_1)
                if delete_result:
                    print(f"Delete operation result for point '{point_id_1}': {delete_result.status}")
                else:
                    print(f"Delete operation failed for point '{point_id_1}'.")
                
                # Verify deletion by trying to get collection info again
                collection_info_after_delete = vector_db_manager.get_collection_info(target_tier)
                if collection_info_after_delete:
                    print(f"Collection '{target_tier}' info after delete: Points count = {collection_info_after_delete.points_count}")


            except ValueError as ve:
                print(f"ValueError during example operations: {ve}")
            except Exception as e:
                # This will catch QdrantExceptions if the server is down or other issues.
                print(f"An error occurred during VectorDBManager example operations: {e}")
            finally:
                if vector_db_manager.client:
                    vector_db_manager.close()
        else:
            print("VectorDBManager client not initialized. Skipping Qdrant examples. Ensure Qdrant server is running.")
    else:
        print("RUN_QDRANT_EXAMPLES is False. Skipping VectorDBManager example operations.")
        print("To run examples, ensure a Qdrant instance is running and set RUN_QDRANT_EXAMPLES to True in the script.")

    print("\nVectorDBManager class definition complete.")
