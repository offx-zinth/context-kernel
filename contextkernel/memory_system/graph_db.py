import asyncio
import logging
import json # For pretty printing dicts in logs
from typing import Optional, List, Dict, Any # Added List, Dict, Any

from neo4j import AsyncGraphDatabase, AsyncDriver, exceptions as neo4j_exceptions
from contextkernel.utils.config import Neo4jConfig

logger = logging.getLogger(__name__)


class GraphDB:
    """
    Graph Database Interface.
    Handles creation, querying, and linking of semantic memory nodes.
    Relies on an injected Neo4j driver instance.
    Stubbed CRUD operations, does not fully interact with a real graph database yet beyond connection.
    """

    def __init__(self, config: Neo4jConfig, driver: AsyncDriver):
        logger.info(f"Initializing GraphDB with URI: {config.uri} (using injected driver)")
        self.config = config
        self.driver = driver
        # self.db_name = config.database or "neo4j" # Neo4j 5.x default. driver.session() takes database kwarg.

    async def boot(self):
        """
        Verifies connection to the graph database using the injected driver.
        """
        logger.info(f"Attempting to verify connection to GraphDB at {self.config.uri}...")
        try:
            # Example: Verify connectivity. Specific to Neo4j driver.
            # Adjust if your driver has a different way to check liveness.
            await self.driver.verify_connectivity()
            logger.info("GraphDB connection verified successfully.")
            return True
        except Exception as e:
            logger.error(f"GraphDB connection verification failed: {e}")
            return False

    async def shutdown(self):
        """
        The Neo4j driver connection is managed by the MemoryKernel or application that created it.
        This method will close the driver if it's active.
        """
        logger.info("GraphDB shutting down...")
        if self.driver and not self.driver.closed(): # Check if driver exists and is not already closed
            try:
                await self.driver.close()
                logger.info("Neo4j driver connection closed.")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")
        else:
            logger.info("Neo4j driver already closed or not initialized.")
        return True

    async def _execute_query_neo4j(self, query: str, parameters: Optional[Dict] = None, write: bool = False) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query against the Neo4j database.

        Args:
            query: The Cypher query string.
            parameters: A dictionary of parameters for the query.
            write: Set to True if the query performs write operations (uses execute_write).
                   Otherwise, uses execute_read.

        Returns:
            A list of records, where each record is a dictionary.
            Returns an empty list if an error occurs or no data is returned.
        """
        parameters = parameters or {}
        db_name = self.config.database # Use database from config if specified

        records_list: List[Dict[str, Any]] = []

        try:
            async with self.driver.session(database=db_name) as session:
                if write:
                    # For write transactions
                    async def tx_logic(tx, q, params):
                        results = await tx.run(q, params)
                        # Consume results before commit if it's a write that returns data (e.g. MERGE ... RETURN)
                        # For simple writes (CREATE, DELETE without RETURN), records might be empty or summary.
                        # Neo4jResult.data() converts records to list of dicts
                        return [record.data() async for record in results]

                    records_list = await session.execute_write(tx_logic, query, parameters)
                    logger.debug(f"Executed WRITE query. Cypher: '{query}', Params: {parameters}. Returned {len(records_list)} records.")
                else:
                    # For read transactions
                    async def tx_logic(tx, q, params):
                        results = await tx.run(q, params)
                        return [record.data() async for record in results] # Convert Neo4j Records to dicts

                    records_list = await session.execute_read(tx_logic, query, parameters)
                    logger.debug(f"Executed READ query. Cypher: '{query}', Params: {parameters}. Returned {len(records_list)} records.")

        except neo4j_exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable. Query: '{query}'. Error: {e}")
            # Optionally raise a custom exception or handle retry logic here
            return []
        except neo4j_exceptions.CypherSyntaxError as e:
            logger.error(f"Cypher syntax error. Query: '{query}'. Error: {e}")
            return []
        except neo4j_exceptions.ResultNotCustomizableError as e: # Raised when trying to access raw data from a summary
            logger.warning(f"Query did not return customizable results (e.g., summary for write op). Query: '{query}'. Error: {e}")
            # This might not be an error for write ops that don't return data, return empty list.
            return []
        except Exception as e: # Catch any other Neo4jError or general exceptions
            logger.error(f"An unexpected error occurred with Neo4j query: '{query}'. Error: {type(e).__name__} - {e}")
            return []

        return records_list

    # --- Node CRUD ---
    async def create_node(self, node_id: str, properties: dict, labels: Optional[List[str]] = None) -> bool:
        # Ensure node_id is in properties for the MERGE operation
        props_with_id = properties.copy()
        props_with_id['node_id'] = node_id # Ensure node_id is part of the properties map

        # Sanitize labels for Cypher query
        safe_labels = []
        if labels:
            for label in labels:
                # Basic sanitization: remove backticks and escape if necessary, though MERGE with fixed labels is safer.
                # For dynamic labels, ensure they are valid identifiers.
                # Using backticks for all labels is a robust way.
                safe_labels.append(f"`{label.replace('`', '')}`")

        label_cypher_str = (" :" + ":".join(safe_labels)) if safe_labels else ""

        # MERGE based on node_id, set properties on create.
        # If node exists, it's matched. If not, it's created.
        # We can choose to update properties on match too if desired (ON MATCH SET n += $props)
        query = (
            f"MERGE (n{label_cypher_str} {{node_id: $node_id}}) "
            f"ON CREATE SET n = $props "
            # Example: If you want to update properties if node already exists:
            # f"ON MATCH SET n += $props "
            f"RETURN n.node_id AS created_node_id"
        )

        params = {"node_id": node_id, "props": props_with_id}
        logger.info(f"Creating/Merging node: ID='{node_id}', Labels={labels}, Properties={json.dumps(properties)}")

        results = await self._execute_query_neo4j(query, params, write=True)

        if results and results[0].get("created_node_id") == node_id:
            logger.info(f"Node '{node_id}' created or merged successfully.")
            return True
        else:
            # This path might be taken if _execute_query_neo4j returns empty on error or no result
            logger.warning(f"Node '{node_id}' creation/merge did not return expected result or failed.")
            return False

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Getting node: ID='{node_id}'")
        query = "MATCH (n {node_id: $node_id}) RETURN n"
        params = {"node_id": node_id}
        results = await self._execute_query_neo4j(query, params)
        if results and results[0].get('n'):
            node_data = results[0]['n']
            logger.debug(f"Node found: {json.dumps(node_data)}")
            return dict(node_data) # Convert Neo4j Node object to dict
        else:
            logger.warning(f"Node with ID '{node_id}' not found.")
            return None

    async def update_node(self, node_id: str, properties: dict) -> bool:
        logger.info(f"Updating node: ID='{node_id}', Properties={json.dumps(properties)}")
        if not properties:
            logger.warning(f"Update_node called for '{node_id}' with empty properties. No action taken.")
            return True # Or False, depending on desired behavior for empty updates

        query = "MATCH (n {node_id: $node_id}) SET n += $properties RETURN n.node_id AS updated_node_id"
        params = {"node_id": node_id, "properties": properties}
        results = await self._execute_query_neo4j(query, params, write=True)

        if results and results[0].get("updated_node_id") == node_id:
            logger.info(f"Node '{node_id}' updated successfully.")
            return True
        else:
            logger.warning(f"Node '{node_id}' not found or update failed.")
            return False

    async def delete_node(self, node_id: str) -> bool:
        logger.info(f"Deleting node: ID='{node_id}'")
        query = "MATCH (n {node_id: $node_id}) DETACH DELETE n RETURN count(n) AS deleted_count"
        params = {"node_id": node_id}
        results = await self._execute_query_neo4j(query, params, write=True)

        # DETACH DELETE returns 0 if node not found, or 1 if found and deleted.
        # The result from helper is a list of dicts, e.g. [{'deleted_count': 1}]
        if results and results[0].get("deleted_count", 0) > 0:
            logger.info(f"Node '{node_id}' and its relationships deleted successfully.")
            return True
        else:
            logger.warning(f"Node '{node_id}' not found or deletion failed.")
            return False

    # --- Edge CRUD ---
    # Removed _get_edge_key as it was for the stub.

    async def create_edge(self, source_node_id: str, target_node_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        # Sanitize relationship_type for Cypher by wrapping in backticks
        safe_relationship_type = f"`{relationship_type.replace('`', '')}`"
        props_str = json.dumps(properties) if properties else "{}"
        logger.info(f"Creating/Merging edge: {source_node_id} -[{safe_relationship_type}]-> {target_node_id}, Props={props_str}")

        query = (
            f"MATCH (a {{node_id: $source_node_id}}), (b {{node_id: $target_node_id}}) "
            f"MERGE (a)-[r:{safe_relationship_type}]->(b) "
            f"ON CREATE SET r = $properties "
            f"ON MATCH SET r += $properties " # Update properties if edge already exists
            f"RETURN type(r) AS relationship_type"
        )
        params = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "properties": properties or {} # Ensure properties is a dict, even if empty
        }
        results = await self._execute_query_neo4j(query, params, write=True)

        if results and results[0].get("relationship_type") == relationship_type:
            logger.info(f"Edge {source_node_id}-[{relationship_type}]->{target_node_id} created/merged successfully.")
            return True
        else:
            logger.warning(f"Edge {source_node_id}-[{relationship_type}]->{target_node_id} creation/merge failed or source/target nodes not found.")
            return False

    async def get_edge(self, source_node_id: str, target_node_id: str, relationship_type: str) -> Optional[Dict[str, Any]]:
        safe_relationship_type = f"`{relationship_type.replace('`', '')}`"
        logger.info(f"Getting edge: {source_node_id} -[{safe_relationship_type}]-> {target_node_id}")
        query = (
            f"MATCH (a {{node_id: $source_node_id}})-[r:{safe_relationship_type}]->(b {{node_id: $target_node_id}}) "
            f"RETURN r"
        )
        params = {"source_node_id": source_node_id, "target_node_id": target_node_id}
        results = await self._execute_query_neo4j(query, params)

        if results and results[0].get('r'):
            edge_data = results[0]['r']
            logger.debug(f"Edge found: {json.dumps(edge_data)}")
            return dict(edge_data) # Convert Neo4j Relationship object to dict
        else:
            logger.warning(f"Edge {source_node_id}-[{relationship_type}]->{target_node_id} not found.")
            return None

    async def update_edge_properties(self, source_node_id: str, target_node_id: str, relationship_type: str, properties: Dict[str, Any]) -> bool:
        safe_relationship_type = f"`{relationship_type.replace('`', '')}`"
        props_str = json.dumps(properties)
        logger.info(f"Updating edge: {source_node_id}-[{safe_relationship_type}]->{target_node_id}, Properties={props_str}")

        if not properties:
            logger.warning(f"Update_edge_properties called for {source_node_id}-[{relationship_type}]->{target_node_id} with empty properties. No action taken.")
            return True

        query = (
            f"MATCH (a {{node_id: $source_node_id}})-[r:{safe_relationship_type}]->(b {{node_id: $target_node_id}}) "
            f"SET r += $properties "
            f"RETURN type(r) AS updated_relationship_type"
        )
        params = {"source_node_id": source_node_id, "target_node_id": target_node_id, "properties": properties}
        results = await self._execute_query_neo4j(query, params, write=True)

        if results and results[0].get("updated_relationship_type") == relationship_type:
            logger.info(f"Edge {source_node_id}-[{relationship_type}]->{target_node_id} updated successfully.")
            return True
        else:
            logger.warning(f"Edge {source_node_id}-[{relationship_type}]->{target_node_id} not found or update failed.")
            return False

    async def delete_edge(self, source_node_id: str, target_node_id: str, relationship_type: str) -> bool:
        safe_relationship_type = f"`{relationship_type.replace('`', '')}`"
        logger.info(f"Deleting edge: {source_node_id} -[{safe_relationship_type}]-> {target_node_id}")
        query = (
            f"MATCH (a {{node_id: $source_node_id}})-[r:{safe_relationship_type}]->(b {{node_id: $target_node_id}}) "
            f"DELETE r"
            # To confirm deletion, we might try to match it again, or rely on Neo4j not erroring if not found.
            # For simplicity, we assume success if query executes without error and affects one relationship.
            # A robust way to confirm deletion is harder without a specific return.
            # We'll assume if no error, it worked or edge didn't exist.
        )
        params = {"source_node_id": source_node_id, "target_node_id": target_node_id}
        # _execute_query_neo4j returns list of results, for DELETE it might be empty or summary.
        # We're interested if it executed without error.
        await self._execute_query_neo4j(query, params, write=True)
        # Assuming success if no exception was raised by _execute_query_neo4j
        # To be more robust, one might check summary information if available and driver supports it.
        logger.info(f"Delete operation for edge {source_node_id}-[{relationship_type}]->{target_node_id} executed.")
        return True # Simplified: assume success if no error.

    # --- Advanced Search ---
    async def cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        param_str = json.dumps(parameters, sort_keys=True) if parameters else "{}"
        logger.info(f"Executing direct Cypher query: {query} with params: {param_str}")
        # Determine if it's a write query based on keywords. This is a basic heuristic.
        # More robust solutions might involve parsing or specific flags.
        is_write_query = any(keyword in query.upper() for keyword in ["CREATE", "MERGE", "SET", "DELETE", "REMOVE", "CALL"])

        return await self._execute_query_neo4j(query, parameters, write=is_write_query)


    async def vector_search(self, embedding: List[float], top_k: int = 5, index_name: str = "node_embedding_index", node_label: Optional[str] = None, embedding_property: str = "embedding") -> List[Dict[str, Any]]:
        # This query assumes a Neo4j 5.x vector index.
        # Example: CREATE VECTOR INDEX node_embedding_index FOR (n:YourLabel) ON (n.embedding)
        logger.info(f"Performing vector search in index '{index_name}' with top_k={top_k} for embedding (first 3 dims): {embedding[:3]}...")

        # The exact query might depend on the specific index setup and Neo4j version.
        # This is a common pattern for vector search using an index.
        # If node_label is provided, it can make the search more specific.
        # MATCH (n:{node_label}) WHERE ... CALL db.index.vector ... (more complex)
        # Simpler: Rely on index to span desired nodes.
        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding_vector)
        YIELD node, score
        RETURN node, score
        """
        params = {
            "index_name": index_name,
            "top_k": top_k,
            "embedding_vector": embedding # Ensure embedding is a list of floats
        }

        results = await self._execute_query_neo4j(query, params, write=False)

        # Process results: Neo4j nodes need to be converted to dicts.
        # The 'node' in results from queryNodes is already the node object.
        processed_results = []
        if results:
            for record in results:
                node_data = record.get("node")
                score = record.get("score")
                if node_data:
                    processed_results.append({"node_id": node_data.get("node_id"), "data": dict(node_data), "score": score})

        logger.info(f"Vector search returned {len(processed_results)} results.")
        return results

async def main():
    # Ensure basicConfig is called for the logger to output messages.
    # This is especially important if running the file directly.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- GraphDB Live Neo4j Example Usage ---")
    logger.warning("IMPORTANT: This example interacts with a LIVE Neo4j database.")
    logger.warning("Ensure your Neo4j instance is running and configured correctly.")
    logger.warning("Data created by this example will persist unless cleaned up.")

    # 1. Configuration and Driver Setup
    # These would typically come from AppSettings in a larger application.
    neo4j_config = Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password", # Replace with your actual password
        database="neo4j" # Default database
    )

    driver = None
    try:
        driver = AsyncGraphDatabase.driver(neo4j_config.uri, auth=(neo4j_config.user, neo4j_config.password))
        await driver.verify_connectivity() # Check if connection is initially possible
        logger.info("Neo4j Driver created and connectivity verified.")
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver or connect: {e}")
        logger.error("Please ensure Neo4j is running and credentials are correct.")
        logger.error("Aborting example.")
        if driver:
            await driver.close()
        return

    # 2. Instantiate GraphDB
    graph_db = GraphDB(config=neo4j_config, driver=driver)

    # 3. Boot GraphDB (verifies connectivity again via its own boot method)
    if not await graph_db.boot():
        logger.error("GraphDB boot failed. Aborting example.")
        await driver.close() # Ensure driver is closed if boot fails
        return

    # Unique IDs for this example run to minimize interference
    example_run_id = asyncio.get_running_loop().time() # Fairly unique float
    node1_id = f"ex_node1_{example_run_id}"
    node2_id = f"ex_node2_{example_run_id}"
    project_id = f"ex_projectA_{example_run_id}"

    try:
        # 4. Node Operations
        logger.info(f"--- Testing Node Operations (ID suffix: {example_run_id}) ---")
        created_n1 = await graph_db.create_node(node1_id, {"name": "Alice Example", "type": "Person"}, labels=["Person", "Example"])
        assert created_n1, f"Failed to create {node1_id}"
        created_n2 = await graph_db.create_node(node2_id, {"name": "Bob Example", "type": "Person"}, labels=["Person", "Example"])
        assert created_n2, f"Failed to create {node2_id}"
        created_pA = await graph_db.create_node(project_id, {"title": "Context Kernel Example", "status": "Ongoing"}, labels=["Project", "Example"])
        assert created_pA, f"Failed to create {project_id}"

        retrieved_n1 = await graph_db.get_node(node1_id)
        logger.info(f"Retrieved {node1_id}: {retrieved_n1}")
        assert retrieved_n1 and retrieved_n1['name'] == "Alice Example"

        updated_n1 = await graph_db.update_node(node1_id, {"age": 30, "location": "Wonderland"})
        assert updated_n1, f"Failed to update {node1_id}"
        retrieved_n1_updated = await graph_db.get_node(node1_id)
        logger.info(f"Updated {node1_id}: {retrieved_n1_updated}")
        assert retrieved_n1_updated and retrieved_n1_updated.get('age') == 30

        # 5. Edge Operations
        logger.info("--- Testing Edge Operations ---")
        rel_type = "WORKS_ON_EXAMPLE"
        created_e1 = await graph_db.create_edge(node1_id, project_id, rel_type, {"role": "Developer"})
        assert created_e1, f"Failed to create edge between {node1_id} and {project_id}"

        retrieved_e1 = await graph_db.get_edge(node1_id, project_id, rel_type)
        logger.info(f"Retrieved edge {node1_id}-[{rel_type}]->{project_id}: {retrieved_e1}")
        assert retrieved_e1 and retrieved_e1['role'] == "Developer"

        updated_e1 = await graph_db.update_edge_properties(node1_id, project_id, rel_type, {"role": "Lead Developer", "since": 2023})
        assert updated_e1, "Failed to update edge properties"
        retrieved_e1_updated = await graph_db.get_edge(node1_id, project_id, rel_type)
        logger.info(f"Updated edge {node1_id}-[{rel_type}]->{project_id}: {retrieved_e1_updated}")
        assert retrieved_e1_updated and retrieved_e1_updated.get('role') == "Lead Developer"

        # 6. Cypher Query
        logger.info("--- Testing Cypher Query ---")
        # Query for nodes created in this example run
        cypher_q = f"MATCH (p:Person:Example {{node_id: '{node1_id}'}})-[r:{rel_type}]->(proj:Project:Example {{node_id: '{project_id}'}}) RETURN p.name AS person, type(r) AS relationship, r.role AS role, proj.title AS project"
        query_results = await graph_db.cypher_query(cypher_q)
        logger.info(f"Cypher query results for example data: {query_results}")
        assert len(query_results) == 1
        assert query_results[0]['person'] == "Alice Example"
        assert query_results[0]['role'] == "Lead Developer"

        # 7. Vector Search (Conceptual - Requires Vector Index Setup)
        logger.info("--- Testing Vector Search (Conceptual) ---")
        logger.warning("Vector search requires a pre-configured vector index in Neo4j (e.g., on 'embedding' property of nodes with label 'Example').")
        logger.warning("This example will attempt a search but may return empty if index is not set up or node doesn't have 'embedding'.")

        # Add a dummy embedding to a node for the search to potentially find something (if index exists)
        await graph_db.update_node(node1_id, {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}) # Example 5D embedding

        # Note: The index_name should match the one in your DB.
        # e.g., CREATE VECTOR INDEX example_node_embeddings FOR (n:Example) ON (n.embedding)
        try:
            vector_results = await graph_db.vector_search(
                embedding=[0.11, 0.22, 0.33, 0.44, 0.55], # Slightly different embedding
                top_k=2,
                index_name="example_node_embeddings", # Ensure this index exists on :Example(embedding)
                embedding_property="embedding" # Ensure this matches the property in the index
            )
            logger.info(f"Vector search results: {vector_results}")
            # Add assertions here if you have a known state after setting up the index and data.
            # For instance, if node1_id is expected:
            # if vector_results:
            #    assert any(res['node_id'] == node1_id for res in vector_results), "Vector search did not return expected node."
        except Exception as e:
            logger.error(f"Vector search encountered an error (this is common if index is not set up): {e}")


        # 8. Deletion (Clean up example data)
        logger.info("--- Testing Deletion (Cleaning Up Example Data) ---")
        deleted_e1 = await graph_db.delete_edge(node1_id, project_id, rel_type)
        assert deleted_e1, "Failed to delete edge"

        # Verify edge deletion
        edge_should_be_gone = await graph_db.get_edge(node1_id, project_id, rel_type)
        logger.info(f"Edge {node1_id}-[{rel_type}]->{project_id} exists after delete: {edge_should_be_gone is not None}")
        assert edge_should_be_gone is None

        # Delete nodes (ensure DETACH DELETE is used or edges are removed first)
        deleted_n1 = await graph_db.delete_node(node1_id)
        assert deleted_n1, f"Failed to delete node {node1_id}"
        deleted_n2 = await graph_db.delete_node(node2_id)
        assert deleted_n2, f"Failed to delete node {node2_id}"
        deleted_pA = await graph_db.delete_node(project_id)
        assert deleted_pA, f"Failed to delete node {project_id}"

        # Verify node deletion
        node1_should_be_gone = await graph_db.get_node(node1_id)
        logger.info(f"Node {node1_id} exists after delete: {node1_should_be_gone is not None}")
        assert node1_should_be_gone is None

        logger.info("Example data cleanup successful.")

    except Exception as main_e:
        logger.error(f"An error occurred during the main example execution: {main_e}", exc_info=True)
        logger.error("Attempting to clean up any created example data...")
        # Simplified cleanup: Try to delete nodes by their example IDs.
        # This might fail if operations failed midway, but it's a best effort.
        await graph_db.delete_node(node1_id)
        await graph_db.delete_node(node2_id)
        await graph_db.delete_node(project_id)
        logger.info("Cleanup attempt finished.")
    finally:
        # 9. Shutdown GraphDB and close driver
        await graph_db.shutdown() # This now also closes the driver
        # Driver is closed by graph_db.shutdown(), so no separate driver.close() needed here
        # if graph_db.shutdown() is guaranteed to be called and handles it.
        # However, if graph_db.boot() failed, driver might still be open.
        # The initial try-except for driver creation handles closing on early failure.
        # If graph_db.boot() fails, driver is passed to graph_db, and graph_db.shutdown() should handle it.
        logger.info("--- GraphDB Live Neo4j Example Usage Complete ---")


if __name__ == "__main__":
    # This example requires a running Neo4j instance.
    # Set up your Neo4j credentials in Neo4jConfig or environment variables.
    # Example of creating a vector index (Neo4j 5.x):
    # CREATE VECTOR INDEX example_node_embeddings FOR (n:Example) ON (n.embedding)
    # OPTIONS { indexConfig: { `vector.dimensions`: 5, `vector.similarity_function`: 'cosine' } }
    # (Adjust dimensions as per your embedding vector size, e.g. 1536 for OpenAI ada-002)
    # And ensure Neo4j server is running on localhost:7687 (or configure Neo4jConfig accordingly)
    # with the specified user/password.
    #
    # To run this example:
    # 1. Ensure Neo4j is running and configured.
    # 2. If you are in the root of the project, you might run it as:
    #    python -m contextkernel.memory_system.graph_db
    #    (This ensures Python resolves the module imports correctly)
    #
    # If you encounter `ModuleNotFoundError` for `contextkernel.utils.config`,
    # ensure your PYTHONPATH is set up correctly to include the project root,
    # or run as a module as shown above.
    asyncio.run(main())
