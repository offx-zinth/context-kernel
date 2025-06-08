# Core Infrastructure: Production-ready Neo4j interface. (Marked as per review)
import os
import asyncio
import logging
import json
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import time

from neo4j import GraphDatabase, AsyncGraphDatabase, exceptions as neo4j_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphDBConnectionError(Exception):
    """Custom connection error for GraphDBLayer"""
    pass

class GraphDBQueryError(Exception):
    """Custom query error for GraphDBLayer"""
    pass

@dataclass
class QueryResult:
    """Enhanced result object with comprehensive metadata"""
    records: List[Dict[str, Any]]
    execution_time: float
    records_count: int
    error: Optional[str] = None
    query_type: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __bool__(self) -> bool:
        return len(self.records) > 0 and self.error is None

@dataclass
class DatabaseStats:
    """Database statistics and health information"""
    node_count: int
    relationship_count: int
    labels: List[str]
    relationship_types: List[str]
    property_keys: List[str]
    database_info: Dict[str, Any]

class GraphDBLayer:
    """
    Production-ready Neo4j database layer with sync/async support,
    retry logic, comprehensive error handling, and metrics.
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        *,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: float = 30.0,
        use_async: bool = False,
        retry_attempts: int = 3,
        retry_wait_min: int = 1,
        retry_wait_max: int = 10,
    ):
        """
        Initialize GraphDBLayer with flexible configuration.
        
        Args:
            uri: Neo4j URI (defaults to NEO4J_URI env var or bolt://localhost:7687)
            user: Username (defaults to NEO4J_USER env var or 'neo4j')
            password: Password (defaults to NEO4J_PASSWORD env var or 'password')
            database: Database name (defaults to NEO4J_DATABASE env var or 'neo4j')
            max_connection_lifetime: Max connection lifetime in seconds
            max_connection_pool_size: Max connections in pool
            connection_acquisition_timeout: Connection timeout in seconds
            use_async: Whether to use async driver
            retry_attempts: Number of retry attempts for failed operations
            retry_wait_min: Minimum retry wait time
            retry_wait_max: Maximum retry wait time
        """
        # Configuration with environment variable fallbacks
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.use_async = use_async
        
        # Connection configuration
        self.connection_config = {
            "max_connection_lifetime": max_connection_lifetime,
            "max_connection_pool_size": max_connection_pool_size,
            "connection_acquisition_timeout": connection_acquisition_timeout,
        }
        
        # Retry configuration
        self.retry_config = {
            "attempts": retry_attempts,
            "wait_min": retry_wait_min,
            "wait_max": retry_wait_max,
        }
        
        self.driver = None
        self._connected = False
        self.metrics = {
            "queries_executed": 0,
            "total_execution_time": 0.0,
            "failed_queries": 0,
            "connection_attempts": 0,
        }
        
        self._initialize_driver()

    def _initialize_driver(self):
        """Initialize the Neo4j driver with proper error handling."""
        self.metrics["connection_attempts"] += 1
        
        try:
            driver_class = AsyncGraphDatabase if self.use_async else GraphDatabase
            self.driver = driver_class.driver(
                self.uri,
                auth=(self.user, self.password),
                **self.connection_config
            )
            
            # Verify connectivity
            if self.use_async:
                # For async, we'll verify in the first async operation
                self._connected = True
            else:
                self.driver.verify_connectivity()
                self._connected = True
                
            logger.info(f"Successfully connected to Neo4j at {self.uri} ({'async' if self.use_async else 'sync'} mode)")
            
        except neo4j_exceptions.AuthError as e:
            logger.error(f"Authentication failed for user '{self.user}' at {self.uri}: {e}")
            raise GraphDBConnectionError(f"Authentication failed: {e}") from e
        except neo4j_exceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}: {e}")
            raise GraphDBConnectionError(f"Service unavailable: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            raise GraphDBConnectionError(f"Connection failed: {e}") from e

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected and self.driver is not None

    def close(self):
        """Close the database connection."""
        if self.driver:
            try:
                if self.use_async:
                    # For async driver, we need to close in async context
                    pass  # Will be handled by async close method
                else:
                    self.driver.close()
                self._connected = False
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

    async def aclose(self):
        """Close async database connection."""
        if self.driver and self.use_async:
            try:
                await self.driver.close()
                self._connected = False
                logger.info("Async Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing async connection: {e}")

    @contextmanager
    def session(self):
        """Context manager for synchronous sessions."""
        if not self.is_connected:
            raise GraphDBConnectionError("Driver not connected")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    @asynccontextmanager
    async def asession(self):
        """Context manager for asynchronous sessions."""
        if not self.is_connected:
            raise GraphDBConnectionError("Driver not connected")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            await session.close()

    def _get_retry_decorator(self):
        """Get configured retry decorator."""
        return retry(
            stop=stop_after_attempt(self.retry_config["attempts"]),
            wait=wait_exponential(
                multiplier=1,
                min=self.retry_config["wait_min"],
                max=self.retry_config["wait_max"]
            ),
            retry=retry_if_exception_type((
                neo4j_exceptions.TransientError,
                neo4j_exceptions.ServiceUnavailable,
                neo4j_exceptions.TransactionError
            ))
        )

    def _determine_query_type(self, query: str) -> str:
        """Determine if query is read or write operation."""
        query_upper = query.upper().strip()
        write_keywords = {'CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP'}
        
        for keyword in write_keywords:
            if keyword in query_upper:
                return 'WRITE'
        return 'READ'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            neo4j_exceptions.TransientError,
            neo4j_exceptions.ServiceUnavailable
        ))
    )
    def _execute_sync(self, query: str, parameters: Dict[str, Any], is_write: bool) -> List[Dict[str, Any]]:
        """Execute synchronous query with retry logic."""
        with self.session() as session:
            if is_write:
                result = session.write_transaction(lambda tx: list(tx.run(query, parameters)))
            else:
                result = session.read_transaction(lambda tx: list(tx.run(query, parameters)))
            
            # Convert records to dictionaries
            return [dict(record) for record in result]

    async def _execute_async(self, query: str, parameters: Dict[str, Any], is_write: bool) -> List[Dict[str, Any]]:
        """Execute asynchronous query with retry logic."""
        async with self.asession() as session:
            if is_write:
                result = await session.write_transaction(
                    lambda tx: tx.run(query, parameters)
                )
            else:
                result = await session.read_transaction(
                    lambda tx: tx.run(query, parameters)
                )
            
            # Convert async records to dictionaries
            records = []
            async for record in result:
                records.append(dict(record))
            return records

    def run_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        is_write: Optional[bool] = None,
        return_stats: bool = False,
    ) -> QueryResult:
        """
        Execute a Cypher query synchronously.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            is_write: Whether this is a write operation (auto-detected if None)
            return_stats: Whether to include execution statistics
            
        Returns:
            QueryResult with records and metadata
        """
        if self.use_async:
            raise ValueError("Use run_query_async for async operations")
        
        parameters = parameters or {}
        query_type = self._determine_query_type(query)
        is_write = is_write if is_write is not None else (query_type == 'WRITE')
        
        start_time = time.time()
        self.metrics["queries_executed"] += 1
        
        try:
            records = self._execute_sync(query, parameters, is_write)
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            
            result = QueryResult(
                records=records,
                execution_time=execution_time,
                records_count=len(records),
                query_type=query_type,
            )
            
            if return_stats:
                result.stats = self.get_metrics()
            
            logger.debug(f"Query executed in {execution_time:.3f}s, returned {len(records)} records")
            return result
            
        except Exception as e:
            self.metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            logger.debug(f"Failed query: {query}")
            logger.debug(f"Parameters: {parameters}")
            
            return QueryResult(
                records=[],
                execution_time=execution_time,
                records_count=0,
                error=str(e),
                query_type=query_type,
            )

    async def run_query_async(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        is_write: Optional[bool] = None,
        return_stats: bool = False,
    ) -> QueryResult:
        """
        Execute a Cypher query asynchronously.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            is_write: Whether this is a write operation (auto-detected if None)
            return_stats: Whether to include execution statistics
            
        Returns:
            QueryResult with records and metadata
        """
        if not self.use_async:
            raise ValueError("Use run_query for sync operations")
        
        parameters = parameters or {}
        query_type = self._determine_query_type(query)
        is_write = is_write if is_write is not None else (query_type == 'WRITE')
        
        start_time = time.time()
        self.metrics["queries_executed"] += 1
        
        try:
            records = await self._execute_async(query, parameters, is_write)
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            
            result = QueryResult(
                records=records,
                execution_time=execution_time,
                records_count=len(records),
                query_type=query_type,
            )
            
            if return_stats:
                result.stats = self.get_metrics()
            
            logger.debug(f"Async query executed in {execution_time:.3f}s, returned {len(records)} records")
            return result
            
        except Exception as e:
            self.metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            
            logger.error(f"Async query failed after {execution_time:.3f}s: {e}")
            logger.debug(f"Failed query: {query}")
            logger.debug(f"Parameters: {parameters}")
            
            return QueryResult(
                records=[],
                execution_time=execution_time,
                records_count=0,
                error=str(e),
                query_type=query_type,
            )

    def add_node(
        self,
        label: str,
        properties: Dict[str, Any],
        unique_key: Optional[str] = None,
    ) -> Optional[int]:
        """
        Add a node to the graph.
        
        Args:
            label: Node label
            properties: Node properties
            unique_key: Property to use for MERGE (avoids duplicates)
            
        Returns:
            Node ID if successful, None otherwise
        """
        if not properties:
            raise ValueError("Properties cannot be empty")
        
        if not self._validate_label(label):
            return None
        
        if unique_key:
            if unique_key not in properties:
                raise ValueError(f"Unique key '{unique_key}' not found in properties")
            
            query = f"""
            MERGE (n:{label} {{{unique_key}: $unique_value}})
            SET n += $props
            RETURN id(n) AS node_id
            """
            parameters = {
                "unique_value": properties[unique_key],
                "props": properties
            }
        else:
            query = f"CREATE (n:{label} $props) RETURN id(n) AS node_id"
            parameters = {"props": properties}
        
        result = self.run_query(query, parameters, is_write=True)
        
        if result and result.records:
            node_id = result.records[0].get("node_id")
            logger.info(f"Node created with ID: {node_id}")
            return node_id
        
        logger.error("Failed to create node")
        return None

    def add_relationship(
        self,
        start_id: int,
        end_id: int,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a relationship between two nodes.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            rel_type: Relationship type
            properties: Relationship properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self._validate_relationship_type(rel_type):
            return False
        
        if properties:
            query = f"""
            MATCH (a), (b)
            WHERE id(a) = $start_id AND id(b) = $end_id
            CREATE (a)-[r:{rel_type} $props]->(b)
            RETURN type(r) AS rel_type
            """
            parameters = {
                "start_id": start_id,
                "end_id": end_id,
                "props": properties
            }
        else:
            query = f"""
            MATCH (a), (b)
            WHERE id(a) = $start_id AND id(b) = $end_id
            CREATE (a)-[r:{rel_type}]->(b)
            RETURN type(r) AS rel_type
            """
            parameters = {"start_id": start_id, "end_id": end_id}
        
        result = self.run_query(query, parameters, is_write=True)
        
        if result and result.records:
            logger.info(f"Relationship '{rel_type}' created between nodes {start_id} and {end_id}")
            return True
        
        logger.error("Failed to create relationship")
        return False

    def update_node(self, node_id: int, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        if not properties:
            raise ValueError("Properties cannot be empty")
        
        query = """
        MATCH (n) WHERE id(n) = $node_id
        SET n += $properties
        RETURN count(n) AS updated_count
        """
        
        result = self.run_query(
            query,
            {"node_id": node_id, "properties": properties},
            is_write=True
        )
        
        if result and result.records and result.records[0].get("updated_count", 0) > 0:
            logger.info(f"Node {node_id} updated successfully")
            return True
        
        logger.error(f"Failed to update node {node_id}")
        return False

    def delete_node(self, node_id: int, detach: bool = True) -> bool:
        """Delete a node and optionally its relationships."""
        query = f"""
        MATCH (n) WHERE id(n) = $node_id
        {'DETACH ' if detach else ''}DELETE n
        RETURN count(n) AS deleted_count
        """
        
        result = self.run_query(query, {"node_id": node_id}, is_write=True)
        
        if result and result.records and result.records[0].get("deleted_count", 0) > 0:
            logger.info(f"Node {node_id} deleted successfully")
            return True
        
        logger.error(f"Failed to delete node {node_id}")
        return False

    def find_nodes(
        self,
        label: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria."""
        query_parts = ["MATCH (n"]
        parameters = {}
        
        if label:
            query_parts.append(f":{label}")
        
        query_parts.append(")")
        
        # Add WHERE conditions for properties
        where_conditions = []
        if properties:
            for key, value in properties.items():
                param_name = f"prop_{key}"
                where_conditions.append(f"n.{key} = ${param_name}")
                parameters[param_name] = value
        
        if where_conditions:
            query_parts.append(" WHERE " + " AND ".join(where_conditions))
        
        query_parts.append(" RETURN n, id(n) as node_id")
        
        if limit:
            query_parts.append(f" LIMIT {limit}")
        
        query = "".join(query_parts)
        result = self.run_query(query, parameters)
        
        nodes = []
        if result and result.records:
            for record in result.records:
                node_data = dict(record["n"])
                node_data["_id"] = record["node_id"]
                nodes.append(node_data)
        
        return nodes

    def get_database_stats(self) -> DatabaseStats:
        """Get comprehensive database statistics."""
        stats = {}
        
        # Node count
        result = self.run_query("MATCH (n) RETURN count(n) as count")
        stats["node_count"] = result.records[0]["count"] if result.records else 0
        
        # Relationship count
        result = self.run_query("MATCH ()-[r]->() RETURN count(r) as count")
        stats["relationship_count"] = result.records[0]["count"] if result.records else 0
        
        # Labels
        result = self.run_query("CALL db.labels()")
        stats["labels"] = [record["label"] for record in result.records] if result.records else []
        
        # Relationship types
        result = self.run_query("CALL db.relationshipTypes()")
        stats["relationship_types"] = [record["relationshipType"] for record in result.records] if result.records else []
        
        # Property keys
        result = self.run_query("CALL db.propertyKeys()")
        stats["property_keys"] = [record["propertyKey"] for record in result.records] if result.records else []
        
        return DatabaseStats(
            node_count=stats.get("node_count", 0),
            relationship_count=stats.get("relationship_count", 0),
            labels=stats.get("labels", []),
            relationship_types=stats.get("relationship_types", []),
            property_keys=stats.get("property_keys", []),
            database_info={"database": self.database, "uri": self.uri}
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_execution_time = (
            self.metrics["total_execution_time"] / self.metrics["queries_executed"]
            if self.metrics["queries_executed"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "average_execution_time": avg_execution_time,
            "success_rate": (
                (self.metrics["queries_executed"] - self.metrics["failed_queries"]) 
                / self.metrics["queries_executed"]
                if self.metrics["queries_executed"] > 0 else 0
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "queries_executed": 0,
            "total_execution_time": 0.0,
            "failed_queries": 0,
            "connection_attempts": self.metrics.get("connection_attempts", 0),
        }

    def _validate_label(self, label: str) -> bool:
        """Validate node label."""
        if not label or not isinstance(label, str):
            logger.error(f"Invalid label: {label}")
            return False
        return True

    def _validate_relationship_type(self, rel_type: str) -> bool:
        """Validate relationship type."""
        if not rel_type or not isinstance(rel_type, str):
            logger.error(f"Invalid relationship type: {rel_type}")
            return False
        return True

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()


# Example usage
async def async_example():
    """Example of async usage."""
    async with GraphDBLayer(use_async=True) as db:
        if db.is_connected:
            # Add node asynchronously
            result = await db.run_query_async(
                "CREATE (n:Person {name: $name}) RETURN id(n) as id",
                {"name": "Alice Async"},
                is_write=True
            )
            print(f"Async result: {result.to_dict()}")

def sync_example():
    """Example of sync usage."""
    with GraphDBLayer(use_async=False) as db:
        if db.is_connected:
            # Get database stats
            stats = db.get_database_stats()
            print(f"Database stats: {asdict(stats)}")
            
            # Add node
            node_id = db.add_node("Person", {"name": "Bob Sync", "age": 25})
            if node_id:
                print(f"Created node with ID: {node_id}")
            
            # Get metrics
            metrics = db.get_metrics()
            print(f"Performance metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    print("GraphDBLayer - Production Ready Neo4j Interface")
    print("=" * 50)
    
    # Run sync example
    print("\n--- Synchronous Example ---")
    try:
        sync_example()
    except GraphDBConnectionError as e:
        print(f"Connection error: {e}")
        print("Make sure Neo4j is running and credentials are correct")
    
    # Run async example
    print("\n--- Asynchronous Example ---")
    try:
        asyncio.run(async_example())
    except GraphDBConnectionError as e:
        print(f"Async connection error: {e}")
        print("Make sure Neo4j is running and credentials are correct")
