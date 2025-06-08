import unittest
from unittest.mock import patch, MagicMock, ANY
from context_kernel.graph_db_layer import GraphDBLayer
from neo4j import exceptions as neo4j_exceptions # For simulating errors


class TestGraphDBLayer(unittest.TestCase):

    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_init_success(self, mock_neo4j_driver_method):
        """Test successful driver initialization and connectivity verification."""
        mock_driver_instance = MagicMock()
        mock_neo4j_driver_method.return_value = mock_driver_instance

        uri = "bolt://testlocalhost:7687"
        user = "testuser"
        password = "testpassword"
        
        gdb = GraphDBLayer(uri=uri, user=user, password=password)

        mock_neo4j_driver_method.assert_called_once_with(uri, auth=(user, password))
        self.assertEqual(gdb.driver, mock_driver_instance)
        mock_driver_instance.verify_connectivity.assert_called_once() # Check if connection is verified

    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_init_auth_error(self, mock_neo4j_driver_method):
        """Test initialization with Neo4j authentication failure."""
        mock_driver_instance = MagicMock()
        mock_neo4j_driver_method.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.side_effect = neo4j_exceptions.AuthError("Authentication failed")

        with patch('builtins.print') as mock_print: # Suppress print
            gdb = GraphDBLayer()
        
        self.assertIsNone(gdb.driver) # Driver should be None on auth failure
        mock_print.assert_any_call(unittest.mock.ANY) # Error message printed

    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_init_service_unavailable(self, mock_neo4j_driver_method):
        """Test initialization when Neo4j service is unavailable."""
        mock_driver_instance = MagicMock()
        mock_neo4j_driver_method.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.side_effect = neo4j_exceptions.ServiceUnavailable("Service down")
        
        with patch('builtins.print') as mock_print:
            gdb = GraphDBLayer()

        self.assertIsNone(gdb.driver)
        mock_print.assert_any_call(unittest.mock.ANY)

    def common_query_execution_setup(self, mock_neo4j_driver_method):
        mock_driver_instance = MagicMock()
        mock_session_instance = MagicMock()
        mock_transaction_instance = MagicMock() # For older style tx.run, or if begin_transaction is used
        mock_result_instance = MagicMock()

        mock_neo4j_driver_method.return_value = mock_driver_instance
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session_instance # For 'with ... as session:'

        # Setup for session.write_transaction or session.read_transaction
        # These methods take a function (work) and then args for that function.
        # The 'work' function itself gets a transaction object.
        def mock_transaction_runner(transaction_function, query, params):
            # transaction_function is the (tx, q, p) -> ... lambda defined in _execute_query
            # It expects a transaction object (tx)
            mock_tx = MagicMock() # This is the 'tx' object passed to the lambda
            mock_tx.run.return_value = mock_result_instance # tx.run() returns results
            # Simulate collecting results as a list of records
            # If the query returns a single value (e.g. id), it's often like [{'node_id': 123}]
            # For simplicity, let's assume the lambda returns what tx.run returns,
            # and _execute_query processes it into a list of records.
            # So, the lambda would return mock_result_instance, which is then iterated.
            # Let's mock what the lambda returns after iterating.
            return [record for record in mock_result_instance]


        mock_session_instance.write_transaction.side_effect = mock_transaction_runner
        mock_session_instance.read_transaction.side_effect = mock_transaction_runner
        
        gdb = GraphDBLayer()
        gdb.driver = mock_driver_instance # Manually set driver if __init__ is not fully mocked here
        return gdb, mock_driver_instance, mock_session_instance, mock_transaction_instance, mock_result_instance


    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_add_node_success(self, mock_neo4j_driver_method):
        """Test successful node addition."""
        gdb, _, mock_session, _, mock_result = self.common_query_execution_setup(mock_neo4j_driver_method)
        
        # Simulate result from 'RETURN id(n) AS node_id'
        mock_record = MagicMock()
        mock_record.__getitem__.side_effect = lambda key: 123 if key == 'node_id' else None
        mock_record.get.side_effect = lambda key, default=None: 123 if key == 'node_id' else default # if .get is used
        type(mock_record).keys = lambda self_mock: ['node_id'] # Make 'in' operator work
        
        mock_result.__iter__.return_value = [mock_record] # tx.run() result is iterable

        label = "TestConcept"
        properties = {"name": "AI", "desc": "Artificial Intelligence"}
        
        node_id = gdb.add_node(label, properties)

        self.assertEqual(node_id, 123)
        # Check that write_transaction was called, and implicitly tx.run inside it
        # The actual query string is constructed dynamically.
        # We check that write_transaction was called with a function, the query, and params.
        # args[0] is the transaction_function, args[1] is the query, args[2] is params
        mock_session.write_transaction.assert_called_once()
        call_args = mock_session.write_transaction.call_args[0]
        called_query = call_args[1]
        called_params = call_args[2]

        self.assertIn(f"CREATE (n:{label} $props) RETURN id(n) AS node_id", called_query)
        self.assertEqual(called_params, {"props": properties})


    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_add_relationship_success(self, mock_neo4j_driver_method):
        """Test successful relationship addition."""
        gdb, _, mock_session, _, mock_result = self.common_query_execution_setup(mock_neo4j_driver_method)

        mock_record = MagicMock()
        mock_record.__getitem__.side_effect = lambda key: "RELATED_TO" if key == 'rel_type' else None
        type(mock_record).keys = lambda self_mock: ['rel_type']
        mock_result.__iter__.return_value = [mock_record]

        rel_type_str = gdb.add_relationship(1, 2, "RELATED_TO", {"since": "2023"})
        
        self.assertEqual(rel_type_str, "RELATED_TO")
        mock_session.write_transaction.assert_called_once()
        call_args = mock_session.write_transaction.call_args[0]
        called_query = call_args[1] # The Cypher query string
        called_params = call_args[2] # The parameters dictionary

        self.assertIn("MATCH (a), (b)", called_query)
        self.assertIn("WHERE id(a) = $start_id AND id(b) = $end_id", called_query)
        self.assertIn("CREATE (a)-[r:RELATED_TO $props]->(b)", called_query)
        self.assertIn("RETURN type(r) AS rel_type", called_query)
        self.assertEqual(called_params, {"start_id": 1, "end_id": 2, "props": {"since": "2023"}})

    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_update_node_properties_success(self, mock_neo4j_driver_method):
        """Test successful node property update."""
        gdb, _, mock_session, _, mock_result = self.common_query_execution_setup(mock_neo4j_driver_method)
        
        # Simulate that the node was found and updated, so a result is returned
        mock_result.__iter__.return_value = [MagicMock()] # Non-empty list means success

        props_to_update = {"status": "verified"}
        success = gdb.update_node_properties(123, props_to_update)

        self.assertTrue(success)
        mock_session.write_transaction.assert_called_once()
        call_args = mock_session.write_transaction.call_args[0]
        called_query = call_args[1]
        called_params = call_args[2]
        
        self.assertIn("MATCH (n)", called_query)
        self.assertIn("WHERE id(n) = $node_id", called_query)
        self.assertIn("SET n += $properties", called_query)
        self.assertIn("RETURN n", called_query)
        self.assertEqual(called_params, {"node_id": 123, "properties": props_to_update})


    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_query_graph_success(self, mock_neo4j_driver_method):
        """Test successful execution of a read query."""
        gdb, _, mock_session, _, mock_result = self.common_query_execution_setup(mock_neo4j_driver_method)
        
        expected_records = [{"name": "Alice"}, {"name": "Bob"}]
        mock_result.__iter__.return_value = expected_records # Simulate tx.run() returning records

        query = "MATCH (p:Person) RETURN p.name AS name"
        params = {"limit": 10}
        
        records = gdb.query_graph(query, params)

        self.assertEqual(records, expected_records)
        # query_graph uses read_transaction
        mock_session.read_transaction.assert_called_once()
        call_args = mock_session.read_transaction.call_args[0]
        self.assertEqual(call_args[1], query) # Query string
        self.assertEqual(call_args[2], params) # Parameters


    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_execute_query_handles_cypher_syntax_error(self, mock_neo4j_driver_method):
        """Test _execute_query handling of CypherSyntaxError."""
        gdb, _, mock_session, _, _ = self.common_query_execution_setup(mock_neo4j_driver_method)
        
        # Simulate a CypherSyntaxError being raised by tx.run()
        mock_session.write_transaction.side_effect = neo4j_exceptions.CypherSyntaxError("Invalid query")

        with patch('builtins.print') as mock_print:
            result = gdb._execute_query("INVALID QUERY", {}, is_write=True)
        
        self.assertIsNone(result)
        mock_print.assert_any_call(unittest.mock.ANY) # Error message was printed


    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_close_driver(self, mock_neo4j_driver_method):
        """Test closing the Neo4j driver."""
        mock_driver_instance = MagicMock()
        mock_neo4j_driver_method.return_value = mock_driver_instance
        
        gdb = GraphDBLayer() # Assume init is successful and sets self.driver
        gdb.driver = mock_driver_instance # Ensure driver is set for this test

        gdb.close()
        mock_driver_instance.close.assert_called_once()

    @patch('context_kernel.graph_db_layer.GraphDatabase.driver')
    def test_close_no_driver(self, mock_neo4j_driver_method):
        """Test close method when driver is None."""
        # Simulate driver init failure
        mock_neo4j_driver_method.side_effect = neo4j_exceptions.ServiceUnavailable("Service down")
        with patch('builtins.print'): # Suppress error print during init
            gdb = GraphDBLayer() 
        
        self.assertIsNone(gdb.driver)
        try:
            gdb.close() # Should not raise an error
        except Exception as e:
            self.fail(f"gdb.close() raised an exception with no driver: {e}")


if __name__ == '__main__':
    unittest.main()
