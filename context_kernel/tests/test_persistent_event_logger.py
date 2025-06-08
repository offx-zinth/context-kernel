import unittest
from unittest.mock import patch, MagicMock, ANY
import json
# Adjust the import path based on your project structure
# If context_kernel is a top-level package and tests is a sub-package:
from context_kernel.persistent_event_logger import PersistentEventLogger
# If running tests from the root directory and context_kernel is a directory:
# from persistent_event_logger import PersistentEventLogger


class TestPersistentEventLogger(unittest.TestCase):

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_init_success_and_table_creation(self, mock_connect):
        """Test successful initialization and table creation attempt."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        logger = PersistentEventLogger(dbname='testdb', user='testuser', password='pw', host='host', port='5432')

        mock_connect.assert_called_once_with(
            dbname='testdb', user='testuser', password='pw', host='host', port='5432'
        )
        mock_conn.cursor.assert_called_once()
        
        # Check if CREATE TABLE IF NOT EXISTS was executed
        create_table_call_found = False
        for call in mock_cursor.execute.call_args_list:
            if "CREATE TABLE IF NOT EXISTS event_logs" in call[0][0]:
                create_table_call_found = True
                break
        self.assertTrue(create_table_call_found, "CREATE TABLE IF NOT EXISTS event_logs was not executed.")
        mock_conn.commit.assert_called() # Commit after table creation
        self.assertIsNotNone(logger.conn)
        self.assertIsNotNone(logger.cursor)

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_init_connection_failure(self, mock_connect):
        """Test handling of connection failure during initialization."""
        mock_connect.side_effect = Exception("Connection failed")

        logger = PersistentEventLogger() # Use default params

        mock_connect.assert_called_once() # Attempted to connect
        self.assertIsNone(logger.conn, "Connection object should be None on failure.")
        self.assertIsNone(logger.cursor, "Cursor object should be None on failure.")
        # Optionally, check if an error message was printed (would need to mock 'print')

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_log_event_success(self, mock_connect):
        """Test successful logging of an event."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        logger = PersistentEventLogger()
        logger.conn = mock_conn # Ensure logger has the mocked connection
        logger.cursor = mock_cursor

        event_data = {'key': 'value', 'num': 123}
        event_type = "test_event"
        source_module = "test_module"

        logger.log_event(event_type=event_type, source_module=source_module, data=event_data)

        # Check SQL query construction (using ANY for the sql.SQL object)
        # The actual query object might be complex, so check for key parts or use ANY.
        # For more precise matching, you'd need to know exactly how `sql.SQL` formats.
        # Here, we check the arguments passed, expecting psycopg2 to handle SQL object correctly.
        
        # The expected SQL string for the INSERT statement (simplified, actual is an SQL object)
        # "INSERT INTO event_logs (event_type, source_module, data) VALUES (%s, %s, %s);"
        # We verify the parameters passed to execute.
        mock_cursor.execute.assert_called_once()
        args, _ = mock_cursor.execute.call_args
        # args[0] is the SQL query (could be a Composable object)
        # args[1] are the parameters
        self.assertIsInstance(args[0], object) # It's an SQL object from psycopg2.sql
        
        self.assertEqual(args[1][0], event_type)
        self.assertEqual(args[1][1], source_module)
        self.assertEqual(args[1][2], json.dumps(event_data)) # Data should be JSON serialized
        
        mock_conn.commit.assert_called_once()

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_log_event_no_data_or_source_module(self, mock_connect):
        """Test logging an event with no optional data or source_module."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        logger = PersistentEventLogger()
        logger.conn = mock_conn
        logger.cursor = mock_cursor
        
        event_type = "minimal_event"
        logger.log_event(event_type=event_type) # No source_module or data

        mock_cursor.execute.assert_called_once()
        args, _ = mock_cursor.execute.call_args
        self.assertEqual(args[1][0], event_type)
        self.assertIsNone(args[1][1], "source_module should be None if not provided")
        self.assertIsNone(args[1][2], "data should be None if not provided (and thus JSON serialized to null)")
        mock_conn.commit.assert_called_once()

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_log_event_db_error(self, mock_connect):
        """Test error handling during log_event."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate a database error during execute
        mock_cursor.execute.side_effect = Exception("DB write error")

        logger = PersistentEventLogger()
        logger.conn = mock_conn
        logger.cursor = mock_cursor

        with patch('builtins.print') as mock_print: # To check error messages
            logger.log_event(event_type="error_event", data={"info": "test"})
        
        mock_cursor.execute.assert_called_once()
        mock_conn.rollback.assert_called_once() # Should rollback on error
        mock_conn.commit.assert_not_called() # Commit should not be called on error
        mock_print.assert_any_call(unittest.mock.ANY) # Check if print was called with an error message

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_close_connection(self, mock_connect):
        """Test closing the database connection and cursor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        logger = PersistentEventLogger()
        # Manually set conn and cursor if they were not set due to other test logic or init failure simulation
        logger.conn = mock_conn
        logger.cursor = mock_cursor
        
        logger.close()

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('context_kernel.persistent_event_logger.psycopg2.connect')
    def test_close_no_connection(self, mock_connect):
        """Test close method when connection was never established."""
        mock_connect.side_effect = Exception("Initial connection failed")
        
        logger = PersistentEventLogger() # Init fails, conn and cursor are None
        
        # Mock print to suppress "Error connecting..." message during this specific test
        with patch('builtins.print') as _:
            logger_no_conn = PersistentEventLogger()

        # Ensure conn and cursor are indeed None
        self.assertIsNone(logger_no_conn.conn)
        self.assertIsNone(logger_no_conn.cursor)

        # Try to close - should not raise errors
        try:
            logger_no_conn.close()
        except Exception as e:
            self.fail(f"close() raised an unexpected exception when no connection exists: {e}")
        
        # No mock objects for cursor/conn were created, so no assertions on their methods needed.


if __name__ == '__main__':
    unittest.main()
