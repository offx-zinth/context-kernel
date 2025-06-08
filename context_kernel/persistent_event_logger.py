# Core Infrastructure: Production-ready PostgreSQL event logger. (Marked as per review)
import psycopg2
import json
from psycopg2 import sql

class PersistentEventLogger:
    """
    A class to log events to a PostgreSQL database.
    """

    def __init__(self, dbname='event_log_db', user='logger_user', password='securepassword', host='localhost', port='5432'):
        """
        Initializes the PersistentEventLogger and connects to the database.

        Args:
            dbname (str): The name of the database.
            user (str): The username for database connection.
            password (str): The password for database connection.
            host (str): The host of the database server.
            port (str): The port of the database server.
        """
        self.conn = None
        self.cursor = None
        try:
            self.conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.cursor = self.conn.cursor()
            self._create_event_logs_table()
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            # Optionally, re-raise the error or handle it as per application's needs
            self.conn = None # Ensure conn is None if connection failed
            self.cursor = None

    def _create_event_logs_table(self):
        """
        Creates the 'event_logs' table if it doesn't already exist.
        """
        if not self.cursor:
            print("Cannot create table: database cursor not available.")
            return

        create_table_query = """
        CREATE TABLE IF NOT EXISTS event_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            event_type VARCHAR(255) NOT NULL,
            source_module VARCHAR(255),
            data JSONB
        );
        """
        try:
            self.cursor.execute(create_table_query)
            self.conn.commit()
        except psycopg2.Error as e:
            print(f"Error creating event_logs table: {e}")
            if self.conn:
                self.conn.rollback() # Rollback in case of error

    def log_event(self, event_type: str, source_module: str = None, data: dict = None):
        """
        Logs an event to the 'event_logs' table.

        Args:
            event_type (str): The type of the event.
            source_module (str, optional): The module or component that generated the event. Defaults to None.
            data (dict, optional): Additional data associated with the event, stored as JSON. Defaults to None.
        """
        if not self.conn or not self.cursor:
            print("Cannot log event: database connection not available.")
            return

        insert_query = sql.SQL("""
        INSERT INTO event_logs (event_type, source_module, data)
        VALUES (%s, %s, %s);
        """)
        
        # Serialize data to JSON string if it's a dictionary, otherwise pass as is (psycopg2 handles None for JSONB)
        json_data = json.dumps(data) if data is not None else None

        try:
            self.cursor.execute(insert_query, (event_type, source_module, json_data))
            self.conn.commit()
        except psycopg2.Error as e:
            print(f"Error logging event: {e}")
            if self.conn:
                self.conn.rollback()

    def close(self):
        """
        Closes the database cursor and connection.
        """
        if self.cursor:
            try:
                self.cursor.close()
            except psycopg2.Error as e:
                print(f"Error closing cursor: {e}")
        if self.conn:
            try:
                self.conn.close()
            except psycopg2.Error as e:
                print(f"Error closing connection: {e}")
        print("Database connection closed.")

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Ensure you have a PostgreSQL server running and configured as per these details
    # or modify them to match your setup.
    print("Initializing logger...")
    # Use different credentials for testing if needed, or ensure the defaults are set up
    logger = PersistentEventLogger(dbname='test_event_db', user='test_user', password='testpassword', host='localhost', port='5432')

    if logger.conn:
        print("Logger initialized successfully. Logging events...")
        logger.log_event(event_type='system_startup', source_module='main_app', data={'message': 'System initialized', 'version': '1.0'})
        logger.log_event(event_type='user_interaction', source_module='auth_module', data={'user_id': 123, 'action': 'login_attempt', 'status': 'success'})
        logger.log_event(event_type='data_processing', source_module='worker_process_alpha', data={'records_processed': 1500, 'status': 'completed'})
        logger.log_event(event_type='system_error', source_module='payment_gateway', data={'error_code': 503, 'message': 'Service unavailable'})
        
        print("Events logged. Closing logger...")
        logger.close()
    else:
        print("Failed to initialize logger. Please check database connection and credentials.")

    # Example with default credentials (ensure 'event_log_db' etc. are set up)
    # print("\nInitializing logger with default credentials...")
    # default_logger = PersistentEventLogger()
    # if default_logger.conn:
    #     print("Default logger initialized. Logging an event...")
    #     default_logger.log_event(event_type='test_event', source_module='default_test', data={'info': 'Testing with default setup'})
    #     default_logger.close()
    # else:
    #     print("Failed to initialize default logger.")
