# Example: Demonstrates integrated memory flow using Context Kernel components.
import uuid
import time # For potential delays or timestamping if needed beyond what components do

# Import all necessary classes from the context_kernel package
from persistent_event_logger import PersistentEventLogger
from working_memory_system import WorkingMemorySystem
from vector_db_manager import VectorDBManager # Assuming TIER_NAMES and models are accessible if needed
from graph_db_layer import GraphDBLayer
from memory_llm_roles import SummarizerUpdaterAgent, MemoryAccessorAgent, DEFAULT_EMBEDDING_SIZE

# qdrant_client.models might be needed if creating PointStructs directly in this script,
# but agents should encapsulate that. For now, not strictly needed here.
# from qdrant_client import models as qdrant_models


def run_showcase():
    """
    Runs the integrated memory flow showcase.
    """
    print("--- Context Kernel Showcase: Integrated Memory Flow ---")

    # Initialize components
    # These will try to connect to their respective databases using default credentials.
    # Ensure Postgres, Qdrant, and Neo4j are running and accessible.
    logger = None
    wms = None
    vector_db_manager = None
    graph_db = None
    summarizer_agent = None
    memory_accessor_agent = None

    try:
        print("\n--- Phase 0: Initializing Components ---")

        # 1. Persistent Event Logger
        print("Initializing PersistentEventLogger...")
        logger = PersistentEventLogger() # Uses defaults: dbname='event_log_db', user='logger_user', etc.
        if not logger.conn:
            print("ERROR: Failed to connect to PostgreSQL for event logging. Showcase cannot proceed fully.")
            # Decide if to exit or continue with logging disabled. For showcase, better to highlight.
            # return # Or raise an error
        else:
            print("PersistentEventLogger initialized successfully.")

        # 2. Working Memory System
        print("Initializing WorkingMemorySystem...")
        wms = WorkingMemorySystem()
        print("WorkingMemorySystem initialized successfully.")

        # 3. Vector Database Manager
        print(f"Initializing VectorDBManager (vector size: {DEFAULT_EMBEDDING_SIZE})...")
        # Ensure DEFAULT_EMBEDDING_SIZE matches what agent placeholders produce
        vector_db_manager = VectorDBManager(vector_size=DEFAULT_EMBEDDING_SIZE)
        if not vector_db_manager.client:
            print("ERROR: Failed to connect to Qdrant. Showcase cannot proceed fully.")
            # return
        else:
            print("VectorDBManager initialized successfully and collections ensured.")

        # 4. Graph Database Layer
        print("Initializing GraphDBLayer...")
        graph_db = GraphDBLayer() # Uses defaults: bolt://localhost:7687, neo4j/password
        if not graph_db.driver:
            print("ERROR: Failed to connect to Neo4j. Showcase cannot proceed fully.")
            # return
        else:
            print("GraphDBLayer initialized successfully.")

        # 5. SummarizerUpdaterAgent ("Librarian")
        print("Initializing SummarizerUpdaterAgent...")
        summarizer_agent = SummarizerUpdaterAgent(
            vector_db_manager=vector_db_manager,
            graph_db_layer=graph_db
            # embedding_model_name will use its default
        )
        print("SummarizerUpdaterAgent initialized successfully.")

        # 6. MemoryAccessorAgent ("Seeker")
        print("Initializing MemoryAccessorAgent...")
        memory_accessor_agent = MemoryAccessorAgent(
            vector_db_manager=vector_db_manager,
            graph_db_layer=graph_db,
            working_memory=wms
            # embedding_model_name will use its default
        )
        print("MemoryAccessorAgent initialized successfully.")
        
        print("\n--- All components initialized. Starting memory flow simulation. ---")

        # --- Stage 1: Raw Input & Initial Processing ---
        print("\n--- Stage 1: Raw Input & Initial Processing ---")
        raw_text_input_1 = "Researchers today announced a breakthrough in quantum computing using novel entanglement algorithms. This could revolutionize data processing speeds."
        print(f"Sample Raw Input 1: \"{raw_text_input_1}\"")

        if logger and logger.conn:
            logger.log_event(
                event_type="raw_input_received",
                source_module="showcase_script",
                data={"text_snippet": raw_text_input_1[:100]} # Log snippet
            )
            print("Logged 'raw_input_received' event.")

        # --- Stage 2: Abstraction & Storage (SummarizerUpdaterAgent) ---
        print("\n--- Stage 2: Abstraction & Storage by SummarizerUpdaterAgent ---")
        if summarizer_agent:
            process_results_1 = summarizer_agent.process_input(
                input_text=raw_text_input_1,
                source_module="showcase_ingestion_1",
                store_raw=True,
                create_summary=True,
                store_in_graph=True
            )
            print(f"Processing Results 1: {process_results_1}")
            if logger and logger.conn:
                logger.log_event(
                    event_type="input_processed",
                    source_module="SummarizerUpdaterAgent",
                    data={"input_snippet": raw_text_input_1[:100], "results": process_results_1}
                )
                print("Logged 'input_processed' event.")
        else:
            print("SummarizerAgent not available, skipping abstraction and storage.")

        # --- Stage 3: Update Working Memory ---
        print("\n--- Stage 3: Update Working Memory ---")
        if wms:
            wms_note_id_1 = wms.add_or_update_note(
                content=f"Processed input about: {raw_text_input_1[:50]}...",
                origin="showcase_script_interaction_1",
                relevance_score=0.9
            )
            print(f"Added note to Working Memory with ID: {wms_note_id_1}")
            if logger and logger.conn:
                logger.log_event(event_type="wms_updated", source_module="showcase_script", data={"note_id": wms_note_id_1, "action": "add_note"})
        else:
            print("WorkingMemorySystem not available, skipping WMS update.")


        # --- Stage 4: Retrieval (MemoryAccessorAgent) ---
        print("\n--- Stage 4: Retrieval by MemoryAccessorAgent ---")
        query_1 = "Tell me about recent quantum computing breakthroughs and novel algorithms."
        print(f"Sample Query 1: \"{query_1}\"")

        if memory_accessor_agent:
            retrieved_memories_1 = memory_accessor_agent.fetch_memory(
                query_text=query_1,
                search_vector_dbs=["RawThoughtsDB", "ChunkSummaryDB"], # Be specific for showcase
                search_graph=True,
                search_working_memory=True,
                top_k_vector=3 # Limit results for readability
            )
            print(f"Retrieved {len(retrieved_memories_1)} memory packets for Query 1:")
            for i, packet in enumerate(retrieved_memories_1):
                content_display = ""
                if isinstance(packet.get('content'), dict):
                    content_display = str({k: (str(v)[:70] + '...' if len(str(v)) > 70 else str(v)) for k,v in packet['content'].items()})
                else:
                    content_display = str(packet.get('content'))[:100] + ('...' if len(str(packet.get('content', ''))) > 100 else '')

                print(f"  {i+1}. Source: {packet['source']}, Score: {packet.get('score', 'N/A')}, "
                      f"Content: {content_display}, Explanation: {packet.get('relevance_explanation')}")

            if logger and logger.conn:
                logger.log_event(
                    event_type="memory_retrieved",
                    source_module="MemoryAccessorAgent",
                    data={"query": query_1, "num_retrieved": len(retrieved_memories_1)}
                )
                print("Logged 'memory_retrieved' event.")
        else:
            print("MemoryAccessorAgent not available, skipping retrieval.")
            retrieved_memories_1 = []


        # --- Stage 5: Simulate another interaction to show memory evolution ---
        print("\n--- Stage 5: Second Interaction & Memory Evolution ---")
        raw_text_input_2 = "A new paper details how these quantum algorithms could also impact cryptography within five years."
        print(f"Sample Raw Input 2: \"{raw_text_input_2}\"")
        if logger and logger.conn:
             logger.log_event(event_type="raw_input_received", source_module="showcase_script", data={"text_snippet": raw_text_input_2[:100]})

        related_graph_node_id = None
        if process_results_1 and process_results_1.get("graph_node_id"):
            related_graph_node_id = process_results_1["graph_node_id"]
            print(f"This second input will be related to graph node ID: {related_graph_node_id} from the first input.")

        if summarizer_agent:
            process_results_2 = summarizer_agent.process_input(
                input_text=raw_text_input_2,
                source_module="showcase_ingestion_2",
                existing_node_id=related_graph_node_id, # Link to previous graph node
                store_raw=True, create_summary=True, store_in_graph=True
            )
            print(f"Processing Results 2: {process_results_2}")
            if logger and logger.conn:
                logger.log_event(event_type="input_processed", source_module="SummarizerUpdaterAgent", data={"input_snippet": raw_text_input_2[:100], "results": process_results_2})
        else:
            print("SummarizerAgent not available.")

        if wms:
            wms_note_id_2 = wms.add_or_update_note(
                content=f"Processed input about: {raw_text_input_2[:50]}... (related to quantum cryptography)",
                origin="showcase_script_interaction_2",
                relevance_score=0.95 # Higher relevance as it's more specific or recent
            )
            print(f"Added note to Working Memory with ID: {wms_note_id_2}")

        # Re-query with MemoryAccessorAgent
        query_2 = "What are the implications of new quantum algorithms on cryptography?"
        print(f"\nSample Query 2 (after second input): \"{query_2}\"")
        if memory_accessor_agent:
            retrieved_memories_2 = memory_accessor_agent.fetch_memory(
                query_text=query_2,
                search_vector_dbs=["RawThoughtsDB", "ChunkSummaryDB"],
                search_graph=True,
                search_working_memory=True,
                top_k_vector=3
            )
            print(f"Retrieved {len(retrieved_memories_2)} memory packets for Query 2:")
            for i, packet in enumerate(retrieved_memories_2):
                content_display = ""
                if isinstance(packet.get('content'), dict):
                    content_display = str({k: (str(v)[:70] + '...' if len(str(v)) > 70 else str(v)) for k,v in packet['content'].items()})
                else:
                    content_display = str(packet.get('content'))[:100] + ('...' if len(str(packet.get('content', ''))) > 100 else '')
                print(f"  {i+1}. Source: {packet['source']}, Score: {packet.get('score', 'N/A')}, "
                      f"Content: {content_display}, Explanation: {packet.get('relevance_explanation')}")

            if logger and logger.conn:
                logger.log_event(event_type="memory_retrieved", source_module="MemoryAccessorAgent", data={"query": query_2, "num_retrieved": len(retrieved_memories_2)})

        print("\n--- Showcase Flow Complete ---")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED DURING THE SHOWCASE ---")
        print(f"Error: {e}")
        if logger and logger.conn: # Try to log the error itself
            logger.log_event(event_type="showcase_error", source_module="showcase_script", data={"error_message": str(e)})
    finally:
        print("\n--- Phase Omega: Closing Connections ---")
        if logger and logger.conn:
            print("Closing PersistentEventLogger connection...")
            logger.close()
        if vector_db_manager and vector_db_manager.client:
            print("Closing VectorDBManager (Qdrant client) connection...")
            vector_db_manager.close() # Assuming VectorDBManager has a close method for its client
        if graph_db and graph_db.driver:
            print("Closing GraphDBLayer (Neo4j driver) connection...")
            graph_db.close()
        
        # WorkingMemorySystem is in-memory, no explicit close needed unless it held external resources.
        # Agents also don't own connections, they use the managers.

        print("--- Connections closed. Showcase finished. ---")


if __name__ == "__main__":
    # This script assumes that dependent services (Postgres, Qdrant, Neo4j) are running
    # with default configurations as set in their respective manager classes.
    # The agents use placeholder functions for actual LLM calls and embeddings.
    run_showcase()
