# Context-Kernel (Memory & Retrieval System)

*Memory Flow:* Data flows in from all modules → Gets abstracted → Stored, linked, and retrieved to empower deeper cognition.

The Context-Kernel is the central memory system of the AI, responsible for storing, managing, and retrieving information across various time scales and levels of abstraction.

## Components

*   **Persistent Event Logging**:
    *   **Purpose**: To maintain a complete and immutable record of system activities.
    *   **Functionality**: Stores immutable records of every interaction (user inputs, system decisions) and intermediate module state.
    *   **Inputs**: Data from all modules.
    *   **Outputs**: Immutable event logs.

*   **Working Memory System**:
    *   **Purpose**: To manage transient, contextually relevant information.
    *   **Functionality**: Logs transient, high-context relevance notes with evolving metadata (e.g., current conversation topics, immediate task parameters).
    *   **Inputs**: Real-time data from active processes.
    *   **Outputs**: Accessible working memory store.

*   **Tiered Vector DBs**:
    *   **Purpose**: To store and retrieve information based on semantic similarity at different levels of abstraction.
    *   **Functionality**:
        1.  **Raw Thoughts**: Embeddings from unfiltered input.
        2.  **Chunk Summaries**: Summaries of recent reasoning or content.
        3.  **Executive Summary**: Condensed system-wide beliefs and trends.
        4.  **STM (Short-Term Memory)**: Cache for active conversational windows.
        5.  **LTM (Long-Term Memory)**: Retained domain knowledge and patterns.
    *   **Inputs**: Embeddings, text summaries, knowledge artifacts from various modules.
    *   **Outputs**: Searchable vector databases.

*   **Graph DB Layer**:
    *   **Purpose**: To maintain a semantic web of interconnected ideas and facts.
    *   **Functionality**: Maintains a semantic web of ideas, interlinking facts, summaries, beliefs, and inferences for high-precision search and relationship discovery.
    *   **Inputs**: Entities, relationships, concepts from processed data.
    *   **Outputs**: Queryable graph database.

*   **LLM Roles**:
    *   **Memory Accessor**:
        *   **Purpose**: To retrieve relevant information from memory.
        *   **Functionality**: Queries relevant data by context window, semantic embedding, or historical precedence.
        *   **Inputs**: Queries, contextual cues.
        *   **Outputs**: Retrieved data from memory tiers.
    *   **Summarizer & Updater**:
        *   **Purpose**: To maintain and update memory stores.
        *   **Functionality**: Continuously ingests new data, updates relevant memory tiers (e.g., creating chunk summaries, updating executive summaries), and flags contradictions or inconsistencies.
        *   **Inputs**: New data from modules, existing memory content.
        *   **Outputs**: Updated memory stores, contradiction alerts.
