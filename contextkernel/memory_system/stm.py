"""
Module for the Short-Term Memory (STM) system of the Context Kernel.

The STM manages a fast-access buffer of recent conversational turns for each session,
now primarily utilizing Redis for persistence and scalability if configured.
It continues to use Hugging Face Transformer models for on-the-fly summarization
and intent tagging of conversation content.
"""
import asyncio
import logging
# from collections import deque # Deque is no longer used if Redis is the primary store
from typing import Any, Dict, List, Optional # Deque removed from here as well
import json
import time

import redis.asyncio as redis # For Redis integration
from .ltm import LTM
from contextkernel.utils.config import NLPServiceConfig, RedisConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
# `pipeline` from transformers was not directly used. Models are used directly.
import torch

# Configure basic logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class STM:
    """
    Short-Term Memory (STM) system.
    Manages recent conversational turns, using Redis as the primary backend if configured.
    Provides functionalities like adding turns, retrieving recent turns, summarizing sessions,
    and flushing sessions to Long-Term Memory (LTM).
    Utilizes Hugging Face Transformer models for summarization and intent tagging.
    """

    DEFAULT_MAX_TURNS = 50
    DEFAULT_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
    DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli" # Zero-shot classification model

    def __init__(self,
                 ltm: LTM,
                 summarizer_nlp_config: NLPServiceConfig,
                 intent_tagging_nlp_config: NLPServiceConfig,
                 max_turns: int = DEFAULT_MAX_TURNS,
                 candidate_intents: Optional[List[str]] = None,
                 redis_config: Optional[RedisConfig] = None):
        """
        Initializes the STM system.

        Args:
            ltm: An instance of the LTM system for flushing sessions.
            summarizer_nlp_config: Configuration for the summarization NLP model.
            intent_tagging_nlp_config: Configuration for the intent tagging NLP model.
            max_turns: Maximum number of recent turns to keep in a session's STM buffer.
            candidate_intents: A list of candidate intent labels for classification.
            redis_config: (Optional) Configuration for connecting to a Redis instance.
                          If provided, Redis will be used for storing conversation turns.
                          If None, STM will operate in a degraded mode for conversation storage.
        """

        if not isinstance(ltm, LTM):
            logger.error("STM initialized with invalid LTM instance.")
            raise ValueError("STM requires a valid LTM instance.")

        self.ltm = ltm
        self.summarizer_nlp_config = summarizer_nlp_config
        self.intent_tagging_nlp_config = intent_tagging_nlp_config
        self.max_turns = max_turns
        self.redis_config = redis_config
        self.redis_client: Optional[redis.Redis] = None # Redis client instance

        self.candidate_intent_labels = candidate_intents or [
            "general inquiry", "problem report", "feature request",
            "positive feedback", "negative feedback", "transactional"
        ]

        # Configurable intent threshold from NLPServiceConfig
        _configured_threshold = getattr(self.intent_tagging_nlp_config, 'intent_threshold', None)
        if _configured_threshold is not None and 0.0 <= _configured_threshold <= 1.0:
            self.intent_classification_threshold = _configured_threshold
            logger.info(f"Using configured intent classification threshold: {self.intent_classification_threshold}")
        else:
            self.intent_classification_threshold = 0.7 # Default threshold
            if _configured_threshold is not None:
                 logger.warning(f"Invalid intent_threshold ({_configured_threshold}) in config. Must be between 0.0 and 1.0. Using default: {self.intent_classification_threshold}")
            else:
                 logger.info(f"Intent classification threshold not configured. Using default: {self.intent_classification_threshold}")

        # Load summarization model and tokenizer
        _summarizer_model_name = self.summarizer_nlp_config.model or self.DEFAULT_SUMMARIZER_MODEL
        try:
            logger.info(f"Loading summarization model: {_summarizer_model_name}")
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(_summarizer_model_name)
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(_summarizer_model_name)

            summarizer_device_name = getattr(self.summarizer_nlp_config, 'device', None)
            self.summarizer_device = torch.device(summarizer_device_name if summarizer_device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.summarizer_model.to(self.summarizer_device)
            logger.info(f"Summarization model '{_summarizer_model_name}' loaded on device '{self.summarizer_device}'.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load summarization model '{_summarizer_model_name}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load critical STM summarization model: {_summarizer_model_name}. STM cannot operate.") from e

        # Load intent tagging (zero-shot classification) model and tokenizer
        _intent_model_name = self.intent_tagging_nlp_config.model or self.DEFAULT_INTENT_MODEL
        try:
            logger.info(f"Loading intent classification model: {_intent_model_name}")
            self.intent_tokenizer = AutoTokenizer.from_pretrained(_intent_model_name)
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(_intent_model_name)

            intent_device_name = getattr(self.intent_tagging_nlp_config, 'device', None)
            self.intent_device = torch.device(intent_device_name if intent_device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.intent_model.to(self.intent_device)
            logger.info(f"Intent classification model '{_intent_model_name}' loaded on device '{self.intent_device}'.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load intent classification model '{_intent_model_name}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load critical STM intent model: {_intent_model_name}. STM cannot operate.") from e

        if not self.redis_config:
            logger.warning("RedisConfig not provided to STM. STM conversation history will not be persisted via Redis.")

        logger.info(f"STM initialized. Summarizer: {_summarizer_model_name} on {self.summarizer_device}, Intent Tagger: {_intent_model_name} on {self.intent_device}, Max Turns: {self.max_turns}. Redis configured: {bool(self.redis_config)}")

    def _get_redis_session_key(self, session_id: str) -> str:
        """Helper to generate a consistent Redis key for a session's conversation history."""
        return f"stm_session:{session_id}"

    async def boot(self):
        """
        Boots the STM system.
        - Attempts to connect to Redis if `redis_config` was provided.
        - NLP models are already loaded during `__init__`.
        """
        logger.info("STM booting up...")
        if self.redis_config and not self.redis_client: # If config provided and client not yet set up
            try:
                redis_kwargs = {
                    'host': self.redis_config.host,
                    'port': self.redis_config.port,
                    'db': self.redis_config.db
                }
                if self.redis_config.password:
                    redis_kwargs['password'] = self.redis_config.password.get_secret_value()

                self.redis_client = redis.Redis(**redis_kwargs)
                await self.redis_client.ping() # Verify connection
                logger.info(f"STM successfully connected to Redis at {self.redis_config.host}:{self.redis_config.port}, DB {self.redis_config.db}")
            except Exception as e:
                logger.error(f"STM failed to connect to Redis: {e}", exc_info=True)
                self.redis_client = None # Ensure client is None if connection failed
                # Depending on policy, could raise an error here to halt boot if Redis is critical.
        elif self.redis_client:
            logger.info("STM already has an active Redis client (e.g., from a previous boot attempt or manual setup).")
        else:
            logger.info("STM: No Redis configuration provided. Skipping Redis connection. Conversation history will not use Redis.")

        # Models are loaded in __init__. A failure there would have raised RuntimeError.
        logger.info("STM boot complete.")
        return True


    async def shutdown(self):
        """
        Shuts down the STM system:
        - Closes the Redis client connection if active.
        - Clears references to NLP models and tokenizers.
        """
        logger.info("STM shutting down...")
        if self.redis_client:
            try:
                await self.redis_client.close()
                # For redis.asyncio, close() manages the pool. Explicit pool disconnect might be needed for older versions or specific setups.
                if hasattr(self.redis_client, 'connection_pool'):
                    await self.redis_client.connection_pool.disconnect()
                logger.info("STM Redis client connection closed.")
            except Exception as e:
                logger.error(f"Error closing STM Redis client connection: {e}", exc_info=True)
            finally:
                self.redis_client = None # Clear client reference

        # Clear model and tokenizer references to allow garbage collection
        self.summarizer_model = None
        self.summarizer_tokenizer = None
        self.intent_model = None
        self.intent_tokenizer = None
        logger.info("STM models and tokenizers references cleared.")
        return True

    async def add_turn(self, session_id: str, turn_data: Dict[str, Any]) -> None:
        """
        Adds a new turn to the specified session's conversation history in Redis.
        The conversation is stored as a list, with `LPUSH` adding to the head (left)
        and `LTRIM` maintaining the list size to `max_turns`.

        Args:
            session_id: The unique identifier for the session.
            turn_data: A dictionary containing turn information (e.g., role, content).
                       A "timestamp" will be added if not present.
        """
        if not self.redis_client:
            logger.error("Redis client not available. Cannot add turn. Please check Redis configuration and STM boot status.")
            return

        if not session_id:
            logger.error("Session ID must be provided to add a turn.")
            return

        if not isinstance(turn_data, dict):
            logger.error(f"turn_data must be a dictionary, got {type(turn_data)} for session '{session_id}'.")
            return

        if "timestamp" not in turn_data: # Ensure every turn has a timestamp
            turn_data["timestamp"] = time.time()

        try:
            session_key = self._get_redis_session_key(session_id)
            serialized_turn_data = json.dumps(turn_data) # Serialize turn to JSON string

            # Use a Redis pipeline for atomic LPUSH and LTRIM operations
            async with self.redis_client.pipeline() as pipe:
                pipe.lpush(session_key, serialized_turn_data)
                pipe.ltrim(session_key, 0, self.max_turns - 1) # Keep the list to max_turns length
                await pipe.execute()

            logger.info(f"Added turn to session '{session_id}' in Redis. Key: '{session_key}'. Turn: {str(turn_data)[:150]}...")
        except Exception as e:
            logger.error(f"Error adding turn to Redis for session '{session_id}': {e}", exc_info=True)


    async def get_recent_turns(self, session_id: str, num_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves recent turns for a session from Redis.
        Turns are returned newest first (due to LPUSH/LRANGE 0 N-1).

        Args:
            session_id: The session identifier.
            num_turns: The number of most recent turns to retrieve.
                       If None, retrieves all turns up to `self.max_turns`.

        Returns:
            A list of turn dictionaries, newest first. Returns empty list if session
            is not found, Redis is unavailable, or an error occurs.
        """
        if not self.redis_client:
            logger.error("Redis client not available. Cannot get recent turns.")
            return []

        if not session_id:
            logger.warning("Session ID not provided for get_recent_turns.")
            return []

        session_key = self._get_redis_session_key(session_id)

        # Determine the range for LRANGE. LRANGE end index is inclusive.
        # To get `num_turns` newest items (from left of list): LRANGE key 0 num_turns-1
        # To get all items (up to max_turns, as list is trimmed): LRANGE key 0 max_turns-1
        start_index = 0
        end_idx = (num_turns - 1) if num_turns and num_turns > 0 else (self.max_turns - 1)

        try:
            serialized_turns = await self.redis_client.lrange(session_key, start_index, end_idx)

            deserialized_turns: List[Dict[str, Any]] = []
            for turn_bytes in serialized_turns: # Redis client typically returns bytes
                try:
                    turn_str = turn_bytes.decode('utf-8') # Decode bytes to string
                    deserialized_turns.append(json.loads(turn_str)) # Deserialize JSON string
                except json.JSONDecodeError as je:
                    logger.error(f"Error deserializing turn from Redis for session '{session_id}': {je}. Turn data: '{turn_bytes[:100]}'")
                except Exception as e_inner:
                    logger.error(f"Unexpected error deserializing turn: {e_inner}", exc_info=True)

            logger.info(f"Retrieved {len(deserialized_turns)} turns for session '{session_id}' from Redis. Requested: {num_turns}, Range: {start_index}-{end_idx}")
            return deserialized_turns
        except Exception as e:
            logger.error(f"Error retrieving turns from Redis for session '{session_id}': {e}", exc_info=True)
            return []

    async def get_full_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all turns for a specified session from Redis (up to `max_turns`).
        Turns are ordered newest first.
        """
        # `get_recent_turns` with num_turns=None or num_turns >= max_turns effectively fetches all.
        # Passing self.max_turns ensures we get up to the configured limit.
        return await self.get_recent_turns(session_id, num_turns=self.max_turns)


    async def _generate_summary_hf(self, session_id: str, turns: List[Dict[str, Any]]) -> str:
        """Helper method to generate summary using Hugging Face transformers."""
        if not self.summarizer_model or not self.summarizer_tokenizer:
            logger.warning(f"Summarization model/tokenizer not available for session '{session_id}'. Returning raw content if any.")
            return "\n".join([f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in turns]) if turns else "No content to summarize."

        logger.info(f"Generating summary for session '{session_id}' with {len(turns)} turns using {self.summarizer_model.name_or_path}.")
        if not turns:
            return "No conversation turns to summarize for this session."

        text_to_summarize = " ".join([str(turn.get("content", "")) for turn in turns if turn.get("content")])
        if not text_to_summarize.strip():
            return "No textual content in turns to summarize."

        loop = asyncio.get_running_loop()
        try:
            def _summarize_sync(): # Synchronous part to run in executor
                inputs = self.summarizer_tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
                inputs = inputs.to(self.summarizer_device) # Move tensors to the correct device
                summary_ids = self.summarizer_model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    num_beams=4,
                    max_length=150, # Consider making this configurable
                    early_stopping=True
                )
                summary_text = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return summary_text

            summary = await loop.run_in_executor(None, _summarize_sync)
            logger.info(f"Generated summary for session '{session_id}'. Length: {len(summary)}")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization for session {session_id}: {e}", exc_info=True)
            return f"Error during summarization: {e}"


    async def _extract_intents_hf(self, session_id: str, turns: List[Dict[str, Any]]) -> List[str]:
        """Helper method to extract intents using Hugging Face zero-shot classification."""
        if not self.intent_model or not self.intent_tokenizer:
            logger.warning(f"Intent model/tokenizer not available for session '{session_id}'. Returning empty list.")
            return []

        logger.info(f"Extracting intents for session '{session_id}' with {len(turns)} turns using {self.intent_model.name_or_path}.")
        if not turns:
            return []

        text_for_intent = turns[-1].get("content", "") # Focus on the last turn for intent
        if not text_for_intent.strip():
            return ["unknown_intent_empty_text"] # Return a specific label for empty text

        loop = asyncio.get_running_loop()
        try:
            def _classify_sync(): # Synchronous part
                inputs = self.intent_tokenizer(text_for_intent, self.candidate_intent_labels, return_tensors="pt", padding=True, truncation=True)
                inputs = inputs.to(self.intent_device) # Move tensors to the correct device
                with torch.no_grad():
                    outputs = self.intent_model(**inputs)
                    logits = outputs.logits

                probabilities = torch.softmax(logits, dim=1).squeeze()

                # Handle cases where probabilities might not be a list (e.g. single label)
                if probabilities.ndim == 0: # Single probability value
                    probabilities_list = [probabilities.item()]
                else:
                    probabilities_list = probabilities.tolist()

                identified_intents = []
                for i, prob in enumerate(probabilities_list):
                    if prob > self.intent_classification_threshold:
                        identified_intents.append(f"{self.candidate_intent_labels[i]} ({prob:.2f})")
                return identified_intents if identified_intents else ["general_discussion"] # Default if no intent passes threshold

            intents = await loop.run_in_executor(None, _classify_sync)
            logger.info(f"Identified intents for session '{session_id}': {intents}")
            return intents
        except Exception as e:
            logger.error(f"Error during intent extraction for session {session_id}: {e}", exc_info=True)
            return [f"error_extracting_intent: {e}"] # Return error message as an intent


    async def summarize_session(self, session_id: str) -> str:
        """
        Summarizes the conversation for a given session ID.
        Retrieves turns from Redis before generating the summary.
        """
        turns = await self.get_full_conversation(session_id)
        if not turns :
            logger.warning(f"Session '{session_id}' not found or empty for summarization.")
            return "Session not found or no content in STM to summarize."

        summary = await self._generate_summary_hf(session_id, turns)
        return summary

    async def flush_session_to_ltm(self, session_id: str, summarize: bool = True, clear_after_flush: bool = True) -> Dict[str, Any]:
        """
        Flushes a session's content (either full text or summary) to Long-Term Memory (LTM).
        Optionally clears the session from STM (Redis) after flushing.

        Args:
            session_id: The ID of the session to flush.
            summarize: If True, summarizes the session before flushing; otherwise, flushes full conversation text.
            clear_after_flush: If True, deletes the session from STM (Redis) after successful flush.

        Returns:
            A dictionary containing the status of the flush operation, LTM ID of the stored chunk,
            content type flushed, and number of original turns.
        """
        logger.info(f"Attempting to flush session '{session_id}' to LTM. Summarize: {summarize}, Clear after: {clear_after_flush}")

        turns = await self.get_full_conversation(session_id)
        if not turns:
            logger.warning(f"Session '{session_id}' not found in STM or is empty. Nothing to flush.")
            return {"status": "not_found_or_empty", "flushed_items_count": 0, "ltm_id": None}

        intents = await self._extract_intents_hf(session_id, turns)

        ltm_chunk_id_base = f"stm_session_{session_id}" # Base for LTM ID
        text_to_store: str

        # Prepare metadata for LTM storage
        first_turn_time = turns[0].get('timestamp') if turns else None # Newest turn due to LPUSH/LRANGE
        last_turn_time = turns[-1].get('timestamp') if turns else None  # Oldest turn in the retrieved set
        metadata: Dict[str, Any] = {
            "source_system": "STM",
            "original_session_id": session_id,
            "num_turns_in_session": len(turns), # Number of turns retrieved (up to max_turns)
            "identified_intents": intents,
            "session_first_turn_time_unix": first_turn_time, # Note: this is the newest turn's time
            "session_last_turn_time_unix": last_turn_time,   # Note: this is the oldest turn's time
            "stm_flush_time_unix": time.time()
        }

        if summarize:
            summary_text = await self._generate_summary_hf(session_id, turns)
            text_to_store = summary_text
            metadata["content_type"] = "session_summary"
            ltm_chunk_id = f"{ltm_chunk_id_base}_summary"
            logger.info(f"Using summary of session '{session_id}' for LTM storage.")
        else:
            # Format full conversation text for storage
            text_to_store = "\n".join([f"[{turn.get('timestamp')}] {turn.get('role', 'N/A')}: {turn.get('content', '')}" for turn in reversed(turns)]) # Reverse for chronological order
            metadata["content_type"] = "full_conversation_text"
            ltm_chunk_id = f"{ltm_chunk_id_base}_full"
            logger.info(f"Using full conversation text of session '{session_id}' for LTM storage.")

        metadata["text_char_length"] = len(text_to_store)
        embedding = await self.ltm.generate_embedding(text_to_store) # Generate embedding for the content

        # Store in LTM
        ltm_stored_id = await self.ltm.store_memory_chunk(
            chunk_id=ltm_chunk_id,
            text_content=text_to_store,
            embedding=embedding,
            metadata=metadata
        )

        # Clear from STM (Redis) if requested
        if clear_after_flush and self.redis_client:
            try:
                session_key = self._get_redis_session_key(session_id)
                await self.redis_client.delete(session_key)
                logger.info(f"Cleared session '{session_id}' (key: {session_key}) from Redis after flushing to LTM.")
            except Exception as e:
                logger.error(f"Error clearing session '{session_id}' from Redis: {e}", exc_info=True)
        elif clear_after_flush and not self.redis_client: # Log if clear requested but no Redis client
            logger.warning(f"Requested to clear session '{session_id}' but Redis client is not available.")


        result = {
            "status": "success",
            "ltm_id": ltm_stored_id,
            "flushed_content_type": metadata["content_type"],
            "num_original_turns": len(turns) # Number of turns processed from STM
        }
        logger.info(f"Successfully flushed session '{session_id}' to LTM. Result: {json.dumps(result)}")
        return result

    async def prefetch_session(self, session_id: str) -> None:
        """
        Placeholder for prefetching session data.
        For Redis-backed STM, this might involve checking key existence or warming local caches if any.
        Currently a no-op.
        """
        logger.info(f"Prefetch requested for session '{session_id}' (stubbed - no specific action for Redis-backed STM currently).")
        await asyncio.sleep(0.01) # Simulate async operation


async def main():
    """Example usage of the STM system, demonstrating Redis integration."""
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- STM Example Usage (with Redis Integration if Configured) ---")
    logger.warning("This example will download Hugging Face models if not already cached locally.")

    # Mock LTM for STM example
    class MockLTM:
        async def boot(self): logger.info("MockLTM booted."); return True
        async def shutdown(self): logger.info("MockLTM shutdown."); return True
        async def generate_embedding(self, text_content: str) -> List[float]:
            logger.debug(f"MockLTM.generate_embedding for: '{text_content[:30]}...'")
            return [0.1] * 384 # Dummy embedding of correct dimension
        async def store_memory_chunk(self, chunk_id: Optional[str], text_content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
            mem_id = chunk_id or str(uuid.uuid4())
            logger.info(f"MockLTM.store_memory_chunk: Storing ID '{mem_id}', text snip '{text_content[:50]}...', meta {metadata}")
            if not hasattr(self, '_stored_ltm_chunks'): self._stored_ltm_chunks = {}
            self._stored_ltm_chunks[mem_id] = {"text_content": text_content, "embedding": embedding, "metadata": metadata}
            return mem_id
        async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
            logger.info(f"MockLTM.get_memory_by_id for ID '{memory_id}'")
            if hasattr(self, '_stored_ltm_chunks'):
                return self._stored_ltm_chunks.get(memory_id)
            return None

    mock_ltm_instance = MockLTM()
    await mock_ltm_instance.boot()

    # NLP Configurations for STM's models
    summarizer_conf = NLPServiceConfig(provider="huggingface_transformer", model="sshleifer/distilbart-cnn-6-6")
    intent_tagger_conf = NLPServiceConfig(provider="huggingface_transformer", model="facebook/bart-large-mnli")
    candidate_intents_for_example = ["general question", "technical support", "sales inquiry", "feedback", "complaint"]

    # --- Redis Configuration for STM ---
    # This example assumes a Redis instance is running locally on default port.
    # Update host/port/db/password as needed for your Redis setup.
    example_redis_config = RedisConfig(host="localhost", port=6379, db=0)
    logger.info(f"Example main will attempt to connect to Redis at: {example_redis_config.host}:{example_redis_config.port}, DB: {example_redis_config.db}")
    logger.warning("If Redis is not available or configured incorrectly, STM operations relying on Redis will fail or log errors.")


    # Instantiate STM with real models and Redis config
    stm_system = None # Initialize to None for finally block
    try:
        stm_system = STM(
            ltm=mock_ltm_instance,
            summarizer_nlp_config=summarizer_conf,
            intent_tagging_nlp_config=intent_tagger_conf,
            max_turns=3, # Keep it small for easy verification
            candidate_intents=candidate_intents_for_example,
            redis_config=example_redis_config
        )
        if not await stm_system.boot(): # Boot STM (models loaded, Redis connection attempted)
             logger.error("STM system failed to boot. Check Redis connection and model paths.")
             if hasattr(mock_ltm_instance, 'shutdown'): await mock_ltm_instance.shutdown()
             return # Exit if boot fails

    except Exception as e:
        logger.error(f"Failed to initialize or boot STM with real models: {e}", exc_info=True)
        logger.error("Ensure you have an internet connection for model downloads on first run, "
                     "and that `transformers`, `torch`, `sentencepiece`, and `redis` (asyncio version) are installed.")
        if hasattr(mock_ltm_instance, 'shutdown'): await mock_ltm_instance.shutdown()
        return


    session_id_1 = "user123_chat_redis_001"
    # Clean up session from previous runs if any (for idempotency of example)
    if stm_system.redis_client: # Check if client is available
        await stm_system.redis_client.delete(stm_system._get_redis_session_key(session_id_1))
    else:
        logger.warning("STM Redis client not available in main example; cannot clean up previous session data.")


    # Add turns to the session (will be stored in Redis)
    ts = time.time()
    await stm_system.add_turn(session_id_1, {"role": "user", "content": "Hello there! I have a question about your product.", "timestamp": ts})
    await stm_system.add_turn(session_id_1, {"role": "assistant", "content": "Hi! I'm happy to help. What's your question?", "timestamp": ts + 1})
    await stm_system.add_turn(session_id_1, {"role": "user", "content": "I'm interested in the new Context Kernel feature. Can you explain it?", "timestamp": ts + 2})

    logger.info(f"\n--- After 3 turns (max_turns=3) for session '{session_id_1}' ---")
    turns_s1_initial = await stm_system.get_full_conversation(session_id_1)
    for i, turn in enumerate(turns_s1_initial): logger.info(f"Turn {i} (initial, newest first) for '{session_id_1}': {json.dumps(turn)}")
    assert len(turns_s1_initial) == 3
    # Due to LPUSH, the last item pushed ("I'm interested...") is at index 0.
    assert turns_s1_initial[0]['content'] == "I'm interested in the new Context Kernel feature. Can you explain it?"

    # Add a 4th turn; the oldest ("Hello there!") should be trimmed.
    await stm_system.add_turn(session_id_1, {"role": "assistant", "content": "Context Kernels are a new way to manage AI memory.", "timestamp": ts + 3})
    logger.info(f"\n--- After 4th turn (max_turns=3) for session '{session_id_1}' ---")
    turns_s1_updated = await stm_system.get_full_conversation(session_id_1)
    for i, turn in enumerate(turns_s1_updated): logger.info(f"Turn {i} (updated, newest first) for '{session_id_1}': {json.dumps(turn)}")
    assert len(turns_s1_updated) == 3 # Max turns constraint
    assert turns_s1_updated[0]['content'] == "Context Kernels are a new way to manage AI memory." # Newest
    assert turns_s1_updated[2]['content'] == "Hi! I'm happy to help. What's your question?" # Oldest remaining

    # Retrieve just the most recent 2 turns
    recent_2_turns_s1 = await stm_system.get_recent_turns(session_id_1, num_turns=2)
    logger.info(f"\nRecent 2 turns for session '{session_id_1}': {json.dumps(recent_2_turns_s1, indent=2)}")
    assert len(recent_2_turns_s1) == 2
    assert recent_2_turns_s1[0]['content'] == "Context Kernels are a new way to manage AI memory."

    # Test summarization (uses turns from Redis)
    logger.info(f"\n--- Generating summary for session '{session_id_1}' ---")
    summary_s1 = await stm_system.summarize_session(session_id_1)
    logger.info(f"Summary for session '{session_id_1}':\n{summary_s1}")
    assert summary_s1 and "Context Kernel" in summary_s1 or "Context Kernels" in summary_s1, "Summary does not seem to reflect content."

    # Test flushing session to LTM (summarized, clear from STM)
    logger.info(f"\n--- Flushing session '{session_id_1}' to LTM (summarized) ---")
    flush_status_s1_summarized = await stm_system.flush_session_to_ltm(session_id_1, summarize=True, clear_after_flush=True)
    logger.info(f"Flush status for session '{session_id_1}' (summarized): {json.dumps(flush_status_s1_summarized)}")
    assert flush_status_s1_summarized["status"] == "success"
    assert flush_status_s1_summarized["ltm_id"] is not None

    turns_s1_after_flush = await stm_system.get_full_conversation(session_id_1) # Should be empty from Redis now
    logger.info(f"Turns in session '{session_id_1}' after summarized flush and clear: {turns_s1_after_flush}")
    assert len(turns_s1_after_flush) == 0

    # Verify content in MockLTM
    if hasattr(mock_ltm_instance, '_stored_ltm_chunks'):
        ltm_entry_s1_data = mock_ltm_instance._stored_ltm_chunks.get(flush_status_s1_summarized["ltm_id"])
        logger.info(f"Data stored in MockLTM for {flush_status_s1_summarized['ltm_id']}: {json.dumps(ltm_entry_s1_data, indent=2)}")
        assert ltm_entry_s1_data is not None
        assert ltm_entry_s1_data["metadata"]["original_session_id"] == session_id_1
        # Check if one of the expected intents (or similar) is present
        assert any(intent_keyword in str(ltm_entry_s1_data["metadata"]["identified_intents"]).lower()
                   for intent_keyword in ["general question", "general inquiry"])


    # --- Another session for full text flush (no clear) ---
    session_id_2 = "user789_chat_redis_002"
    if stm_system.redis_client: # Cleanup for idempotency
        await stm_system.redis_client.delete(stm_system._get_redis_session_key(session_id_2))

    ts2 = time.time()
    await stm_system.add_turn(session_id_2, {"role": "user", "content": "I'm having an issue with product X.", "timestamp": ts2})
    await stm_system.add_turn(session_id_2, {"role": "assistant", "content": "I'm sorry. Describe the problem.", "timestamp": ts2 + 1})

    logger.info(f"\n--- Flushing session '{session_id_2}' to LTM (full text, no clear) ---")
    flush_status_s2_full = await stm_system.flush_session_to_ltm(session_id_2, summarize=False, clear_after_flush=False)
    logger.info(f"Flush status for session '{session_id_2}' (full text): {json.dumps(flush_status_s2_full)}")
    assert flush_status_s2_full["status"] == "success"

    turns_s2_after_flush_no_clear = await stm_system.get_full_conversation(session_id_2)
    logger.info(f"Turns in session '{session_id_2}' after full flush (no clear): {turns_s2_after_flush_no_clear}")
    assert len(turns_s2_after_flush_no_clear) == 2 # Should still be in Redis

    if hasattr(mock_ltm_instance, '_stored_ltm_chunks'):
        ltm_entry_s2_data = mock_ltm_instance._stored_ltm_chunks.get(flush_status_s2_full["ltm_id"])
        assert ltm_entry_s2_data is not None
        assert "problem report" in str(ltm_entry_s2_data["metadata"]["identified_intents"]).lower()
        assert "product X" in ltm_entry_s2_data["text_content"]

    # Final cleanup of example keys from Redis
    if stm_system.redis_client:
        logger.info("Cleaning up example session keys from Redis...")
        await stm_system.redis_client.delete(stm_system._get_redis_session_key(session_id_1))
        await stm_system.redis_client.delete(stm_system._get_redis_session_key(session_id_2))
        # empty_session_id was already deleted or never created effectively in Redis if that test ran.

    # Shutdown sequence
    if stm_system: await stm_system.shutdown()
    await mock_ltm_instance.shutdown()

    logger.info("--- STM Example Usage (with Redis Integration) Complete ---")

if __name__ == "__main__":
    # This main function uses real Hugging Face models for summarization and intent tagging.
    # It requires `transformers` and `torch` to be installed.
    # `redis` (asyncio version) is needed for Redis integration.
    # Example: pip install transformers torch sentencepiece redis[hiredis]
    # Models are downloaded on first use if not already cached by Hugging Face.
    # LTM is mocked to simplify this example. A Redis server should be running for this example.
    asyncio.run(main())
