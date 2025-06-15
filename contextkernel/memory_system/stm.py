import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Deque
import json # For formatting dicts in logs/summaries
import time # For adding timestamps if not present

import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Deque
import json # For formatting dicts in logs/summaries
import time # For adding timestamps if not present

from .ltm import LTM
from contextkernel.utils.config import NLPServiceConfig, RedisConfig # Added RedisConfig for completeness if STM were Redis-backed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch # Often a peer dependency for transformers

# Configure basic logging. If run standalone, this basicConfig will apply.
# Otherwise, it's assumed the importing module (like MemoryKernel) configures logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class STM:
    """
    Short-Term Memory (STM) system.
    Manages a fast-access buffer of recent conversational turns or items using a rolling window.
    Uses Hugging Face Transformer models for summarization and intent tagging.
    The conversation buffer itself is currently in-memory.
    """

    DEFAULT_MAX_TURNS = 50
    DEFAULT_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
    DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli" # Zero-shot classification model

    def __init__(self,
                 ltm: LTM,
                 summarizer_nlp_config: NLPServiceConfig, # .model = HF model name for summarization
                 intent_tagging_nlp_config: NLPServiceConfig, # .model = HF model name for zero-shot intent
                 max_turns: int = DEFAULT_MAX_TURNS,
                 candidate_intents: Optional[List[str]] = None):

        if not isinstance(ltm, LTM):
            logger.error("STM initialized with invalid LTM instance.")
            raise ValueError("STM requires a valid LTM instance.")

        self.ltm = ltm
        self.summarizer_nlp_config = summarizer_nlp_config
        self.intent_tagging_nlp_config = intent_tagging_nlp_config
        self.max_turns = max_turns

        self.candidate_intent_labels = candidate_intents or [
            "general inquiry", "problem report", "feature request",
            "positive feedback", "negative feedback", "transactional"
        ]

        # Load summarization model and tokenizer
        _summarizer_model_name = self.summarizer_nlp_config.model or self.DEFAULT_SUMMARIZER_MODEL
        try:
            logger.info(f"Loading summarization model: {_summarizer_model_name}")
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(_summarizer_model_name)
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(_summarizer_model_name)
            logger.info("Summarization model and tokenizer loaded.")
        except Exception as e:
            logger.error(f"Error loading summarization model '{_summarizer_model_name}': {e}", exc_info=True)
            # Fallback or raise - for now, log and continue, methods will fail if model is None
            self.summarizer_tokenizer = None
            self.summarizer_model = None
            # Consider raising RuntimeError here if summarization is critical

        # Load intent tagging (zero-shot classification) model and tokenizer
        _intent_model_name = self.intent_tagging_nlp_config.model or self.DEFAULT_INTENT_MODEL
        try:
            logger.info(f"Loading intent classification model: {_intent_model_name}")
            self.intent_tokenizer = AutoTokenizer.from_pretrained(_intent_model_name)
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(_intent_model_name)
            # For zero-shot, we can also use the pipeline for convenience
            # self.intent_classifier_pipeline = pipeline("zero-shot-classification", model=_intent_model_name, tokenizer=_intent_model_name)
            logger.info("Intent classification model and tokenizer loaded.")
        except Exception as e:
            logger.error(f"Error loading intent model '{_intent_model_name}': {e}", exc_info=True)
            self.intent_tokenizer = None
            self.intent_model = None
            # self.intent_classifier_pipeline = None

        self._conversations_stub: Dict[str, Deque[Dict[str, Any]]] = {}
        logger.info(f"STM initialized. Summarizer: {_summarizer_model_name}, Intent Tagger: {_intent_model_name}, Max Turns: {self.max_turns}.")

    async def boot(self):
        """
        Models are loaded in __init__. This method can be used for other setup if needed.
        """
        logger.info("STM booting up...")
        # Models are loaded in __init__. If any failed, relevant functionalities will be impaired.
        if not self.summarizer_model:
            logger.warning("Summarization model not available for STM.")
        if not self.intent_model:
            logger.warning("Intent tagging model not available for STM.")
        await asyncio.sleep(0.01) # Simulate any minor setup
        logger.info("STM boot complete.")
        return True

    async def shutdown(self):
        """
        Clears model references. Actual model cleanup is handled by Python's GC.
        """
        logger.info("STM shutting down...")
        self.summarizer_model = None
        self.summarizer_tokenizer = None
        self.intent_model = None
        self.intent_tokenizer = None
        # self.intent_classifier_pipeline = None
        logger.info("STM models and tokenizers references cleared.")
        return True

    async def add_turn(self, session_id: str, turn_data: Dict[str, Any]) -> None:
        """
        Adds a new turn/item to the specified session's buffer.
        Manages the rolling window.
        `turn_data` should be a dictionary, e.g., {"role": "user", "content": "Hello", "timestamp": time.time()}
        """
        if not session_id:
            logger.error("Session ID must be provided to add a turn.")
            return

        if not isinstance(turn_data, dict):
            logger.error(f"turn_data must be a dictionary, got {type(turn_data)} for session '{session_id}'.")
            return

        if session_id not in self._conversations_stub:
            self._conversations_stub[session_id] = deque(maxlen=self.max_turns)
            logger.info(f"New session '{session_id}' created in STM with max_turns={self.max_turns}.")

        if "timestamp" not in turn_data:
            turn_data["timestamp"] = time.time() # Add current time if timestamp is missing

        self._conversations_stub[session_id].append(turn_data)
        logger.info(f"Added turn to session '{session_id}'. Buffer size: {len(self._conversations_stub[session_id])}. Turn: {str(turn_data)[:150]}...") # Log more of the turn

    async def get_recent_turns(self, session_id: str, num_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the last `num_turns` for a session. If `num_turns` is None or 0 or invalid, return all turns for the session.
        """
        if session_id not in self._conversations_stub:
            logger.warning(f"Session '{session_id}' not found in STM for get_recent_turns.")
            return []

        session_deque = self._conversations_stub[session_id]
        if num_turns is None or not isinstance(num_turns, int) or num_turns <= 0 or num_turns >= len(session_deque):
            turns_to_return = list(session_deque)
        else:
            turns_to_return = list(session_deque)[-num_turns:]

        logger.info(f"Retrieved {len(turns_to_return)} turns for session '{session_id}'. Requested: {num_turns}")
        return turns_to_return

    async def get_full_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all turns for a specified session. Alias for get_recent_turns with num_turns=None.
        """
        return await self.get_recent_turns(session_id, num_turns=None)

    async def _generate_summary_hf(self, session_id: str, turns: List[Dict[str, Any]]) -> str:
        if not self.summarizer_model or not self.summarizer_tokenizer:
            logger.warning(f"Summarization model/tokenizer not available for session '{session_id}'. Returning raw content.")
            return "\n".join([f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in turns]) if turns else "No content."

        logger.info(f"Generating summary for session '{session_id}' with {len(turns)} turns using {self.summarizer_model.name_or_path}.")
        if not turns:
            return "No conversation turns to summarize for this session."

        # Concatenate turns to form input text for summarization
        text_to_summarize = " ".join([str(turn.get("content", "")) for turn in turns if turn.get("content")])
        if not text_to_summarize.strip():
            return "No textual content in turns to summarize."

        loop = asyncio.get_running_loop()
        try:
            # Synchronous Hugging Face model inference needs to be run in an executor
            def _summarize_sync():
                inputs = self.summarizer_tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
                # Ensure model is on CPU if torch.cuda not available or not desired for this specific model
                # device = "cuda" if torch.cuda.is_available() else "cpu"
                # self.summarizer_model.to(device)
                # inputs = inputs.to(device)
                summary_ids = self.summarizer_model.generate(
                    inputs['input_ids'],
                    num_beams=4, # Example generation parameters
                    max_length=150, # Adjust as needed
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


    async def _extract_intents_hf(self, session_id: str, turns: List[Dict[str, Any]], threshold: float = 0.7) -> List[str]:
        if not self.intent_model or not self.intent_tokenizer:
            logger.warning(f"Intent model/tokenizer not available for session '{session_id}'. Returning empty list.")
            return []

        logger.info(f"Extracting intents for session '{session_id}' with {len(turns)} turns using {self.intent_model.name_or_path}.")
        if not turns:
            return []

        # Use last turn's content or concatenate a few recent turns for intent detection
        text_for_intent = turns[-1].get("content", "") # Focus on the last turn
        # Alternatively, concatenate last few turns:
        # text_for_intent = " ".join([str(turn.get("content", "")) for turn in turns[-3:] if turn.get("content")])

        if not text_for_intent.strip():
            return ["unknown_intent_empty_text"]

        loop = asyncio.get_running_loop()
        try:
            # Using pipeline for zero-shot for simplicity, can be done manually too.
            # Need to instantiate pipeline with model and tokenizer if not done in __init__
            # For this example, let's do the manual zero-shot steps if pipeline wasn't created

            def _classify_sync():
                inputs = self.intent_tokenizer(text_for_intent, self.candidate_intent_labels, return_tensors="pt", padding=True, truncation=True)
                # device = "cuda" if torch.cuda.is_available() else "cpu"
                # self.intent_model.to(device)
                # inputs = inputs.to(device)
                with torch.no_grad(): # Important for inference
                    logits = self.intent_model(**inputs).logits

                probabilities = torch.softmax(logits, dim=1).squeeze().tolist() # Get probabilities for the text against all candidate labels

                identified_intents = []
                for i, prob in enumerate(probabilities):
                    if prob > threshold:
                        identified_intents.append(f"{self.candidate_intent_labels[i]} ({prob:.2f})")
                return identified_intents if identified_intents else ["general_discussion"]

            intents = await loop.run_in_executor(None, _classify_sync)
            logger.info(f"Identified intents for session '{session_id}': {intents}")
            return intents
        except Exception as e:
            logger.error(f"Error during intent extraction for session {session_id}: {e}", exc_info=True)
            return [f"error_extracting_intent: {e}"]


    async def summarize_session(self, session_id: str) -> str:
        if session_id not in self._conversations_stub:
            logger.warning(f"Session '{session_id}' not found for summarization.")
            return "Session not found in STM."

        turns = await self.get_full_conversation(session_id)
        if not turns:
            logger.info(f"No turns in session '{session_id}' to summarize.")
            return "No content in session to summarize."

        summary = await self._generate_summary_hf(session_id, turns)
        return summary

    async def flush_session_to_ltm(self, session_id: str, summarize: bool = True, clear_after_flush: bool = True) -> Dict[str, Any]:
        logger.info(f"Attempting to flush session '{session_id}' to LTM. Summarize: {summarize}, Clear after: {clear_after_flush}")

        if session_id not in self._conversations_stub or not self._conversations_stub[session_id]:
            logger.warning(f"Session '{session_id}' not found in STM or is empty. Nothing to flush.")
            return {"status": "not_found_or_empty", "flushed_items_count": 0, "ltm_id": None}

        turns = await self.get_full_conversation(session_id)
        intents = await self._extract_intents_hf(session_id, turns) # Use new HF method

        ltm_chunk_id_base = f"stm_session_{session_id}"
        text_to_store: str

        # More detailed metadata
        first_turn_time = turns[0].get('timestamp') if turns else None
        last_turn_time = turns[-1].get('timestamp') if turns else None
        metadata: Dict[str, Any] = {
            "source_system": "STM",
            "original_session_id": session_id,
            "num_turns_in_session": len(turns),
            "identified_intents": intents,
            "session_start_time_unix": first_turn_time,
            "session_end_time_unix": last_turn_time,
            "stm_flush_time_unix": time.time()
        }

        if summarize:
            summary = await self.summarize_session(session_id)
            text_to_store = summary
            metadata["content_type"] = "session_summary"
            ltm_chunk_id = f"{ltm_chunk_id_base}_summary"
            logger.info(f"Using summary of session '{session_id}' for LTM storage.")
        else:
            text_to_store = "\n".join([f"[{turn.get('timestamp')}] {turn.get('role', 'N/A')}: {turn.get('content', '')}" for turn in turns])
            metadata["content_type"] = "full_conversation_text"
            ltm_chunk_id = f"{ltm_chunk_id_base}_full"
            logger.info(f"Using full conversation text of session '{session_id}' for LTM storage.")

        metadata["text_char_length"] = len(text_to_store)
        embedding = await self.ltm.generate_embedding(text_to_store) # LTM handles caching for its embeddings

        ltm_stored_id = await self.ltm.store_memory_chunk(
            chunk_id=ltm_chunk_id,
            text_content=text_to_store,
            embedding=embedding,
            metadata=metadata
        )

        if clear_after_flush:
            if session_id in self._conversations_stub:
                self._conversations_stub[session_id].clear()
                logger.info(f"Cleared session '{session_id}' from STM after flushing to LTM.")

        result = {
            "status": "success",
            "ltm_id": ltm_stored_id,
            "flushed_content_type": metadata["content_type"],
            "num_original_turns": len(turns)
        }
        logger.info(f"Successfully flushed session '{session_id}' to LTM. Result: {json.dumps(result)}")
        return result

    async def prefetch_session(self, session_id: str) -> None:
        logger.info(f"Prefetch requested for session '{session_id}' (stubbed - no action for in-memory STM).")
        await asyncio.sleep(0.01)


async def main():
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- STM Example Usage (with Injected Dependencies) ---")

    # STM Example Usage (with Real Hugging Face Models)
    # This example requires `transformers` and `torch`. Models are downloaded on first run.
    # A mock LTM is used to simplify the example.

    logger.info("--- STM Example Usage (with Real Hugging Face Models) ---")
    logger.warning("This example will download Hugging Face models if not already cached locally.")

    # Mock LTM for STM example
    class MockLTM:
        async def boot(self): logger.info("MockLTM booted."); return True
        async def shutdown(self): logger.info("MockLTM shutdown."); return True
        async def generate_embedding(self, text_content: str) -> List[float]:
            logger.debug(f"MockLTM.generate_embedding for: '{text_content[:30]}...'")
            return [0.1] * 384 # Dummy embedding
        async def store_memory_chunk(self, chunk_id: Optional[str], text_content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
            mem_id = chunk_id or str(uuid.uuid4())
            logger.info(f"MockLTM.store_memory_chunk: Storing ID '{mem_id}', text snip '{text_content[:50]}...', meta {metadata}")
            # Store it internally if needed for verification in the example
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
    # Ensure these model names are valid Hugging Face model identifiers.
    # Using smaller, faster models for the example.
    # Summarizer: sshleifer/distilbart-cnn-6-6 is smaller than 12-6
    summarizer_conf = NLPServiceConfig(provider="huggingface_transformer", model="sshleifer/distilbart-cnn-6-6")
    # Intent Tagger (Zero-Shot): facebook/bart-large-mnli is quite large.
    # Using a smaller alternative if available or keeping it and noting potential download size/time.
    # For now, let's stick to the one mentioned in task, or a smaller one if known.
    # MoritzLaurer/mDeBERTa-v3-base-mnli-xnli is a multilingual option that might be smaller than bart-large.
    # For simplicity, we'll use the one from the prompt:
    intent_tagger_conf = NLPServiceConfig(provider="huggingface_transformer", model="facebook/bart-large-mnli")

    candidate_intents_for_example = ["general question", "technical support", "sales inquiry", "feedback", "complaint"]


    # Instantiate STM with real models (will be loaded in STM's __init__)
    try:
        stm_system = STM(
            ltm=mock_ltm_instance,
            summarizer_nlp_config=summarizer_conf,
            intent_tagging_nlp_config=intent_tagger_conf,
            max_turns=3,
            candidate_intents=candidate_intents_for_example
        )
        await stm_system.boot() # Boot STM (models should be loaded now)
    except Exception as e:
        logger.error(f"Failed to initialize or boot STM with real models: {e}", exc_info=True)
        logger.error("Ensure you have an internet connection for model downloads on first run, "
                     "and that `transformers`, `torch`, `sentencepiece` are installed.")
        if hasattr(mock_ltm_instance, 'shutdown'): await mock_ltm_instance.shutdown()
        return


    session_id_1 = "user123_chat_hf_001"

    await stm_system.add_turn(session_id_1, {"role": "user", "content": "Hello there! I have a question about your product.", "timestamp": time.time()})
    await stm_system.add_turn(session_id_1, {"role": "assistant", "content": "Hi! I'm happy to help. What's your question?", "timestamp": time.time() + 1})
    await stm_system.add_turn(session_id_1, {"role": "user", "content": "I'm interested in the new Context Kernel feature. Can you explain it and how it helps with AI memory?", "timestamp": time.time() + 2})

    logger.info(f"\n--- After 3 turns (max_turns=3) for session '{session_id_1}' ---")
    turns_s1_initial = await stm_system.get_full_conversation(session_id_1)
    for i, turn in enumerate(turns_s1_initial): logger.info(f"Turn {i} (initial) for '{session_id_1}': {json.dumps(turn)}")
    assert len(turns_s1_initial) == 3
    assert turns_s1_initial[0]['content'] == "Hello there! I have a question about your product."

    await stm_system.add_turn(session_id_1, {"role": "assistant", "content": "Context Kernels are indeed a fascinating new development in AI memory, allowing for more persistent and relevant information recall.", "timestamp": time.time() + 3})
    logger.info(f"\n--- After 4th turn (max_turns=3) for session '{session_id_1}' ---")
    turns_s1_updated = await stm_system.get_full_conversation(session_id_1)
    for i, turn in enumerate(turns_s1_updated): logger.info(f"Turn {i} (updated) for '{session_id_1}': {json.dumps(turn)}")
    assert len(turns_s1_updated) == 3 # Max turns constraint
    assert turns_s1_updated[0]['content'] == "Hi! I'm happy to help. What's your question?" # First turn should be gone

    recent_2_turns_s1 = await stm_system.get_recent_turns(session_id_1, num_turns=2)
    logger.info(f"\nRecent 2 turns for session '{session_id_1}': {json.dumps(recent_2_turns_s1, indent=2)}")
    assert len(recent_2_turns_s1) == 2

    logger.info(f"\n--- Generating summary for session '{session_id_1}' ---")
    summary_s1 = await stm_system.summarize_session(session_id_1) # Uses real summarization model
    logger.info(f"Summary for session '{session_id_1}':\n{summary_s1}")
    assert summary_s1 and "Context Kernel" in summary_s1, "Summary does not seem to reflect content."

    logger.info(f"\n--- Flushing session '{session_id_1}' to LTM (summarized) ---")
    flush_status_s1_summarized = await stm_system.flush_session_to_ltm(session_id_1, summarize=True, clear_after_flush=True)
    logger.info(f"Flush status for session '{session_id_1}' (summarized): {json.dumps(flush_status_s1_summarized)}")
    assert flush_status_s1_summarized["status"] == "success"
    assert flush_status_s1_summarized["ltm_id"] is not None

    turns_s1_after_flush = await stm_system.get_full_conversation(session_id_1)
    logger.info(f"Turns in session '{session_id_1}' after summarized flush and clear: {turns_s1_after_flush}")
    assert len(turns_s1_after_flush) == 0

    # Verify what was stored in MockLTM
    if hasattr(mock_ltm_instance, '_stored_ltm_chunks'):
        ltm_entry_s1_data = mock_ltm_instance._stored_ltm_chunks.get(flush_status_s1_summarized["ltm_id"])
        logger.info(f"Data stored in MockLTM for {flush_status_s1_summarized['ltm_id']}: {json.dumps(ltm_entry_s1_data, indent=2)}")
        assert ltm_entry_s1_data is not None
        assert ltm_entry_s1_data["metadata"]["original_session_id"] == session_id_1
        assert "general inquiry" in str(ltm_entry_s1_data["metadata"]["identified_intents"]).lower() or \
               "general question" in str(ltm_entry_s1_data["metadata"]["identified_intents"]).lower()


    # --- Another session for full text flush ---
    session_id_2 = "user789_chat_hf_002"
    await stm_system.add_turn(session_id_2, {"role": "user", "content": "I'm having an issue with product X. It's not working as expected.", "timestamp": time.time()})
    await stm_system.add_turn(session_id_2, {"role": "assistant", "content": "I'm sorry to hear that. Could you describe the problem in more detail?", "timestamp": time.time() + 1})

    logger.info(f"\n--- Flushing session '{session_id_2}' to LTM (full text) ---")
    flush_status_s2_full = await stm_system.flush_session_to_ltm(session_id_2, summarize=False, clear_after_flush=False)
    logger.info(f"Flush status for session '{session_id_2}' (full text): {json.dumps(flush_status_s2_full)}")
    assert flush_status_s2_full["status"] == "success"

    turns_s2_after_flush_no_clear = await stm_system.get_full_conversation(session_id_2)
    logger.info(f"Turns in session '{session_id_2}' after full flush (no clear): {turns_s2_after_flush_no_clear}")
    assert len(turns_s2_after_flush_no_clear) == 2 # Should still be there

    if hasattr(mock_ltm_instance, '_stored_ltm_chunks'):
        ltm_entry_s2_data = mock_ltm_instance._stored_ltm_chunks.get(flush_status_s2_full["ltm_id"])
        logger.info(f"Data stored in MockLTM for {flush_status_s2_full['ltm_id']}: {json.dumps(ltm_entry_s2_data, indent=2)}")
        assert ltm_entry_s2_data is not None
        assert "problem report" in str(ltm_entry_s2_data["metadata"]["identified_intents"]).lower()
        assert "product X" in ltm_entry_s2_data["text_content"]


    await stm_system.prefetch_session("some_other_session_id") # Example call

    # Shutdown sequence
    await stm_system.shutdown()
    await mock_ltm_instance.shutdown()

    logger.info("--- STM Example Usage (with Real Hugging Face Models) Complete ---")

if __name__ == "__main__":
    # This main function uses real Hugging Face models for summarization and intent tagging.
    # It requires `transformers` and `torch` to be installed.
    # Models are downloaded on first use if not already cached by Hugging Face.
    # LTM is mocked to simplify this example.
    asyncio.run(main())
