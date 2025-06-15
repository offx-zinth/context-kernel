import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Deque
import json # For formatting dicts in logs/summaries
import time # For adding timestamps if not present

# Assuming LTM is in the same package directory
from .ltm import LTM

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Commenting out basicConfig to avoid conflict if MemoryKernel also calls it.
logger = logging.getLogger(__name__)

class STM:
    """
    Short-Term Memory (STM) system.
    Manages a fast-access buffer of recent conversational turns or items using a rolling window.
    """

    DEFAULT_MAX_TURNS = 50  # Default number of turns to keep per session

    def __init__(self, ltm_instance: LTM, max_turns: int = DEFAULT_MAX_TURNS, redis_config: Optional[Dict] = None):
        if not isinstance(ltm_instance, LTM):
            logger.error("STM initialized with invalid LTM instance.")
            raise ValueError("STM requires a valid LTM instance.")

        self.ltm = ltm_instance
        self.max_turns = max_turns
        self.redis_config = redis_config or {"type": "stubbed_in_memory_deque"}

        # _conversations_stub stores: session_id -> deque([turn_data_dict, ...])
        # Each turn_data_dict should ideally have 'timestamp', 'role', 'content'
        self._conversations_stub: Dict[str, Deque[Dict[str, Any]]] = {}

        logger.info(f"STM initialized with LTM instance, max_turns: {self.max_turns}, Config: {self.redis_config}")

    async def boot(self):
        """
        Simulates connecting to any backing store for STM (e.g., Redis) if used.
        """
        logger.info("STM booting up... (simulating connection/setup)")
        await asyncio.sleep(0.01)
        logger.info("STM boot complete. System is 'online'.")
        return True

    async def shutdown(self):
        """
        Simulates disconnecting or cleaning up STM resources.
        """
        logger.info("STM shutting down... (simulating disconnection/cleanup)")
        await asyncio.sleep(0.01)
        logger.info("STM shutdown complete.")
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

    async def _stub_summarizer_call(self, session_id: str, turns: List[Dict[str, Any]]) -> str:
        logger.info(f"Simulating summarizer call for session '{session_id}' with {len(turns)} turns.")
        if not turns:
            return "No conversation turns to summarize for this session."

        await asyncio.sleep(0.02) # Simulate API call latency

        text_summary = f"Summary of session '{session_id}' ({len(turns)} turns):\n"
        for i, turn in enumerate(turns):
            role = turn.get("role", "unknown_role")
            content = str(turn.get("content", "[no content]"))[:70]
            text_summary += f"  Turn {i+1} ({role}): {content}...\n"
        text_summary += f"Overall, the conversation involved topics like {', '.join(set(turn.get('role','N/A') for turn in turns))} and ended with content from {turns[-1].get('role','N/A')}."

        logger.info(f"Generated stub summary for session '{session_id}'. Length: {len(text_summary)}")
        return text_summary

    async def _stub_intent_tagging(self, session_id: str, turns: List[Dict[str, Any]]) -> List[str]:
        logger.info(f"Simulating intent tagging for session '{session_id}' with {len(turns)} turns.")
        if not turns:
            return []

        await asyncio.sleep(0.01)
        intents = set()
        for turn in turns:
            content = str(turn.get("content", "")).lower()
            if "question" in content or "?" in content: intents.add("question_asking")
            if "buy" in content or "purchase" in content or "order" in content: intents.add("purchase_intent")
            if "help" in content or "support" in content or "problem" in content: intents.add("seeking_support")
            if "kernel" in content or "context" in content: intents.add("context_kernel_discussion")

        final_intents = list(intents) if intents else ["general_discussion"]
        logger.info(f"Identified stub intents for session '{session_id}': {final_intents}")
        return final_intents

    async def summarize_session(self, session_id: str) -> str:
        if session_id not in self._conversations_stub:
            logger.warning(f"Session '{session_id}' not found for summarization.")
            return "Session not found in STM."

        turns = await self.get_full_conversation(session_id)
        if not turns:
            logger.info(f"No turns in session '{session_id}' to summarize.")
            return "No content in session to summarize."

        summary = await self._stub_summarizer_call(session_id, turns)
        return summary

    async def flush_session_to_ltm(self, session_id: str, summarize: bool = True, clear_after_flush: bool = True) -> Dict[str, Any]:
        logger.info(f"Attempting to flush session '{session_id}' to LTM. Summarize: {summarize}, Clear after: {clear_after_flush}")

        if session_id not in self._conversations_stub or not self._conversations_stub[session_id]:
            logger.warning(f"Session '{session_id}' not found in STM or is empty. Nothing to flush.")
            return {"status": "not_found_or_empty", "flushed_items_count": 0, "ltm_id": None}

        turns = await self.get_full_conversation(session_id)
        intents = await self._stub_intent_tagging(session_id, turns)

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

    logger.info("--- STM Example Usage ---")

    ltm_stub = LTM()
    await ltm_stub.boot()

    stm_system = STM(ltm_instance=ltm_stub, max_turns=3)
    await stm_system.boot()

    session_id_1 = "user123_chat456"

    await stm_system.add_turn(session_id_1, {"role": "user", "content": "Hello STM!"})
    await stm_system.add_turn(session_id_1, {"role": "assistant", "content": "Hello User! How can I help?"})
    await stm_system.add_turn(session_id_1, {"role": "user", "content": "Tell me about Context Kernels. Any questions?"})

    logger.info(f"\n--- After 3 turns (max_turns=3) ---")
    turns_s1 = await stm_system.get_full_conversation(session_id_1)
    for i, turn in enumerate(turns_s1): logger.info(f"Turn {i}: {json.dumps(turn)}")
    assert len(turns_s1) == 3
    assert turns_s1[0]['content'] == "Hello STM!"

    await stm_system.add_turn(session_id_1, {"role": "assistant", "content": "Context Kernels are a conceptual framework for AI memory. No problem."})
    logger.info(f"\n--- After 4th turn (max_turns=3) ---")
    turns_s1_updated = await stm_system.get_full_conversation(session_id_1)
    for i, turn in enumerate(turns_s1_updated): logger.info(f"Turn {i}: {json.dumps(turn)}")
    assert len(turns_s1_updated) == 3
    assert turns_s1_updated[0]['content'] == "Hello User! How can I help?"

    recent_2_turns = await stm_system.get_recent_turns(session_id_1, num_turns=2)
    logger.info(f"\nRecent 2 turns: {json.dumps(recent_2_turns, indent=2)}")
    assert len(recent_2_turns) == 2

    summary_s1 = await stm_system.summarize_session(session_id_1)
    logger.info(f"\nSummary for session '{session_id_1}':\n{summary_s1}")
    assert "Summary of session" in summary_s1
    assert "Context Kernels are a conceptual framework" in summary_s1

    logger.info(f"\n--- Flushing session '{session_id_1}' to LTM (summarized) ---")
    flush_status_summarized = await stm_system.flush_session_to_ltm(session_id_1, summarize=True, clear_after_flush=True)
    logger.info(f"Flush status (summarized): {json.dumps(flush_status_summarized)}")
    assert flush_status_summarized["status"] == "success"

    turns_after_flush = await stm_system.get_full_conversation(session_id_1)
    logger.info(f"Turns in session '{session_id_1}' after summarized flush and clear: {turns_after_flush}")
    assert len(turns_after_flush) == 0

    ltm_entry_summarized = await ltm_stub.get_memory_by_id(flush_status_summarized["ltm_id"])
    logger.info(f"LTM entry for summarized flush: {json.dumps(ltm_entry_summarized, indent=2)}")
    assert ltm_entry_summarized is not None
    assert ltm_entry_summarized["metadata"]["original_session_id"] == session_id_1
    assert "context_kernel_discussion" in ltm_entry_summarized["metadata"]["identified_intents"]
    assert "question_asking" in ltm_entry_summarized["metadata"]["identified_intents"]


    session_id_2 = "user789_chat002"
    await stm_system.add_turn(session_id_2, {"role": "user", "content": "What is AI safety? Any problems with it?"})
    await stm_system.add_turn(session_id_2, {"role": "assistant", "content": "AI safety is a field dedicated to ensuring artificial intelligence systems are designed and operate in ways that are safe and beneficial to humans."})

    logger.info(f"\n--- Flushing session '{session_id_2}' to LTM (full text) ---")
    flush_status_full = await stm_system.flush_session_to_ltm(session_id_2, summarize=False, clear_after_flush=False)
    logger.info(f"Flush status (full text): {json.dumps(flush_status_full)}")
    assert flush_status_full["status"] == "success"

    turns_s2_after_flush_no_clear = await stm_system.get_full_conversation(session_id_2)
    logger.info(f"Turns in session '{session_id_2}' after full flush (no clear): {turns_s2_after_flush_no_clear}")
    assert len(turns_s2_after_flush_no_clear) == 2

    ltm_entry_full = await ltm_stub.get_memory_by_id(flush_status_full["ltm_id"])
    logger.info(f"LTM entry for full flush: {json.dumps(ltm_entry_full, indent=2)}")
    assert ltm_entry_full is not None
    assert "What is AI safety?" in ltm_entry_full["text_content"]
    assert "seeking_support" in ltm_entry_full["metadata"]["identified_intents"] # from "problems"

    await stm_system.prefetch_session("some_other_session_id")

    await stm_system.shutdown()
    await ltm_stub.shutdown()
    logger.info("--- STM Example Usage Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
