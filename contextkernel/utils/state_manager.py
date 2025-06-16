# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

try:
    import aioredis # type: ignore
except ImportError:
    aioredis = None # type: ignore

# ContextKernel imports
from contextkernel.core_logic.exceptions import MemoryAccessError, ConfigurationError

logger = logging.getLogger(__name__)

class AbstractStateManager(ABC):
    """Abstract base class for state managers."""

    @abstractmethod
    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the state for a given session ID.

        Args:
            session_id: The unique identifier for the session.

        Returns:
            A dictionary representing the state, or None if not found.
        """
        pass

    @abstractmethod
    async def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        Saves the state for a given session ID.

        Args:
            session_id: The unique identifier for the session.
            state: A dictionary representing the state to be saved.
        """
        pass

    @abstractmethod
    async def delete_state(self, session_id: str) -> None:
        """
        Deletes the state for a given session ID.

        Args:
            session_id: The unique identifier for the session.
        """
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def close(self):
        """Closes any underlying connections or resources."""
        pass


class InMemoryStateManager(AbstractStateManager):
    """Manages state in an in-memory dictionary."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        logger.info("InMemoryStateManager initialized.")

    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"InMemoryStateManager: Getting state for session_id: {session_id}")
        return self._store.get(session_id)

    async def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        logger.debug(f"InMemoryStateManager: Saving state for session_id: {session_id}, State: {state}")
        self._store[session_id] = state

    async def delete_state(self, session_id: str) -> None:
        logger.debug(f"InMemoryStateManager: Deleting state for session_id: {session_id}")
        if session_id in self._store:
            del self._store[session_id]

    async def close(self):
        logger.info("InMemoryStateManager closed (no-op).")
        pass


class RedisStateManager(AbstractStateManager):
    """Manages state using a Redis backend."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        if aioredis is None:
            msg = "aioredis library not found. Please install with `pip install aioredis`."
            logger.error(msg)
            # Raising ConfigurationError as aioredis is a required dependency for this state manager
            raise ConfigurationError(msg)

        # Construct Redis URL, including password if provided
        if password:
            self.redis_url = f"redis://:{password}@{host}:{port}/{db}"
        else:
            self.redis_url = f"redis://{host}:{port}/{db}"

        try:
            # `from_url` is the standard way to create a Redis client/pool in aioredis v2+
            self.redis = aioredis.from_url(self.redis_url)
            logger.info(f"RedisStateManager initialized, connected to redis://{host}:{port}/{db}")
        except aioredis.RedisError as e: # More specific exception for Redis connection issues
            logger.error(f"Failed to initialize Redis client with URL redis://{host}:{port}/{db}: {e}", exc_info=True)
            raise ConfigurationError(f"Redis connection failed at {host}:{port}: {e}") from e
        except Exception as e: # Catch any other unexpected errors during initialization
            logger.error(f"An unexpected error occurred during Redis client initialization: {e}", exc_info=True)
            raise ConfigurationError(f"Unexpected error initializing Redis client: {e}") from e

    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.redis: # Should ideally not be hit if __init__ raises ConfigurationError
            logger.error("Redis client not available. Cannot get state.")
            raise ConfigurationError("Redis client not initialized.")
        try:
            logger.debug(f"RedisStateManager: Getting state for session_id: {session_id}")
            raw_state = await self.redis.get(session_id)
            if raw_state:
                return json.loads(raw_state.decode('utf-8'))
            return None
        except aioredis.RedisError as e:
            logger.error(f"RedisStateManager: Redis error getting state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Redis GET operation failed for session {session_id}") from e
        except json.JSONDecodeError as e:
            logger.error(f"RedisStateManager: Error decoding JSON state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Failed to decode state for session {session_id}") from e

    async def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        if not self.redis: # Should ideally not be hit
            logger.error("Redis client not available. Cannot save state.")
            raise ConfigurationError("Redis client not initialized.")
        try:
            logger.debug(f"RedisStateManager: Saving state for session_id: {session_id}") # State not logged to avoid large log entries
            await self.redis.set(session_id, json.dumps(state))
        except aioredis.RedisError as e:
            logger.error(f"RedisStateManager: Redis error saving state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Redis SET operation failed for session {session_id}") from e

    async def delete_state(self, session_id: str) -> None:
        if not self.redis: # Should ideally not be hit
            logger.error("Redis client not available. Cannot delete state.")
            raise ConfigurationError("Redis client not initialized.")
        try:
            logger.debug(f"RedisStateManager: Deleting state for session_id: {session_id}")
            await self.redis.delete(session_id)
        except aioredis.RedisError as e:
            logger.error(f"RedisStateManager: Redis error deleting state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Redis DELETE operation failed for session {session_id}") from e

    async def close(self):
        """Closes the Redis connection pool."""
        if self.redis:
            try:
                await self.redis.close()
                # In aioredis, `close()` starts the closing process for the pool.
                # `wait_closed()` could be used if we needed to ensure full closure before proceeding,
                # but often it's managed by the library or not strictly necessary for basic cleanup.
                logger.info("Redis connection pool closed.")
            except aioredis.RedisError as e: # Catch potential errors during close
                logger.error(f"Error closing Redis connection pool: {e}", exc_info=True)
            except Exception as e:
                 logger.error(f"Unexpected error closing Redis connection pool: {e}", exc_info=True)
        self.redis = None # type: ignore

    # Implementing __aenter__ and __aexit__ for asynchronous context management
    async def __aenter__(self):
        # Connection is established in __init__. If it failed, an error would have been raised.
        # So, if we reach here, self.redis should be valid.
        logger.debug("RedisStateManager entered context.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.debug("RedisStateManager exiting context, closing connection.")
        await self.close()

[end of contextkernel/utils/state_manager.py]
