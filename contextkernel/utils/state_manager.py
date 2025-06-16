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


class RedisStateManager(AbstractStateManager):
    """Manages state using a Redis backend."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        if aioredis is None:
            msg = "aioredis library not found. Please install with `pip install aioredis redis`."
            logger.error(msg)
            raise ImportError(msg)

        self.redis_url = f"redis://{host}:{port}"
        # Include password in URL if provided
        if password:
            self.redis_url = f"redis://:{password}@{host}:{port}/{db}"
        else:
            self.redis_url = f"redis://{host}:{port}/{db}"

        try:
            # `from_url` is the correct way for aioredis v2+
            self.redis = aioredis.from_url(self.redis_url)
            logger.info(f"RedisStateManager initialized with URL: redis://{host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client with URL {self.redis_url}: {e}", exc_info=True)
            # Fallback or raise, depending on desired behavior. Here we'll let it raise implicitly
            # if self.redis is not set, or handle it more gracefully.
            # For now, if connection fails here, subsequent calls will fail.
            self.redis = None # type: ignore


    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            logger.error("Redis client not initialized. Cannot get state.")
            return None
        try:
            logger.debug(f"RedisStateManager: Getting state for session_id: {session_id}")
            raw_state = await self.redis.get(session_id)
            if raw_state:
                return json.loads(raw_state.decode('utf-8')) # type: ignore
            return None
        except Exception as e:
            logger.error(f"RedisStateManager: Error getting state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Redis GET failed for session {session_id}") from e

    async def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        if not self.redis:
            logger.error("Redis client not initialized. Cannot save state.")
            # Optionally raise ConfigurationError here if Redis connection is absolutely necessary
            return
        try:
            logger.debug(f"RedisStateManager: Saving state for session_id: {session_id}, State: {state}")
            await self.redis.set(session_id, json.dumps(state))
        except Exception as e:
            logger.error(f"RedisStateManager: Error saving state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Redis SET failed for session {session_id}") from e

    async def delete_state(self, session_id: str) -> None:
        if not self.redis:
            logger.error("Redis client not initialized. Cannot delete state.")
            return
        try:
            logger.debug(f"RedisStateManager: Deleting state for session_id: {session_id}")
            await self.redis.delete(session_id)
        except Exception as e:
            logger.error(f"RedisStateManager: Error deleting state for session_id {session_id}: {e}", exc_info=True)
            raise MemoryAccessError(f"Redis DELETE failed for session {session_id}") from e

    async def close(self):
        """Closes the Redis connection."""
        if self.redis:
            try:
                await self.redis.close()
                # For aioredis v2, `close()` should be followed by `wait_closed()` if managed explicitly.
                # However, typically `aioredis.from_url` creates a connection pool that manages connections.
                # Explicit closing might be needed on application shutdown.
                # await self.redis.wait_closed() # This might be needed for older aioredis or specific use cases
                logger.info("Redis connection closed.")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}", exc_info=True)
