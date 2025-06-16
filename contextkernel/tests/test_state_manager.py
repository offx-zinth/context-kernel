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

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Modules to test
from contextkernel.utils.state_manager import InMemoryStateManager, RedisStateManager


class TestInMemoryStateManager:
    @pytest.mark.asyncio
    async def test_save_and_get_state(self):
        manager = InMemoryStateManager()
        session_id = "test_session_123"
        state_data = {"key1": "value1", "count": 5}

        await manager.save_state(session_id, state_data)
        retrieved_state = await manager.get_state(session_id)

        assert retrieved_state is not None
        assert retrieved_state == state_data

    @pytest.mark.asyncio
    async def test_get_state_non_existent(self):
        manager = InMemoryStateManager()
        retrieved_state = await manager.get_state("non_existent_session")
        assert retrieved_state is None

    @pytest.mark.asyncio
    async def test_delete_state(self):
        manager = InMemoryStateManager()
        session_id = "session_to_delete"
        state_data = {"message": "hello"}
        await manager.save_state(session_id, state_data)

        retrieved_before_delete = await manager.get_state(session_id)
        assert retrieved_before_delete is not None

        await manager.delete_state(session_id)
        retrieved_after_delete = await manager.get_state(session_id)
        assert retrieved_after_delete is None

    @pytest.mark.asyncio
    async def test_overwrite_state(self):
        manager = InMemoryStateManager()
        session_id = "session_overwrite"
        initial_state = {"version": 1}
        updated_state = {"version": 2, "new_field": "added"}

        await manager.save_state(session_id, initial_state)
        await manager.save_state(session_id, updated_state)

        retrieved_state = await manager.get_state(session_id)
        assert retrieved_state == updated_state


@pytest.fixture
def mock_aioredis_client(monkeypatch):
    """Mocks aioredis.from_url and the client it returns."""
    mock_redis_instance = AsyncMock() # This will be the object returned by from_url
    mock_redis_instance.get = AsyncMock(return_value=None)
    mock_redis_instance.set = AsyncMock()
    mock_redis_instance.delete = AsyncMock()
    mock_redis_instance.close = AsyncMock() # For the close method in RedisStateManager

    mock_from_url = MagicMock(return_value=mock_redis_instance)
    monkeypatch.setattr("contextkernel.utils.state_manager.aioredis.from_url", mock_from_url)
    return mock_from_url, mock_redis_instance


class TestRedisStateManager:
    def test_init_success(self, mock_aioredis_client):
        """Tests successful initialization of RedisStateManager."""
        mock_constructor, _ = mock_aioredis_client
        manager = RedisStateManager(host="testhost", port=1234)
        assert manager.redis is not None
        mock_constructor.assert_called_once_with("redis://testhost:1234/0")

    def test_init_failure_import_error(self, monkeypatch, caplog):
        """Tests initialization failure if aioredis is not installed."""
        monkeypatch.setattr("contextkernel.utils.state_manager.aioredis", None)
        with pytest.raises(ImportError) as excinfo:
            RedisStateManager()
        assert "aioredis library not found" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_save_and_get_state_redis(self, mock_aioredis_client):
        _, mock_redis = mock_aioredis_client
        manager = RedisStateManager() # Uses mocked client

        session_id = "redis_session_1"
        state_data = {"user": "test_user", "data": [1, 2, 3]}

        # Mock what redis.get would return after a set
        async def mock_get_after_set(key):
            if key == session_id:
                return json.dumps(state_data).encode('utf-8')
            return None
        mock_redis.get.side_effect = mock_get_after_set

        await manager.save_state(session_id, state_data)
        mock_redis.set.assert_called_once_with(session_id, json.dumps(state_data))

        retrieved_state = await manager.get_state(session_id)
        mock_redis.get.assert_called_with(session_id) # Check it was called again for get
        assert retrieved_state == state_data

    @pytest.mark.asyncio
    async def test_get_state_redis_non_existent(self, mock_aioredis_client):
        _, mock_redis = mock_aioredis_client
        mock_redis.get.return_value = None # Simulate key not found
        manager = RedisStateManager()

        retrieved_state = await manager.get_state("empty_session")
        assert retrieved_state is None
        mock_redis.get.assert_called_once_with("empty_session")

    @pytest.mark.asyncio
    async def test_delete_state_redis(self, mock_aioredis_client):
        _, mock_redis = mock_aioredis_client
        manager = RedisStateManager()
        session_id = "redis_delete_me"

        await manager.delete_state(session_id)
        mock_redis.delete.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_redis_connection_error_get(self, mock_aioredis_client, caplog):
        _, mock_redis = mock_aioredis_client
        mock_redis.get.side_effect = Exception("Redis connection failed!")
        manager = RedisStateManager()

        result = await manager.get_state("any_session")
        assert result is None
        assert "Error getting state" in caplog.text
        assert "Redis connection failed!" in caplog.text

    @pytest.mark.asyncio
    async def test_redis_connection_error_save(self, mock_aioredis_client, caplog):
        _, mock_redis = mock_aioredis_client
        mock_redis.set.side_effect = Exception("Redis write error!")
        manager = RedisStateManager()

        await manager.save_state("any_session", {"data":"test"})
        assert "Error saving state" in caplog.text
        assert "Redis write error!" in caplog.text

    @pytest.mark.asyncio
    async def test_redis_close_connection(self, mock_aioredis_client):
        _, mock_redis = mock_aioredis_client
        manager = RedisStateManager()
        await manager.close()
        mock_redis.close.assert_called_once()
