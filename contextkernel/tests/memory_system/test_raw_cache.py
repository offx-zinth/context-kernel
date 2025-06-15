import asyncio
import unittest
from unittest.mock import AsyncMock, patch
import json

from contextkernel.memory_system.raw_cache import RawCache
from contextkernel.utils.config import RedisConfig
# Assuming redis.exceptions might be needed for error simulation
# import redis.exceptions as redis_exceptions

class TestRawCache(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_redis_client = AsyncMock()
        self.redis_config = RedisConfig(host="mockhost", port=6379, db=0)
        self.cache = RawCache(config=self.redis_config, client=self.mock_redis_client, default_ttl_seconds=3600)

    async def test_boot_ping_success(self):
        self.mock_redis_client.ping.return_value = True
        self.assertTrue(await self.cache.boot())
        self.mock_redis_client.ping.assert_called_once()

    async def test_boot_ping_failure(self):
        self.mock_redis_client.ping.return_value = False
        self.assertFalse(await self.cache.boot())
        self.mock_redis_client.ping.assert_called_once()

    async def test_boot_ping_exception(self):
        self.mock_redis_client.ping.side_effect = Exception("Redis connection failed")
        self.assertFalse(await self.cache.boot())
        self.mock_redis_client.ping.assert_called_once()

    async def test_set_get_simple_value(self):
        key, namespace = "test_key", "test_ns"
        value = "simple_value"
        namespaced_key = f"{namespace}:{key}"

        self.mock_redis_client.set.return_value = True # Simulate successful set
        await self.cache.set(key, value, namespace=namespace)
        self.mock_redis_client.set.assert_called_once_with(namespaced_key, value, ex=self.cache.default_ttl)

        self.mock_redis_client.get.return_value = value.encode('utf-8') # Redis client returns bytes
        retrieved_value = await self.cache.get(key, namespace=namespace)
        self.mock_redis_client.get.assert_called_once_with(namespaced_key)
        self.assertEqual(retrieved_value, value)

    async def test_set_get_complex_value(self):
        key, namespace = "complex_key", "test_ns"
        value = {"data": "some_data", "list": [1, 2, 3]}
        namespaced_key = f"{namespace}:{key}"
        serialized_value = json.dumps(value)

        await self.cache.set(key, value, namespace=namespace)
        self.mock_redis_client.set.assert_called_once_with(namespaced_key, serialized_value, ex=self.cache.default_ttl)

        self.mock_redis_client.get.return_value = serialized_value.encode('utf-8')
        retrieved_value = await self.cache.get(key, namespace=namespace)
        self.assertEqual(retrieved_value, value)

    async def test_get_non_existent_key(self):
        self.mock_redis_client.get.return_value = None
        retrieved_value = await self.cache.get("non_existent_key", "test_ns")
        self.assertIsNone(retrieved_value)

    async def test_delete_key(self):
        key, namespace = "delete_key", "test_ns"
        namespaced_key = f"{namespace}:{key}"
        self.mock_redis_client.delete.return_value = 1 # Simulate one key deleted

        deleted = await self.cache.delete(key, namespace=namespace)
        self.assertTrue(deleted)
        self.mock_redis_client.delete.assert_called_once_with(namespaced_key)

        self.mock_redis_client.delete.return_value = 0 # Simulate key not found for deletion
        deleted_again = await self.cache.delete(key, namespace=namespace)
        self.assertFalse(deleted_again)

    async def test_exists_key(self):
        key, namespace = "exists_key", "test_ns"
        namespaced_key = f"{namespace}:{key}"
        self.mock_redis_client.exists.return_value = 1

        exists = await self.cache.exists(key, namespace=namespace)
        self.assertTrue(exists)
        self.mock_redis_client.exists.assert_called_once_with(namespaced_key)

        self.mock_redis_client.exists.return_value = 0
        exists_not = await self.cache.exists("another_key", namespace=namespace)
        self.assertFalse(exists_not)

    async def test_extend_ttl(self):
        key, namespace = "ttl_key", "test_ns"
        namespaced_key = f"{namespace}:{key}"
        additional_ttl = 1000

        # Simulate key exists and has a current TTL
        self.mock_redis_client.exists.return_value = 1
        self.mock_redis_client.pttl.return_value = 500 * 1000 # 500 seconds in ms
        self.mock_redis_client.expire.return_value = True

        extended = await self.cache.extend_ttl(key, additional_ttl_seconds=additional_ttl, namespace=namespace)
        self.assertTrue(extended)
        self.mock_redis_client.exists.assert_called_once_with(namespaced_key)
        self.mock_redis_client.pttl.assert_called_once_with(namespaced_key)
        self.mock_redis_client.expire.assert_called_once_with(namespaced_key, 500 + additional_ttl)

    async def test_extend_ttl_non_existent_key(self):
        self.mock_redis_client.exists.return_value = 0
        extended = await self.cache.extend_ttl("ghost_key", 1000, "test_ns")
        self.assertFalse(extended)

    async def test_extend_ttl_negative_additional_ttl(self):
        extended = await self.cache.extend_ttl("any_key", -100, "test_ns")
        self.assertFalse(extended)
        self.mock_redis_client.exists.assert_not_called() # Should return early

    async def test_get_multiple_keys(self):
        keys = ["key1", "key2", "key3"]
        namespace = "multi_ns"
        namespaced_keys = [f"{namespace}:{k}" for k in keys]

        # Values returned by mget are bytes or None
        mock_return_values = [
            json.dumps({"id": 1, "data": "value1"}).encode('utf-8'),
            None, # key2 not found
            "simple_value3".encode('utf-8')
        ]
        self.mock_redis_client.mget.return_value = mock_return_values

        expected_results = {
            "key1": {"id": 1, "data": "value1"},
            "key2": None,
            "key3": "simple_value3"
        }

        results = await self.cache.get_multiple(keys, namespace=namespace)
        self.mock_redis_client.mget.assert_called_once_with(namespaced_keys)
        self.assertEqual(results, expected_results)

    async def test_get_multiple_empty_keys(self):
        results = await self.cache.get_multiple([], namespace="test_ns")
        self.assertEqual(results, {})
        self.mock_redis_client.mget.assert_not_called()

    async def test_set_with_specific_ttl(self):
        key, namespace, ttl = "ttl_test_key", "test_ns", 12345
        namespaced_key = f"{namespace}:{key}"
        await self.cache.set(key, "some_value", ttl_seconds=ttl, namespace=namespace)
        self.mock_redis_client.set.assert_called_once_with(namespaced_key, "some_value", ex=ttl)

    async def test_set_non_positive_ttl(self):
        key, namespace = "non_pos_ttl", "test_ns"
        namespaced_key = f"{namespace}:{key}"

        # Test with ttl = 0
        self.mock_redis_client.delete.reset_mock() # Reset delete mock before this specific test path
        result_zero_ttl = await self.cache.set(key, "value", ttl_seconds=0, namespace=namespace)
        self.assertFalse(result_zero_ttl)
        self.mock_redis_client.delete.assert_called_once_with(namespaced_key) # Should try to delete
        self.mock_redis_client.set.assert_not_called() # Should not call set

        # Test with ttl < 0
        self.mock_redis_client.delete.reset_mock()
        result_neg_ttl = await self.cache.set(key, "value", ttl_seconds=-100, namespace=namespace)
        self.assertFalse(result_neg_ttl)
        self.mock_redis_client.delete.assert_called_once_with(namespaced_key)
        self.mock_redis_client.set.assert_not_called()


    async def test_get_redis_error(self):
        # Simulate a Redis error on `get`
        self.mock_redis_client.get.side_effect = Exception("Simulated Redis Error") # Using generic Exception for simplicity

        value = await self.cache.get("error_key", "test_ns")
        self.assertIsNone(value) # Expect None on error
        # Optionally, check logs if logging is implemented for errors

    async def test_set_redis_error(self):
        # Simulate a Redis error on `set`
        self.mock_redis_client.set.side_effect = Exception("Simulated Redis Error")

        success = await self.cache.set("error_key", "value", namespace="test_ns")
        self.assertFalse(success) # Expect False on error

if __name__ == '__main__':
    unittest.main()
