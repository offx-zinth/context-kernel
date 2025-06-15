import asyncio
import logging
import json # For serializing complex types for Redis
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis as RedisClient # Renamed to avoid conflict
from contextkernel.utils.config import RedisConfig

logger = logging.getLogger(__name__)

class RawCache:
    """
    Raw Cache system for holding ephemeral inputs with short TTLs, backed by Redis.
    """

    DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes

    def __init__(self, config: RedisConfig, client: RedisClient, default_ttl_seconds: int = DEFAULT_TTL_SECONDS):
        logger.info("Initializing RawCache system with Redis.")
        self.config = config
        self.client = client # This client is expected to be managed (connected/closed) externally
        self.default_ttl = default_ttl_seconds
        logger.info(f"RawCache initialized with Redis config: host={config.host}, port={config.port}, db={config.db}, default TTL: {self.default_ttl}s")

    async def boot(self):
        """
        Checks connectivity with the Redis service.
        The client is assumed to be connected by the time it's passed to __init__.
        """
        logger.info("RawCache booting up...")
        try:
            if await self.client.ping():
                logger.info("Successfully pinged Redis. RawCache boot complete.")
                return True
            else:
                logger.error("Failed to ping Redis. RawCache boot failed.")
                return False
        except Exception as e:
            logger.error(f"Error during Redis ping: {e}. RawCache boot failed.")
            return False

    async def shutdown(self):
        """
        The Redis client connection is managed externally. This method can be a no-op
        or used for any RawCache-specific cleanup if necessary in the future.
        """
        logger.info("RawCache shutting down. (Redis client connection managed externally)")
        # If the client was created by this class, it should be closed here.
        # e.g., await self.client.close()
        # However, per DI principles, the passed client is managed externally.
        await asyncio.sleep(0.01) # Simulate any brief cleanup
        logger.info("RawCache shutdown complete.")
        return True

    def _get_namespaced_key(self, key: str, namespace: Optional[str] = None) -> str:
        if namespace:
            return f"{namespace}:{key}"
        return key

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, namespace: Optional[str] = None) -> bool:
        """
        Stores a key-value pair in Redis with a TTL.
        Serializes complex Python objects to JSON strings for storage.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        ttl_to_use = ttl_seconds if ttl_seconds is not None else self.default_ttl

        if ttl_to_use <= 0:
            logger.warning(f"TTL for key '{actual_key}' is non-positive ({ttl_to_use}s). Key will not be stored or will be deleted if it exists.")
            # Ensure key is deleted if TTL is non-positive
            await self.client.delete(actual_key)
            return False

        try:
            # Serialize value to string if it's not already bytes or string
            if not isinstance(value, (str, bytes, int, float)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = value

            await self.client.set(actual_key, serialized_value, ex=ttl_to_use)
            logger.info(f"Set key '{actual_key}' in Redis. TTL: {ttl_to_use}s. Value snippet: '{str(serialized_value)[:50]}...'")
            return True
        except Exception as e:
            logger.error(f"Error setting key '{actual_key}' in Redis: {e}")
            return False

    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Retrieves a value from Redis.
        Deserializes JSON strings back to Python objects if applicable.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        try:
            value_bytes = await self.client.get(actual_key)
            if value_bytes is not None:
                logger.info(f"Cache HIT for key '{actual_key}'.")
                value_str = value_bytes.decode('utf-8') # Assuming UTF-8 encoding
                try:
                    # Attempt to deserialize if it looks like JSON
                    if (value_str.startswith('{') and value_str.endswith('}')) or \
                       (value_str.startswith('[') and value_str.endswith(']')):
                        return json.loads(value_str)
                    return value_str # Return as string if not clearly JSON
                except json.JSONDecodeError:
                    return value_str # Return as string if JSON deserialization fails
            else:
                logger.info(f"Cache MISS for key '{actual_key}'.")
                return None
        except Exception as e:
            logger.error(f"Error getting key '{actual_key}' from Redis: {e}")
            return None

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Deletes a key from Redis.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        try:
            result = await self.client.delete(actual_key)
            if result > 0:
                logger.info(f"Deleted key '{actual_key}' from Redis.")
                return True
            else:
                logger.info(f"Attempted to delete non-existent key '{actual_key}'.")
                return False
        except Exception as e:
            logger.error(f"Error deleting key '{actual_key}' from Redis: {e}")
            return False

    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Checks if a key exists in Redis.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        try:
            exists = await self.client.exists(actual_key) > 0
            logger.debug(f"Key '{actual_key}' exists in Redis: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking existence of key '{actual_key}' in Redis: {e}")
            return False

    async def get_multiple(self, keys: List[str], namespace: Optional[str] = None) -> Dict[str, Optional[Any]]:
        """
        Retrieves multiple keys from Redis.
        Returns a dictionary of key-value pairs. Keys not found will have a value of None.
        """
        results: Dict[str, Optional[Any]] = {}
        if not keys:
            logger.info("get_multiple called with no keys.")
            return results

        actual_keys = [self._get_namespaced_key(key, namespace) for key in keys]
        logger.info(f"Getting multiple keys (count: {len(keys)}) with namespace '{namespace}' using MGET. Actual keys: {actual_keys}")

        try:
            values_bytes_list = await self.client.mget(actual_keys)

            for original_key, value_bytes in zip(keys, values_bytes_list):
                if value_bytes is not None:
                    value_str = value_bytes.decode('utf-8')
                    try:
                        if (value_str.startswith('{') and value_str.endswith('}')) or \
                           (value_str.startswith('[') and value_str.endswith(']')):
                            results[original_key] = json.loads(value_str)
                        else:
                            results[original_key] = value_str
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to deserialize JSON for key '{original_key}' (actual: '{self._get_namespaced_key(original_key, namespace)}'). Returning raw string.")
                        results[original_key] = value_str
                else:
                    results[original_key] = None

            logger.info(f"MGET results processed for {len(keys)} keys.")
            return results
        except Exception as e:
            logger.error(f"Error getting multiple keys from Redis using MGET: {e}")
            # Fallback or partial results might be an option, but for now, return None for all on error
            return {key: None for key in keys}

    async def extend_ttl(self, key: str, additional_ttl_seconds: int, namespace: Optional[str] = None) -> bool:
        """
        Extends the TTL of an existing key in Redis using PEXPIRE.
        If additional_ttl_seconds is positive, it sets a new TTL from the current time.
        Redis `EXPIRE` command with a new TTL value effectively extends or shortens it.
        """
        actual_key = self._get_namespaced_key(key, namespace)

        if additional_ttl_seconds <= 0:
            logger.warning(f"Additional TTL is non-positive ({additional_ttl_seconds}s). TTL not extended for key '{actual_key}'.")
            return False

        try:
            # First, check if the key exists.
            if not await self.client.exists(actual_key):
                logger.warning(f"Cannot extend TTL for non-existent key '{actual_key}'.")
                return False

            # Get current TTL in seconds
            current_ttl_ms = await self.client.pttl(actual_key)
            if current_ttl_ms == -2: # Key does not exist
                 logger.warning(f"Cannot extend TTL for non-existent key '{actual_key}' (checked again).")
                 return False
            if current_ttl_ms == -1: # Key exists but has no TTL
                logger.warning(f"Key '{actual_key}' has no TTL. Setting a new TTL.")
                new_ttl_seconds = additional_ttl_seconds
            else:
                new_ttl_seconds = (current_ttl_ms // 1000) + additional_ttl_seconds

            if new_ttl_seconds <= 0:
                logger.warning(f"Calculated new TTL for key '{actual_key}' is non-positive ({new_ttl_seconds}s). Deleting key instead.")
                await self.delete(actual_key, namespace) # Or let it expire naturally if behavior is preferred
                return False

            # Set the new expiration time in seconds from now
            success = await self.client.expire(actual_key, new_ttl_seconds)
            if success:
                logger.info(f"Extended TTL for key '{actual_key}'. New TTL: {new_ttl_seconds}s.")
            else:
                # This case might happen if the key expired between the PTTL check and EXPIRE command.
                logger.warning(f"Failed to extend TTL for key '{actual_key}', possibly expired or type mismatch.")
            return success
        except Exception as e:
            logger.error(f"Error extending TTL for key '{actual_key}' in Redis: {e}")
            return False

    # clear_expired_keys is not needed as Redis handles this automatically.

async def main():
    # Setup basic logging for the example
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- RawCache Example Usage (with Redis) ---")

    # Example: Instantiating with RedisConfig and a Redis client
    # In a real application, AppSettings would provide this config.
    redis_config = RedisConfig(host="localhost", port=6379, db=0)

    # Create a Redis client instance
    # Note: Ensure Redis server is running for this example to work.
    # For production, client creation and management would be more robust.
    try:
        redis_client = RedisClient(host=redis_config.host,
                                   port=redis_config.port,
                                   db=redis_config.db,
                                   password=redis_config.password,
                                   decode_responses=False) # Important: get bytes for manual JSON handling
        await redis_client.ping() # Verify connection
    except Exception as e:
        logger.error(f"Could not connect to Redis for example: {e}. Skipping example execution.")
        return

    cache = RawCache(config=redis_config, client=redis_client, default_ttl_seconds=5)

    if not await cache.boot():
        logger.error("Cache boot failed. Exiting example.")
        await redis_client.close() # Clean up client
        return

    # Test basic set and get
    await cache.set("key1", "value1", namespace="test_ns")
    await cache.set("key2", {"data": "complex", "list": [1, 2, 3]}, ttl_seconds=10, namespace="test_ns")

    val1 = await cache.get("key1", namespace="test_ns")
    logger.info(f"Got key1: {val1}")
    assert val1 == "value1"

    val2 = await cache.get("key2", namespace="test_ns")
    logger.info(f"Got key2: {val2}")
    assert val2 == {"data": "complex", "list": [1, 2, 3]}

    # Test expiration
    await cache.set("short_lived", "will expire soon", ttl_seconds=1, namespace="test_ns")
    logger.info("Waiting for 'short_lived' to expire (1 second)...")
    await asyncio.sleep(1.5)
    expired_val = await cache.get("short_lived", namespace="test_ns")
    logger.info(f"Got short_lived after 1.5s: {expired_val}")
    assert expired_val is None

    # Test exists
    key1_exists = await cache.exists("key1", namespace="test_ns")
    logger.info(f"key1 exists: {key1_exists}")
    assert key1_exists is True

    # Test delete
    await cache.delete("key2", namespace="test_ns")
    key2_exists_after_delete = await cache.exists("key2", namespace="test_ns")
    logger.info(f"key2 exists after delete: {key2_exists_after_delete}")
    assert key2_exists_after_delete is False

    # Test get_multiple
    multi_keys = ["key1", "key2", "non_existent_key"]
    multi_vals = await cache.get_multiple(multi_keys, namespace="test_ns")
    logger.info(f"Get multiple results: {multi_vals}")
    assert multi_vals["key1"] == "value1"
    assert multi_vals["key2"] is None
    assert multi_vals["non_existent_key"] is None

    # Test extend_ttl
    await cache.set("extend_me", "initial_value", ttl_seconds=2, namespace="test_ns")
    extend_success = await cache.extend_ttl("extend_me", additional_ttl_seconds=5, namespace="test_ns")
    logger.info(f"TTL extension for 'extend_me' successful: {extend_success}")
    assert extend_success is True

    logger.info("Waiting 3 seconds to check if 'extend_me' is still there (original 2s + extended 5s)...")
    await asyncio.sleep(3) # Original 2s + 3s = 5s total elapsed. TTL was extended by 5, so it has ~4s left (2+5-3)
    extended_val = await cache.get("extend_me", namespace="test_ns")
    logger.info(f"Value of 'extend_me' after 3s (was extended): {extended_val}")
    assert extended_val == "initial_value"

    logger.info("Waiting another 4 seconds for 'extend_me' to finally expire (total original 2s + extended 5s = 7s)...")
    # Total elapsed time: 3s (previous wait) + 4s (this wait) = 7s. Original TTL was 2s, extended by 5s. So it should expire.
    await asyncio.sleep(4)
    extended_val_final = await cache.get("extend_me", namespace="test_ns")
    logger.info(f"Value of 'extend_me' after 7s: {extended_val_final}")
    assert extended_val_final is None


    # Test setting with non-positive TTL
    set_zero_ttl = await cache.set("zero_ttl_key", "zero ttl", ttl_seconds=0, namespace="test_ns")
    logger.info(f"Set 'zero_ttl_key' with TTL 0 success: {set_zero_ttl}")
    assert not set_zero_ttl # Should fail or not store
    zero_ttl_val = await cache.get("zero_ttl_key", namespace="test_ns")
    logger.info(f"Got 'zero_ttl_key' (should be None): {zero_ttl_val}")
    assert zero_ttl_val is None


    await cache.shutdown()
    # Important: Close the Redis client connection
    await redis_client.close()
    logger.info("--- RawCache Example Usage Complete (with Redis) ---")

if __name__ == "__main__":
    # This example requires a running Redis instance.
    # You might need to install redis first: pip install redis
    # And ensure Redis server is running on localhost:6379 (or configure RedisConfig accordingly)
    #
    # To run this example:
    # 1. Ensure Redis is running.
    # 2. If you are in the root of the project, you might run it as:
    #    python -m contextkernel.memory_system.raw_cache
    #    (This ensures Python resolves the module imports correctly)
    #
    # If you encounter `ModuleNotFoundError` for `contextkernel.utils.config`,
    # ensure your PYTHONPATH is set up correctly to include the project root,
    # or run as a module as shown above.
    asyncio.run(main())
