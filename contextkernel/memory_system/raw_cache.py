import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Commenting out basicConfig to avoid conflict if MemoryKernel also calls it.
logger = logging.getLogger(__name__)

class RawCache:
    """
    Raw Cache system for holding ephemeral inputs with short TTLs.
    Simulates a Redis-like cache with manual TTL management.
    """

    DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes

    def __init__(self, redis_config: Optional[Dict] = None, default_ttl_seconds: int = DEFAULT_TTL_SECONDS):
        logger.info("Initializing RawCache system (stubbed).")

        # _cache_stub stores: key -> (data, expiration_timestamp, original_ttl_seconds)
        self._cache_stub: Dict[str, Tuple[Any, float, int]] = {}
        self.redis_config = redis_config or {"type": "stubbed_in_memory_dict"}
        self.default_ttl = default_ttl_seconds

        logger.info(f"RawCache initialized with config: {self.redis_config}, default TTL: {self.default_ttl}s")

    async def boot(self):
        """
        Simulates connecting to the cache service (e.g., Redis).
        """
        logger.info("RawCache booting up... (simulating connection)")
        await asyncio.sleep(0.01) # Simulate connection time
        logger.info("RawCache boot complete. Stubbed cache is 'online'.")
        return True

    async def shutdown(self):
        """
        Simulates disconnecting from the cache service.
        """
        logger.info("RawCache shutting down... (simulating disconnection)")
        self._cache_stub.clear() # Clear cache on shutdown for this stub
        await asyncio.sleep(0.01) # Simulate disconnection time
        logger.info("RawCache shutdown complete.")
        return True

    def _get_namespaced_key(self, key: str, namespace: Optional[str] = None) -> str:
        if namespace:
            return f"{namespace}:{key}"
        return key

    def _evict(self, actual_key: str, reason: str = "expired") -> None:
        if actual_key in self._cache_stub:
            expired_data, _, _ = self._cache_stub.pop(actual_key)
            logger.info(f"Evicted key '{actual_key}' from cache due to: {reason}. Value snippet: '{str(expired_data)[:50]}...'")
        else:
            logger.debug(f"Attempted to evict non-existent key '{actual_key}'.")


    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, namespace: Optional[str] = None) -> bool:
        """
        Stores a key-value pair in the cache with a TTL.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        ttl_to_use = ttl_seconds if ttl_seconds is not None else self.default_ttl

        if ttl_to_use <= 0:
            logger.warning(f"TTL for key '{actual_key}' is non-positive ({ttl_to_use}s). Key will not be stored or will be evicted if it exists.")
            if actual_key in self._cache_stub: # Evict if it exists with a bad TTL
                self._evict(actual_key, reason="non_positive_ttl_on_set")
            return False # Indicate operation didn't store the value as expected

        expiration_time = time.time() + ttl_to_use
        self._cache_stub[actual_key] = (value, expiration_time, ttl_to_use)
        logger.info(f"Set key '{actual_key}' in cache. TTL: {ttl_to_use}s. Value snippet: '{str(value)[:50]}...'")
        return True

    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Retrieves a value from the cache if the key exists and is not expired.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        entry = self._cache_stub.get(actual_key)

        if entry:
            data, expiration_time, _ = entry
            if time.time() < expiration_time:
                logger.info(f"Cache HIT for key '{actual_key}'.")
                return data
            else:
                logger.info(f"Cache MISS (expired) for key '{actual_key}'. Evicting.")
                self._evict(actual_key, reason="expired_on_get")
                return None
        else:
            logger.info(f"Cache MISS for key '{actual_key}'.")
            return None

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Deletes a key from the cache.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        if actual_key in self._cache_stub:
            self._evict(actual_key, reason="deleted_by_api") # Use _evict to ensure logging
            # logger.info(f"Deleted key '{actual_key}' from cache.") # _evict logs this
            return True
        else:
            logger.info(f"Attempted to delete non-existent key '{actual_key}'.")
            return False

    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Checks if a non-expired key exists in the cache.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        entry = self._cache_stub.get(actual_key)
        if entry:
            _, expiration_time, _ = entry
            if time.time() < expiration_time:
                logger.debug(f"Key '{actual_key}' exists and is not expired.")
                return True
            else:
                logger.debug(f"Key '{actual_key}' exists but is expired.")
                return False
        logger.debug(f"Key '{actual_key}' does not exist.")
        return False

    async def get_multiple(self, keys: List[str], namespace: Optional[str] = None) -> Dict[str, Optional[Any]]:
        """
        Retrieves multiple keys from the cache.
        Returns a dictionary of key-value pairs. Keys not found or expired will have a value of None.
        """
        results: Dict[str, Optional[Any]] = {}
        logger.info(f"Getting multiple keys (count: {len(keys)}) with namespace '{namespace}'.")
        for key_item in keys: # Renamed to avoid confusion with outer 'key' if it was in scope
            results[key_item] = await self.get(key_item, namespace)
        return results

    async def extend_ttl(self, key: str, additional_ttl_seconds: int, namespace: Optional[str] = None) -> bool:
        """
        Extends the TTL of an existing, non-expired key.
        """
        actual_key = self._get_namespaced_key(key, namespace)
        entry = self._cache_stub.get(actual_key)

        if not entry:
            logger.warning(f"Cannot extend TTL for non-existent key '{actual_key}'.")
            return False

        data, expiration_time, _ = entry # original_ttl is not directly used here for new calculation
        current_time = time.time()

        if current_time >= expiration_time:
            logger.warning(f"Cannot extend TTL for already expired key '{actual_key}'. Evicting.")
            self._evict(actual_key, reason="expired_on_extend_ttl")
            return False

        if additional_ttl_seconds <=0:
            logger.warning(f"Additional TTL is non-positive ({additional_ttl_seconds}s). TTL not extended for key '{actual_key}'.")
            return False

        new_expiration_time = expiration_time + additional_ttl_seconds
        # For the new "original_ttl", it's more accurate to consider it as the new total duration from now
        new_total_ttl_from_now = int(new_expiration_time - current_time)

        self._cache_stub[actual_key] = (data, new_expiration_time, new_total_ttl_from_now)
        logger.info(f"Extended TTL for key '{actual_key}' by {additional_ttl_seconds}s. New expiration in {new_total_ttl_from_now}s.")
        return True

    async def clear_expired_keys(self) -> int:
        """
        Manually iterates through the cache and removes all expired keys.
        Returns the count of cleared keys.
        """
        logger.info("Starting manual scan to clear all expired keys...")
        current_time = time.time()
        # It's important to collect keys to delete first, then delete, to avoid modifying dict during iteration.
        expired_keys_to_evict = [
            key for key, (_, expiration_time, _) in self._cache_stub.items()
            if current_time >= expiration_time
        ]

        for key_to_evict in expired_keys_to_evict:
            self._evict(key_to_evict, reason="cleared_by_manual_scan")

        logger.info(f"Cleared {len(expired_keys_to_evict)} expired keys during manual scan.")
        return len(expired_keys_to_evict)

async def main():
    # Setup basic logging for the example
    if not logger.handlers: # Ensure logger is configured if run standalone
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- RawCache Example Usage ---")
    cache = RawCache(default_ttl_seconds=5)
    await cache.boot()

    await cache.set("key1", "value1", namespace="test_ns")
    await cache.set("key2", {"data": "complex"}, ttl_seconds=10, namespace="test_ns")
    await cache.set("short_lived", "will expire soon", ttl_seconds=1, namespace="test_ns")
    await cache.set("no_ttl_key", "uses default ttl", namespace="other_ns")
    await cache.set("zero_ttl_key", "zero ttl", ttl_seconds=0, namespace="test_ns") # Should not be stored or be immediately invalid
    zero_ttl_val = await cache.get("zero_ttl_key", namespace="test_ns")
    logger.info(f"Got zero_ttl_key: {zero_ttl_val}")
    assert zero_ttl_val is None


    val1 = await cache.get("key1", namespace="test_ns")
    logger.info(f"Got key1: {val1}")
    assert val1 == "value1"

    val_other = await cache.get("no_ttl_key", namespace="other_ns")
    logger.info(f"Got no_ttl_key from other_ns: {val_other}")
    assert val_other == "uses default ttl"

    logger.info("Waiting for 'short_lived' to expire (1 second)...")
    await asyncio.sleep(1.5)
    expired_val = await cache.get("short_lived", namespace="test_ns")
    logger.info(f"Got short_lived after 1.5s: {expired_val}")
    assert expired_val is None

    key1_exists = await cache.exists("key1", namespace="test_ns")
    logger.info(f"key1 exists: {key1_exists}")
    assert key1_exists is True

    short_lived_exists = await cache.exists("short_lived", namespace="test_ns")
    logger.info(f"short_lived exists: {short_lived_exists}")
    assert short_lived_exists is False

    await cache.delete("key2", namespace="test_ns")
    key2_exists_after_delete = await cache.exists("key2", namespace="test_ns")
    logger.info(f"key2 exists after delete: {key2_exists_after_delete}")
    assert key2_exists_after_delete is False

    multi_keys = ["key1", "key2", "non_existent_key", "short_lived"]
    multi_vals = await cache.get_multiple(multi_keys, namespace="test_ns")
    logger.info(f"Get multiple results: {multi_vals}")
    assert multi_vals["key1"] == "value1"
    assert multi_vals["key2"] is None
    assert multi_vals["non_existent_key"] is None
    assert multi_vals["short_lived"] is None

    await cache.set("extend_me", "initial_value", ttl_seconds=2, namespace="test_ns")
    extend_success = await cache.extend_ttl("extend_me", additional_ttl_seconds=5, namespace="test_ns")
    logger.info(f"TTL extension for 'extend_me' successful: {extend_success}")
    assert extend_success is True

    logger.info("Waiting 3 seconds to check if 'extend_me' is still there...")
    await asyncio.sleep(3)
    extended_val = await cache.get("extend_me", namespace="test_ns")
    logger.info(f"Value of 'extend_me' after 3s (was extended): {extended_val}")
    assert extended_val == "initial_value"

    logger.info("Waiting another 4 seconds for 'extend_me' to finally expire (original 2s + extended 5s = 7s total)...")
    await asyncio.sleep(4)
    extended_val_final = await cache.get("extend_me", namespace="test_ns")
    logger.info(f"Value of 'extend_me' after 7s: {extended_val_final}")
    assert extended_val_final is None

    extend_fail_non_existent = await cache.extend_ttl("ghost_key", 5, namespace="test_ns")
    logger.info(f"TTL extension for 'ghost_key' successful: {extend_fail_non_existent}")
    assert extend_fail_non_existent is False

    await cache.set("exp1", "data1", ttl_seconds=1)
    await cache.set("exp2", "data2", ttl_seconds=1)
    await cache.set("keeper", "data3", ttl_seconds=10)
    logger.info("Waiting 1.5 seconds for exp1 and exp2 to expire...")
    await asyncio.sleep(1.5)
    cleared_count = await cache.clear_expired_keys()
    logger.info(f"Cleared {cleared_count} keys via manual scan.")
    assert cleared_count >= 2 # Should be at least exp1 and exp2
    assert await cache.exists("keeper") is True
    assert await cache.exists("exp1") is False

    await cache.shutdown()
    logger.info("--- RawCache Example Usage Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
