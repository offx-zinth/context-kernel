import asyncio
import json

class MockRedis:
    def __init__(self):
        self._data = {}

    async def get(self, key: str):
        await asyncio.sleep(0.005)  # Simulate async I/O
        value = self._data.get(key)
        if value is not None:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value # Return as is if not JSON (e.g. simple string)
        return None

    async def set(self, key: str, value, expire: int = None): # `expire` is in seconds
        await asyncio.sleep(0.005)  # Simulate async I/O
        if isinstance(value, (dict, list)):
            self._data[key] = json.dumps(value)
        else:
            self._data[key] = str(value)
        # Note: Actual expiration logic is not implemented in this mock
        if expire:
            # print(f"MockRedis: SET {key} with EX {expire}s (expiration not implemented)")
            pass
        return True

    async def delete(self, key: str):
        await asyncio.sleep(0.005)  # Simulate async I/O
        if key in self._data:
            del self._data[key]
            return 1  # Returns number of keys deleted
        return 0

    async def exists(self, key: str):
        await asyncio.sleep(0.005) # Simulate async I/O
        return 1 if key in self._data else 0

    async def incr(self, key: str):
        await asyncio.sleep(0.005) # Simulate async I/O
        current_value = int(self._data.get(key, "0"))
        new_value = current_value + 1
        self._data[key] = str(new_value)
        return new_value

    async def ping(self):
        await asyncio.sleep(0.001)
        return True

    # Utility to inspect data for testing purposes
    def get_all_data(self):
        return self._data

    def clear_all_data(self):
        self._data = {}
