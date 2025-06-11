from app.memory_manager import MemoryManager

def test_fetch():
    mm = MemoryManager()
    assert mm.fetch("key") == {"memory": "fetched"}

def test_save():
    mm = MemoryManager()
    assert mm.save("key", {"data": 1}) is True