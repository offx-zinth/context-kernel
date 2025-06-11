from pydantic import BaseModel

class MemoryChunk(BaseModel):
    id: str
    content: str