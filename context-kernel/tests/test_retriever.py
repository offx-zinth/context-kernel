from app.llm1_retriever import LLM1Retriever

def test_retrieve():
    retriever = LLM1Retriever()
    assert retriever.retrieve("query") == {"result": "mocked retrieval"}