from app.context_agent import ContextAgent

def test_detect_intent():
    agent = ContextAgent()
    assert agent.detect_intent("hello") == "intent"