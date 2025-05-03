import pytest
from unittest.mock import MagicMock, patch
import json
from dnddice import DiceRollState
import dnddice
from langgraph.graph import StateGraph

@pytest.fixture
def empty_state():
    """Fixture that provides an empty DiceRollState."""
    return {
        "input": "",
        "dice_type": "",
        "roll_result": None,
        "context": "",
        "output": "",
        "messages": [],
        "tool_calls": [],
        "tool_results": []
    }

@pytest.fixture
def mock_ollama():
    """Fixture that provides a mocked Ollama agent."""
    with patch('dnddice.dnddice.OllamaAgent') as mock:
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = '{"dice_type": "d20", "count": 1, "context": "general roll"}'
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_dice_rolls():
    """Fixture that provides deterministic dice rolls."""
    with patch('random.randint', return_value=10) as mock:
        yield mock

@pytest.fixture
def test_graph():
    """Fixture that provides a compiled test graph."""
    workflow = StateGraph(DiceRollState)
    workflow.add_node("determine_roll", dnddice.determine_roll)
    workflow.add_node("execute_tool", dnddice.execute_tool)
    workflow.add_node("format_result", dnddice.format_result)
    workflow.set_entry_point("determine_roll")
    workflow.add_edge("determine_roll", "execute_tool")
    workflow.add_edge("execute_tool", "format_result")
    workflow.set_finish_point("format_result")
    return workflow.compile()

@pytest.fixture
def d20_roll_result():
    """Fixture that provides a sample d20 roll result."""
    return {
        "rolls": [15],
        "total": 15,
        "dice_type": "d20",
        "count": 1,
        "critical_hit": False,
        "critical_fail": False
    }

@pytest.fixture
def attack_state(d20_roll_result):
    """Fixture that provides a state after an attack roll."""
    return {
        "input": "I attack the goblin",
        "dice_type": "d20",
        "roll_result": d20_roll_result,
        "context": "attack roll",
        "output": "",
        "messages": [],
        "tool_calls": [{"tool_name": "roll_d20", "tool_args": {"count": 1}}],
        "tool_results": [d20_roll_result]
    } 