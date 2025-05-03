import pytest
import json
from unittest.mock import patch, MagicMock
from dnddice import DiceRollState
import dnddice
from langgraph.graph import StateGraph

class TestGraphWorkflow:
    """Test the complete LangGraph workflow."""
    
    @patch('dnddice.dnddice.OllamaAgent')
    @patch('random.randint')
    def test_end_to_end_initiative(self, mock_randint, mock_ollama):
        """Test end-to-end workflow for an initiative roll."""
        # Setup mocks
        mock_randint.return_value = 17  # Fixed dice roll
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = '{"dice_type": "d20", "count": 1, "context": "initiative roll"}'
        mock_ollama.return_value = mock_instance
        
        # Create a test graph
        workflow = StateGraph(DiceRollState)
        workflow.add_node("determine_roll", dnddice.determine_roll)
        workflow.add_node("execute_tool", dnddice.execute_tool)
        workflow.add_node("format_result", dnddice.format_result)
        workflow.set_entry_point("determine_roll")
        workflow.add_edge("determine_roll", "execute_tool")
        workflow.add_edge("execute_tool", "format_result")
        workflow.set_finish_point("format_result")
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            "input": "Roll for initiative",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Execute the workflow
        result = app.invoke(initial_state)
        
        # Verify expected results
        assert result["dice_type"] == "d20"
        assert result["context"] == "initiative roll"
        assert result["roll_result"]["dice_type"] == "d20"
        assert result["roll_result"]["rolls"] == [17]
        assert result["roll_result"]["total"] == 17
        assert "17" in result["output"]
        assert "initiative" in result["output"].lower() or "combat" in result["output"].lower()
    
    @patch('dnddice.dnddice.OllamaAgent')
    @patch('random.randint')
    def test_end_to_end_attack(self, mock_randint, mock_ollama):
        """Test end-to-end workflow for an attack roll."""
        # Setup mocks
        mock_randint.return_value = 20  # Critical hit
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = '{"dice_type": "d20", "count": 1, "context": "attack roll"}'
        mock_ollama.return_value = mock_instance
        
        # Create a test graph
        workflow = StateGraph(DiceRollState)
        workflow.add_node("determine_roll", dnddice.determine_roll)
        workflow.add_node("execute_tool", dnddice.execute_tool)
        workflow.add_node("format_result", dnddice.format_result)
        workflow.set_entry_point("determine_roll")
        workflow.add_edge("determine_roll", "execute_tool")
        workflow.add_edge("execute_tool", "format_result")
        workflow.set_finish_point("format_result")
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            "input": "I attack the dragon with my sword",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Execute the workflow
        result = app.invoke(initial_state)
        
        # Verify expected results
        assert result["dice_type"] == "d20"
        assert result["context"] == "attack roll"
        assert result["roll_result"]["dice_type"] == "d20"
        assert result["roll_result"]["rolls"] == [20]
        assert result["roll_result"]["total"] == 20
        assert result["roll_result"]["critical_hit"] == True
        assert "20" in result["output"]
        assert "Critical Hit" in result["output"]
    
    @patch('dnddice.dnddice.OllamaAgent')
    @patch('random.randint')
    def test_end_to_end_damage(self, mock_randint, mock_ollama):
        """Test end-to-end workflow for a damage roll."""
        # Setup different dice values for different calls
        mock_randint.side_effect = [6, 8, 5]  # Three dice values
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = '{"dice_type": "d8", "count": 3, "context": "damage roll longsword"}'
        mock_ollama.return_value = mock_instance
        
        # Create a test graph
        workflow = StateGraph(DiceRollState)
        workflow.add_node("determine_roll", dnddice.determine_roll)
        workflow.add_node("execute_tool", dnddice.execute_tool)
        workflow.add_node("format_result", dnddice.format_result)
        workflow.set_entry_point("determine_roll")
        workflow.add_edge("determine_roll", "execute_tool")
        workflow.add_edge("execute_tool", "format_result")
        workflow.set_finish_point("format_result")
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            "input": "Roll damage for my longsword",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Execute the workflow
        result = app.invoke(initial_state)
        
        # Verify expected results
        assert result["dice_type"] == "d8"
        assert "damage" in result["context"]
        assert "longsword" in result["context"]
        assert result["roll_result"]["dice_type"] == "d8"
        assert len(result["roll_result"]["rolls"]) == 3
        assert result["roll_result"]["total"] == 19  # 6 + 8 + 5
        assert "damage" in result["output"].lower()
    
    @patch('dnddice.dnddice.OllamaAgent')
    def test_error_handling(self, mock_ollama):
        """Test error handling in the workflow."""
        # Setup the mock to raise an exception
        mock_instance = MagicMock()
        # Instead of making the mock always raise an exception, let's make it return an invalid response
        # that will be handled by our error handling code
        mock_instance.get_completion.return_value = "This is not valid JSON"
        mock_ollama.return_value = mock_instance
        
        # Create a test graph
        workflow = StateGraph(DiceRollState)
        workflow.add_node("determine_roll", dnddice.determine_roll)
        workflow.add_node("execute_tool", dnddice.execute_tool)
        workflow.add_node("format_result", dnddice.format_result)
        workflow.set_entry_point("determine_roll")
        workflow.add_edge("determine_roll", "execute_tool")
        workflow.add_edge("execute_tool", "format_result")
        workflow.set_finish_point("format_result")
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            "input": "Roll for something",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Execute the workflow - it should not crash despite the invalid LLM response
        result = app.invoke(initial_state)
        
        # Verify fallback behavior
        assert result["dice_type"] == "d20"  # Default die
        assert result["context"] == "general roll"
        assert result["roll_result"] is not None  # Should still have a roll
    
    @patch('dnddice.dnddice.OllamaAgent')
    @patch('random.randint')
    def test_unknown_tool(self, mock_randint, mock_ollama):
        """Test workflow behavior with an unknown tool."""
        # Setup mocks
        mock_randint.return_value = 10
        mock_instance = MagicMock()
        # Return a dice type that doesn't exist
        mock_instance.get_completion.return_value = '{"dice_type": "d1000", "count": 1, "context": "special roll"}'
        mock_ollama.return_value = mock_instance
        
        # Create a test graph
        workflow = StateGraph(DiceRollState)
        workflow.add_node("determine_roll", dnddice.determine_roll)
        workflow.add_node("execute_tool", dnddice.execute_tool)
        workflow.add_node("format_result", dnddice.format_result)
        workflow.set_entry_point("determine_roll")
        workflow.add_edge("determine_roll", "execute_tool")
        workflow.add_edge("execute_tool", "format_result")
        workflow.set_finish_point("format_result")
        app = workflow.compile()
        
        # Initial state
        initial_state = {
            "input": "Roll a special die",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Execute the workflow
        result = app.invoke(initial_state)
        
        # Verify fallback to d20
        assert result["dice_type"] == "d1000"  # This will come from LLM
        assert result["context"] == "special roll"
        assert result["roll_result"]["dice_type"] == "d20"  # But the actual die rolled is d20 (fallback)
        assert result["roll_result"]["rolls"] == [10] 