import pytest
import json
from unittest.mock import patch, MagicMock
from dnddice import determine_roll, execute_tool, format_result, DiceRollState

class TestWorkflowNodes:
    """Tests for the individual workflow nodes in the D&D dice roller."""
    
    @patch('dnddice.OllamaAgent')
    def test_determine_roll_d20_attack(self, mock_ollama):
        """Test that determine_roll correctly identifies a d20 attack roll."""
        # Setup mock Ollama response
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = '{"dice_type": "d20", "count": 1, "context": "attack roll"}'
        mock_ollama.return_value = mock_instance
        
        # Initial state
        state = {
            "input": "I attack the goblin with my sword",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Call the function
        result = determine_roll(state)
        
        # Check the LLM was called with the correct input
        mock_instance.get_completion.assert_called_once()
        assert any(state["input"] in str(arg) for arg in mock_instance.get_completion.call_args[0])
        
        # Check the result has the expected values
        assert result["dice_type"] == "d20"
        assert result["context"] == "attack roll"
        assert result["tool_calls"][0]["tool_name"] == "roll_d20"
        assert result["tool_calls"][0]["tool_args"]["count"] == 1
    
    @patch('dnddice.OllamaAgent')
    def test_determine_roll_damage(self, mock_ollama):
        """Test that determine_roll correctly identifies a damage roll."""
        # Setup mock Ollama response
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = '{"dice_type": "d8", "count": 2, "context": "damage roll"}'
        mock_ollama.return_value = mock_instance
        
        # Initial state
        state = {
            "input": "I roll damage for my longsword",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Call the function
        result = determine_roll(state)
        
        # Check the result has the expected values
        assert result["dice_type"] == "d8"
        assert result["context"] == "damage roll"
        assert result["tool_calls"][0]["tool_name"] == "roll_d8"
        assert result["tool_calls"][0]["tool_args"]["count"] == 2
    
    @patch('dnddice.OllamaAgent')
    def test_determine_roll_invalid_json(self, mock_ollama):
        """Test that determine_roll handles invalid JSON responses."""
        # Setup mock Ollama response with invalid JSON
        mock_instance = MagicMock()
        mock_instance.get_completion.return_value = 'This is not JSON'
        mock_ollama.return_value = mock_instance
        
        # Initial state
        state = {
            "input": "Something unclear",
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        # Call the function
        result = determine_roll(state)
        
        # Check that it falls back to defaults
        assert result["dice_type"] == "d20"  # Default die
        assert result["context"] == "general roll"  # Default context
        assert result["tool_calls"][0]["tool_name"] == "roll_d20"
        assert result["tool_calls"][0]["tool_args"]["count"] == 1
    
    def test_execute_tool_d20(self):
        """Test execute_tool with a d20 roll."""
        # Initial state with a tool call
        state = {
            "input": "roll for initiative",
            "dice_type": "d20",
            "roll_result": None,
            "context": "initiative",
            "output": "",
            "messages": [],
            "tool_calls": [{
                "tool_name": "roll_d20",
                "tool_args": {"count": 1}
            }],
            "tool_results": []
        }
        
        # Patch the random roll to get a deterministic result
        with patch('random.randint', return_value=15):
            result = execute_tool(state)
        
        # Check the result
        assert len(result["tool_results"]) == 1
        assert result["roll_result"]["dice_type"] == "d20"
        assert result["roll_result"]["rolls"] == [15]
        assert result["roll_result"]["total"] == 15
    
    def test_execute_tool_multiple_dice(self):
        """Test execute_tool with multiple dice."""
        # Initial state with a tool call for multiple dice
        state = {
            "input": "roll 3d6 for fireball damage",
            "dice_type": "d6",
            "roll_result": None,
            "context": "damage",
            "output": "",
            "messages": [],
            "tool_calls": [{
                "tool_name": "roll_d6",
                "tool_args": {"count": 3}
            }],
            "tool_results": []
        }
        
        # Patch the random roll to get a deterministic result
        with patch('random.randint', return_value=4):
            result = execute_tool(state)
        
        # Check the result
        assert len(result["tool_results"]) == 1
        assert result["roll_result"]["dice_type"] == "d6"
        assert result["roll_result"]["rolls"] == [4, 4, 4]
        assert result["roll_result"]["total"] == 12
        assert result["roll_result"]["count"] == 3
    
    def test_format_result_attack(self):
        """Test format_result for an attack roll."""
        # Initial state after dice roll
        state = {
            "input": "I attack the goblin",
            "dice_type": "d20",
            "roll_result": {
                "rolls": [18],
                "total": 18,
                "dice_type": "d20",
                "count": 1,
                "critical_hit": False,
                "critical_fail": False
            },
            "context": "attack roll",
            "output": "",
            "messages": [],
            "tool_calls": [{
                "tool_name": "roll_d20",
                "tool_args": {"count": 1}
            }],
            "tool_results": [{
                "rolls": [18],
                "total": 18,
                "dice_type": "d20",
                "count": 1,
                "critical_hit": False,
                "critical_fail": False
            }]
        }
        
        result = format_result(state)
        
        # Check output formatting
        assert "attack" in result["output"].lower() or "hit" in result["output"].lower()
        assert "18" in result["output"]
    
    def test_format_result_critical_hit(self):
        """Test format_result for a critical hit."""
        # Initial state with a critical hit
        state = {
            "input": "I attack the dragon",
            "dice_type": "d20",
            "roll_result": {
                "rolls": [20],
                "total": 20,
                "dice_type": "d20",
                "count": 1,
                "critical_hit": True,
                "critical_fail": False
            },
            "context": "attack roll",
            "output": "",
            "messages": [],
            "tool_calls": [{
                "tool_name": "roll_d20",
                "tool_args": {"count": 1}
            }],
            "tool_results": [{
                "rolls": [20],
                "total": 20,
                "dice_type": "d20",
                "count": 1,
                "critical_hit": True,
                "critical_fail": False
            }]
        }
        
        result = format_result(state)
        
        # Check for critical hit message
        assert "CRITICAL HIT" in result["output"]
        assert "20" in result["output"]
    
    def test_format_result_damage(self):
        """Test format_result for a damage roll."""
        # Initial state for damage roll
        state = {
            "input": "Roll damage for my greataxe",
            "dice_type": "d12",
            "roll_result": {
                "rolls": [8],
                "total": 8,
                "dice_type": "d12",
                "count": 1
            },
            "context": "damage roll greataxe",
            "output": "",
            "messages": [],
            "tool_calls": [{
                "tool_name": "roll_d12",
                "tool_args": {"count": 1}
            }],
            "tool_results": [{
                "rolls": [8],
                "total": 8,
                "dice_type": "d12",
                "count": 1
            }]
        }
        
        result = format_result(state)
        
        # Check for damage-specific message
        assert "damage" in result["output"].lower()
        assert "8" in result["output"]
        assert "greataxe" in result["output"].lower()
    
    def test_format_result_multiple_dice(self):
        """Test format_result with multiple dice."""
        # Initial state with multiple dice
        state = {
            "input": "I cast fireball, rolling 8d6",
            "dice_type": "d6",
            "roll_result": {
                "rolls": [3, 5, 1, 6, 4, 2, 3, 5],
                "total": 29,
                "dice_type": "d6",
                "count": 8
            },
            "context": "fireball damage",
            "output": "",
            "messages": [],
            "tool_calls": [{
                "tool_name": "roll_d6",
                "tool_args": {"count": 8}
            }],
            "tool_results": [{
                "rolls": [3, 5, 1, 6, 4, 2, 3, 5],
                "total": 29,
                "dice_type": "d6",
                "count": 8
            }]
        }
        
        result = format_result(state)
        
        # Check that the output includes all rolls and the total
        assert "29" in result["output"]  # Total
        assert "[3, 5, 1, 6, 4, 2, 3, 5]" in result["output"] or "3, 5, 1, 6, 4, 2, 3, 5" in result["output"]  # Individual rolls
        assert "8d6" in result["output"] or "8 d6" in result["output"]  # Dice description 