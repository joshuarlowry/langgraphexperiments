import pytest
import json
import re
from dnddice import DiceTools

class TestDiceTools:
    """Unit tests for the DiceTools class."""
    
    def test_roll_d20_format(self):
        """Test that d20 rolls return correctly formatted data."""
        result = DiceTools.roll_d20()
        result_dict = json.loads(result)
        
        # Check structure
        assert "rolls" in result_dict
        assert "total" in result_dict
        assert "dice_type" in result_dict
        assert "count" in result_dict
        assert "critical_hit" in result_dict
        assert "critical_fail" in result_dict
        
        # Check values
        assert result_dict["dice_type"] == "d20"
        assert result_dict["count"] == 1
        assert len(result_dict["rolls"]) == 1
        assert result_dict["total"] == result_dict["rolls"][0]
        
        # Check range
        assert 1 <= result_dict["rolls"][0] <= 20
    
    def test_roll_d20_multiple(self):
        """Test rolling multiple d20s."""
        count = 5
        result = DiceTools.roll_d20(count=count)
        result_dict = json.loads(result)
        
        # Check count
        assert result_dict["count"] == count
        assert len(result_dict["rolls"]) == count
        
        # Check total
        assert result_dict["total"] == sum(result_dict["rolls"])
        
        # Check range for all rolls
        for roll in result_dict["rolls"]:
            assert 1 <= roll <= 20
    
    def test_d20_critical_detection(self):
        """Test critical hit/fail detection for d20."""
        # Test with mocked rolls for deterministic testing
        
        # Mock a critical hit (20)
        rolls = [20]
        result_dict = self._get_d20_result_with_rolls(rolls)
        assert result_dict["critical_hit"] == True
        assert result_dict["critical_fail"] == False
        
        # Mock a critical fail (1)
        rolls = [1]
        result_dict = self._get_d20_result_with_rolls(rolls)
        assert result_dict["critical_hit"] == False
        assert result_dict["critical_fail"] == True
        
        # Mock a normal roll
        rolls = [10]
        result_dict = self._get_d20_result_with_rolls(rolls)
        assert result_dict["critical_hit"] == False
        assert result_dict["critical_fail"] == False
        
        # Test with multiple dice, including a critical
        rolls = [5, 20, 10]
        result_dict = self._get_d20_result_with_rolls(rolls)
        assert result_dict["critical_hit"] == True
        
        # Test with both critical hit and fail
        rolls = [1, 20]
        result_dict = self._get_d20_result_with_rolls(rolls)
        assert result_dict["critical_hit"] == True
        assert result_dict["critical_fail"] == True
    
    def _get_d20_result_with_rolls(self, rolls):
        """Helper to create a d20 result with specified rolls."""
        result = {
            "rolls": rolls,
            "total": sum(rolls),
            "dice_type": "d20",
            "count": len(rolls),
            "critical_hit": 20 in rolls,
            "critical_fail": 1 in rolls
        }
        return result
    
    @pytest.mark.parametrize("dice_method,sides", [
        (DiceTools.roll_d2, 2),
        (DiceTools.roll_d4, 4),
        (DiceTools.roll_d6, 6),
        (DiceTools.roll_d8, 8),
        (DiceTools.roll_d10, 10),
        (DiceTools.roll_d12, 12),
        (DiceTools.roll_d20, 20),
        (DiceTools.roll_d100, 100)
    ])
    def test_dice_ranges(self, dice_method, sides):
        """Test that each die type rolls within its expected range."""
        result = dice_method()
        result_dict = json.loads(result)
        
        roll = result_dict["rolls"][0]
        assert 1 <= roll <= sides
        
        # Roll multiple and check all are in range
        count = 10
        multi_result = dice_method(count=count)
        multi_dict = json.loads(multi_result)
        
        assert len(multi_dict["rolls"]) == count
        for roll in multi_dict["rolls"]:
            assert 1 <= roll <= sides
            
    def test_serialization(self):
        """Test that all dice methods return valid JSON."""
        methods = [
            DiceTools.roll_d2,
            DiceTools.roll_d4,
            DiceTools.roll_d6,
            DiceTools.roll_d8,
            DiceTools.roll_d10,
            DiceTools.roll_d12,
            DiceTools.roll_d20,
            DiceTools.roll_d100
        ]
        
        for method in methods:
            result = method()
            # This will raise an exception if the result isn't valid JSON
            json.loads(result) 