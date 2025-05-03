import random
import json
import re
from typing import Dict, List, TypedDict, Any, Callable, Tuple
from langgraph.graph import StateGraph
from ollamaAgent import OllamaAgent

class DiceRollState(TypedDict):
    input: str
    dice_type: str
    roll_result: Any  # Can be int or list of ints for multiple dice
    context: str      # Game context (attack, initiative, damage, etc.)
    output: str
    messages: List[Any]  # For tool node compatibility (using langchain message types)
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

class DiceTools:
    """
    A collection of tools for rolling different types of dice in tabletop RPGs.
    Each tool represents a different die (d2, d4, d6, d8, d10, d12, d20, d100).
    """
    
    @staticmethod
    def roll_d2(count: int = 1) -> str:
        """Roll a 2-sided die (coin flip)"""
        rolls = [random.randint(1, 2) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d2",
            "count": count
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d4(count: int = 1) -> str:
        """Roll a 4-sided die"""
        rolls = [random.randint(1, 4) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d4",
            "count": count
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d6(count: int = 1) -> str:
        """Roll a 6-sided die"""
        rolls = [random.randint(1, 6) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d6",
            "count": count
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d8(count: int = 1) -> str:
        """Roll an 8-sided die"""
        rolls = [random.randint(1, 8) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d8",
            "count": count
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d10(count: int = 1) -> str:
        """Roll a 10-sided die"""
        rolls = [random.randint(1, 10) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d10",
            "count": count
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d12(count: int = 1) -> str:
        """Roll a 12-sided die"""
        rolls = [random.randint(1, 12) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d12",
            "count": count
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d20(count: int = 1) -> str:
        """Roll a 20-sided die"""
        rolls = [random.randint(1, 20) for _ in range(count)]
        total = sum(rolls)
        
        # Check for critical hits/fails
        has_crit_hit = 20 in rolls
        has_crit_fail = 1 in rolls
        
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d20",
            "count": count,
            "critical_hit": has_crit_hit,
            "critical_fail": has_crit_fail
        }
        return json.dumps(result)
    
    @staticmethod
    def roll_d100(count: int = 1) -> str:
        """Roll a 100-sided die (percentile dice)"""
        rolls = [random.randint(1, 100) for _ in range(count)]
        total = sum(rolls)
        result = {
            "rolls": rolls,
            "total": total,
            "dice_type": "d100",
            "count": count
        }
        return json.dumps(result)

def determine_roll(state: DiceRollState) -> DiceRollState:
    """
    Use the LLM to determine which dice to roll based on the user's input.
    """
    ollama_agent = OllamaAgent(model="llama2")
    
    system_message = """
    You are a tabletop RPG assistant that helps determine which dice to roll based on player commands.
    Analyze the player's input and determine:
    1. Which type of die to roll (d2, d4, d6, d8, d10, d12, d20, d100)
    2. How many dice to roll
    3. The context of the roll (attack, damage, initiative, saving throw, etc.)
    
    Common scenarios:
    - Initiative rolls use d20
    - Attack rolls use d20
    - Damage rolls depend on the weapon (daggers: d4, swords: d6-d8, greataxes: d12, etc.)
    - Ability checks use d20
    - Saving throws use d20
    
    Respond in JSON format with these fields:
    {
        "dice_type": "d20", // one of: d2, d4, d6, d8, d10, d12, d20, d100
        "count": 1, // number of dice to roll
        "context": "attack roll" // brief description of the roll context
    }
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": state["input"]}
    ]
    
    response = ollama_agent.get_completion(messages)
    
    # Extract JSON from response
    try:
        # Find JSON pattern in the response
        json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            roll_info = json.loads(json_str)
            
            state["dice_type"] = roll_info.get("dice_type", "d20")
            state["context"] = roll_info.get("context", "general roll")
            
            # Prepare tool call
            tool_name = f"roll_{state['dice_type']}"
            count = roll_info.get("count", 1)
            
            # Create a simple tool call for our custom execute_tool function
            state["tool_calls"] = [{
                "tool_name": tool_name,
                "tool_args": {"count": count}
            }]
            
        else:
            # Fallback if JSON parsing fails
            state["dice_type"] = "d20"
            state["context"] = "general roll"
            
            state["tool_calls"] = [{
                "tool_name": "roll_d20",
                "tool_args": {"count": 1}
            }]
    
    except Exception as e:
        # Default fallback
        state["dice_type"] = "d20"
        state["context"] = "general roll"
        
        state["tool_calls"] = [{
            "tool_name": "roll_d20",
            "tool_args": {"count": 1}
        }]
    
    return state

def execute_tool(state: DiceRollState) -> DiceRollState:
    """
    Execute the tool call specified in the state.
    """
    if not state.get("tool_calls"):
        # No tool calls to execute
        state["roll_result"] = None
        return state
    
    # Get the first tool call (we only support one at a time for now)
    tool_call = state["tool_calls"][0]
    tool_name = tool_call.get("tool_name", "")
    tool_args = tool_call.get("tool_args", {})
    
    # Map of tool names to their implementation methods
    tool_map = {
        "roll_d2": DiceTools.roll_d2,
        "roll_d4": DiceTools.roll_d4,
        "roll_d6": DiceTools.roll_d6,
        "roll_d8": DiceTools.roll_d8,
        "roll_d10": DiceTools.roll_d10,
        "roll_d12": DiceTools.roll_d12,
        "roll_d20": DiceTools.roll_d20,
        "roll_d100": DiceTools.roll_d100
    }
    
    # Default to d20 if the requested tool doesn't exist
    if tool_name not in tool_map:
        print(f"Warning: Unknown tool '{tool_name}'. Defaulting to d20.")
        tool_name = "roll_d20"
    
    # Execute the tool with the given arguments
    count = tool_args.get("count", 1)
    try:
        count = int(count)
    except (ValueError, TypeError):
        count = 1
    
    # Get and execute the tool function
    tool_func = tool_map[tool_name]
    result_json = tool_func(count=count)
    
    # Parse the result
    try:
        result = json.loads(result_json)
        state["roll_result"] = result
    except json.JSONDecodeError:
        # Fallback for any parsing issues
        state["roll_result"] = {"error": "Failed to parse roll result"}
    
    return state

def format_result(state: DiceRollState) -> DiceRollState:
    """
    Format the roll results into a human-readable output.
    """
    # If we don't have a roll result, return early
    if not state.get("roll_result"):
        state["output"] = "No dice were rolled. Please try again with a clearer command."
        return state
    
    roll_result = state["roll_result"]
    context = state.get("context", "general roll")
    dice_type = roll_result.get("dice_type", "")
    
    # Build the output message
    output_parts = []
    
    # Add context-specific intro
    if "attack" in context.lower():
        output_parts.append(f"🎯 Attack Roll ({dice_type}):")
    elif "damage" in context.lower():
        output_parts.append(f"💥 Damage Roll ({dice_type}):")
    elif "initiative" in context.lower():
        output_parts.append(f"⚡ Initiative Roll ({dice_type}):")
    elif "save" in context.lower() or "saving throw" in context.lower():
        output_parts.append(f"🛡️ Saving Throw ({dice_type}):")
    elif "skill" in context.lower() or "check" in context.lower():
        output_parts.append(f"🔍 Skill Check ({dice_type}):")
    else:
        output_parts.append(f"🎲 {context.capitalize()} ({dice_type}):")
    
    # Add the roll details
    rolls = roll_result.get("rolls", [])
    total = roll_result.get("total", 0)
    count = roll_result.get("count", len(rolls))
    
    # Format based on single or multiple dice
    if count == 1:
        roll_value = rolls[0] if rolls else 0
        output_parts.append(f"You rolled a {roll_value}!")
        
        # Add special messaging for d20 crits
        if dice_type == "d20":
            if roll_value == 20:
                output_parts.append("🌟 Critical Hit! 🌟")
            elif roll_value == 1:
                output_parts.append("💀 Critical Fail! 💀")
    else:
        # Format multiple dice rolls
        roll_str = ", ".join(str(r) for r in rolls)
        output_parts.append(f"You rolled: [{roll_str}]")
        output_parts.append(f"Total: {total}")
        
        # Add special messaging for d20 crits in multiple rolls
        if dice_type == "d20":
            crit_hits = rolls.count(20)
            crit_fails = rolls.count(1)
            
            if crit_hits > 0:
                output_parts.append(f"🌟 {crit_hits} Critical Hit{'s' if crit_hits > 1 else ''}! 🌟")
            if crit_fails > 0:
                output_parts.append(f"💀 {crit_fails} Critical Fail{'s' if crit_fails > 1 else ''}! 💀")
    
    # Join all parts with newlines
    state["output"] = "\n".join(output_parts)
    
    return state

if __name__ == "__main__":
    # Example usage
    from langgraph.graph import StateGraph
    
    # Create a simple workflow
    workflow = StateGraph(DiceRollState)
    
    # Add nodes
    workflow.add_node("determine_roll", determine_roll)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("format_result", format_result)
    
    # Add edges
    workflow.add_edge("determine_roll", "execute_tool")
    workflow.add_edge("execute_tool", "format_result")
    
    # Set entry and exit points
    workflow.set_entry_point("determine_roll")
    workflow.set_finish_point("format_result")
    
    # Compile the graph
    app = workflow.compile()
    
    # Test with some example inputs
    test_inputs = [
        "I want to roll for initiative",
        "I attack the goblin with my longsword",
        "I need to roll 3d6 for fire damage",
        "Roll a d20 for my perception check"
    ]
    
    for input_text in test_inputs:
        print(f"\nInput: {input_text}")
        
        state = {
            "input": input_text,
            "dice_type": "",
            "roll_result": None,
            "context": "",
            "output": "",
            "messages": [],
            "tool_calls": [],
            "tool_results": []
        }
        
        result = app.invoke(state)
        print(f"Output: {result['output']}") 