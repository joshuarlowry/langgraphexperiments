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

def process_tool_results(state: DiceRollState) -> DiceRollState:
    """
    Process tool results from ToolNode format back to our state format
    """
    # Debug info
    print(f"Messages received: {state.get('messages')}")
    
    # Check if we have messages
    if not state.get("messages"):
        state["output"] = "No dice were rolled."
        return state
    
    # Extract tool results from messages
    tool_results = []
    
    # Look for ToolMessage objects in messages
    for message in state.get("messages", []):
        if isinstance(message, ToolMessage):
            try:
                print(f"Found ToolMessage: {message}")
                # Get content from the tool message
                result_data = message.content
                print(f"Tool result data: {result_data}")
                
                if isinstance(result_data, str):
                    # Try to parse JSON response
                    try:
                        processed_data = json.loads(result_data)
                        tool_results.append(processed_data)
                        print(f"Processed JSON data: {processed_data}")
                    except json.JSONDecodeError:
                        # If not valid JSON, use as-is
                        tool_results.append(result_data)
                        print(f"Using raw string data: {result_data}")
                else:
                    # Use any other data type as-is
                    tool_results.append(result_data)
                    print(f"Using non-string data: {result_data}")
                
            except Exception as e:
                print(f"Error processing tool results: {e}")
    
    if tool_results:
        state["tool_results"] = tool_results
        # Keep the most recent roll result
        state["roll_result"] = tool_results[-1]
        print(f"Final tool results: {tool_results}")
    else:
        print("No tool results found in messages")
    
    return state

def format_result(state: DiceRollState) -> DiceRollState:
    """
    Format the roll result with appropriate context for better user experience.
    Always include the actual roll values.
    """
    if not state["tool_results"]:
        state["output"] = "No dice were rolled."
        return state
    
    # Get the most recent tool result
    result = state["tool_results"][-1]
    context = state["context"]
    
    # Extract roll information based on the available data structure
    # This handles both direct object return and string representation
    if isinstance(result, dict):
        # If result is already a dictionary
        dice_type = result.get("dice_type", "")
        rolls = result.get("rolls", [])
        total = result.get("total", 0)
        count = result.get("count", 1)
    elif isinstance(result, str):
        # Try to extract information from string result 
        # (simplistic approach, can be improved)
        import re
        dice_match = re.search(r'd(\d+)', result)
        dice_type = f"d{dice_match.group(1)}" if dice_match else "d?"
        
        # Extract numbers for rolls
        numbers = re.findall(r'\d+', result)
        if numbers:
            total = int(numbers[-1])  # Assume last number is total
            rolls = [int(n) for n in numbers[:-1]] if len(numbers) > 1 else [total]
            count = len(rolls)
        else:
            total = 0
            rolls = []
            count = 0
    else:
        # Unknown format, use placeholder values
        dice_type = state.get("dice_type", "d?")
        rolls = []
        total = 0
        state["output"] = f"Rolled {dice_type} but couldn't interpret the result."
        return state
    
    # Base output always includes the actual roll values
    if count > 1:
        base_output = f"You rolled {count}{dice_type} and got: {rolls} for a total of {total}. "
    else:
        base_output = f"You rolled {dice_type} and got: {total}. "
    
    # Add context-specific information
    if dice_type == "d20":
        has_crit_hit = 20 in rolls if isinstance(rolls, list) else False
        has_crit_fail = 1 in rolls if isinstance(rolls, list) else False
        
        if "initiative" in context.lower():
            state["output"] = base_output
            if total >= 15:
                state["output"] += "You'll likely act early in the combat round!"
            else:
                state["output"] += "You'll take your turn based on this initiative order."
        
        elif "attack" in context.lower():
            state["output"] = base_output
            if has_crit_hit:
                state["output"] += "CRITICAL HIT! You hit and deal double damage!"
            elif has_crit_fail:
                state["output"] += "CRITICAL MISS! Your attack fails spectacularly."
            elif total >= 15:
                state["output"] += "That's likely to hit most enemies!"
            else:
                state["output"] += "This might hit depending on the enemy's armor class."
        
        elif "save" in context.lower() or "saving throw" in context.lower():
            state["output"] = base_output
            if total >= 15:
                state["output"] += "You successfully resist the effect!"
            else:
                state["output"] += "You might be affected depending on the difficulty."
        
        elif "check" in context.lower() or "ability" in context.lower():
            state["output"] = base_output
            if total >= 15:
                state["output"] += "That's a good roll, likely to succeed!"
            else:
                state["output"] += "Success depends on the difficulty of the task."
        
        else:
            state["output"] = base_output
    
    elif "damage" in context.lower():
        weapon_info = ""
        if "dagger" in context.lower():
            weapon_info = "dagger"
        elif "shortsword" in context.lower():
            weapon_info = "shortsword"
        elif "longsword" in context.lower():
            weapon_info = "longsword"
        elif "greatsword" in context.lower():
            weapon_info = "greatsword"
        elif "greataxe" in context.lower():
            weapon_info = "greataxe"
        elif "bow" in context.lower():
            weapon_info = "bow"
        elif "crossbow" in context.lower():
            weapon_info = "crossbow"
        elif "staff" in context.lower():
            weapon_info = "staff"
        elif "mace" in context.lower():
            weapon_info = "mace"
        elif "warhammer" in context.lower():
            weapon_info = "warhammer"
        
        if weapon_info:
            state["output"] = f"You rolled {count}{dice_type} for your {weapon_info} damage and got: {rolls} for a total of {total} damage!"
        else:
            state["output"] = f"You rolled {count}{dice_type} for damage and got: {rolls} for a total of {total} damage!"
    
    else:
        state["output"] = base_output
    
    return state

def execute_tool(state: DiceRollState) -> DiceRollState:
    """Execute the appropriate dice tool based on the determined roll."""
    if not state.get("tool_calls"):
        return state
    
    # Get the first tool call
    tool_call = state["tool_calls"][0]
    tool_name = tool_call.get("tool_name")
    tool_args = tool_call.get("tool_args", {})
    
    # Map tool names to functions
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
    
    if tool_name in tool_map:
        try:
            # Execute the tool directly as a function
            tool_func = tool_map[tool_name]
            count = tool_args.get("count", 1)
            result_json = tool_func(count=count)
            
            # Parse the result
            result = json.loads(result_json)
            
            # Store the result
            state["tool_results"] = [result]
            state["roll_result"] = result
        except Exception as e:
            # Set a default result if tool execution fails
            state["tool_results"] = [{
                "rolls": [0],
                "total": 0,
                "dice_type": state.get("dice_type", "d20"),
                "count": tool_args.get("count", 1),
                "error": str(e)
            }]
    else:
        # Set a default result for unknown tools
        state["tool_results"] = [{
            "rolls": [0],
            "total": 0,
            "dice_type": state.get("dice_type", "d20"),
            "count": tool_args.get("count", 1),
            "error": f"Unknown tool: {tool_name}"
        }]
    
    return state

# Example usage as main
if __name__ == "__main__":
    try:
        # Create a log file for debugging
        import logging
        logging.basicConfig(filename='dnd_dice.log', level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Starting D&D Dice Roller")
        
        # Create an Ollama agent for natural language processing
        logging.info("Creating Ollama agent")
        ollama_agent = OllamaAgent(model="llama2")
        
        # Create a graph that just uses a direct workflow without ToolNode
        logging.info("Creating StateGraph")
        workflow = StateGraph(DiceRollState)
        
        # Add nodes
        workflow.add_node("determine_roll", determine_roll)
        workflow.add_node("execute_tool", execute_tool)
        workflow.add_node("format_result", format_result)
        
        # Create edges for our simplified workflow
        workflow.set_entry_point("determine_roll")
        workflow.add_edge("determine_roll", "execute_tool")
        workflow.add_edge("execute_tool", "format_result")
        
        # Set format_result as the finish point
        workflow.set_finish_point("format_result")
        
        # Compile the graph
        logging.info("Compiling workflow")
        app = workflow.compile()
        
        # Clear terminal and show intro - using simpler ASCII
        print("\n")  # Skip terminal clear for now
        print("\n==== D&D Dice Roller ====")
        print("========================")
        print("Examples:")
        print("  * 'Roll for initiative'")
        print("  * 'I attack the goblin with my longsword'")
        print("  * 'Roll damage for my greataxe'")
        print("  * 'Roll a d20 for perception check'")
        print("  * 'Roll 3d6 for fireball damage'")
        print("========================")
        
        while True:
            try:
                user_input = input("\nWhat would you like to roll? (or 'exit' to quit): ")
                logging.info(f"User input: {user_input}")
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\nFarewell, adventurer! May your next rolls be criticals.")
                    logging.info("User exited")
                    break
                
                # Show "rolling" animation
                import sys
                import time
                
                print("\nRolling", end="", flush=True)
                for _ in range(3):
                    time.sleep(0.3)
                    print(".", end="", flush=True)
                print("\n")
                
                initial_state = {
                    "input": user_input,
                    "dice_type": "",
                    "roll_result": None,
                    "context": "",
                    "output": "",
                    "messages": [],
                    "tool_calls": [],
                    "tool_results": []
                }
                
                logging.info("Invoking app")
                result = app.invoke(initial_state)
                logging.info(f"Result: {result}")
                
                # Ensure we have an output to display, even if something went wrong
                if not result["output"] or result["output"] == "No dice were rolled.":
                    if "roll_result" in result and result["roll_result"]:
                        # We have a result but format_result failed
                        roll_data = result["roll_result"]
                        if isinstance(roll_data, dict):
                            dice = roll_data.get("dice_type", "?")
                            rolls = roll_data.get("rolls", [])
                            total = roll_data.get("total", 0)
                            print(f"You rolled {dice} and got: {rolls} for a total of {total}")
                        else:
                            # Just print whatever we have
                            print(f"Dice roll result: {roll_data}")
                    else:
                        # No result at all
                        print("Sorry, I couldn't roll the dice properly. Please try again with a different wording.")
                else:
                    # Print the formatted output
                    print(f"{result['output']}")
                    
            except KeyboardInterrupt:
                print("\n\nFarewell, adventurer! May your next rolls be criticals.")
                logging.info("KeyboardInterrupt - User exited")
                break
            except Exception as e:
                logging.exception("Error in main loop")
                print(f"\nAn error occurred: {e}")
                print("Please try again.")
    
    except Exception as e:
        import traceback
        with open('dnd_dice_error.log', 'w') as f:
            f.write(f"Error starting application: {e}\n")
            f.write(traceback.format_exc())
        print(f"Error starting application: {e}")
        print("Check dnd_dice_error.log for details.")


