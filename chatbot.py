from typing import Dict, List, TypedDict, Any
from langgraph.graph import StateGraph
from ollamaAgent import OllamaAgent

# Define our state type
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    response: str

# Create an Ollama agent
agent = OllamaAgent(model="llama2")

# Function to process user input
def get_user_input(state: ChatState) -> ChatState:
    user_input = input("You: ")
    
    # Check for exit command
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        exit(0)
        
    # Add the user message to the conversation history
    state["messages"].append({"role": "user", "content": user_input})
    return state

# Function to generate AI response
def generate_response(state: ChatState) -> ChatState:
    # Get completion from the Ollama agent
    response = agent.get_completion(state["messages"])
    
    # Add the assistant's response to the conversation history
    state["messages"].append({"role": "assistant", "content": response})
    state["response"] = response
    
    # Display the response
    print(f"Assistant: {response}")
    print()  # Add a blank line for readability
    
    return state

# Create the chatbot workflow
def create_chatbot():
    # Initialize the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("get_input", get_user_input)
    workflow.add_node("generate_response", generate_response)
    
    # Create edges
    workflow.set_entry_point("get_input")
    workflow.add_edge("get_input", "generate_response")
    workflow.add_edge("generate_response", "get_input")
    
    # No finish point - this will run in a loop until user exits
    
    # Compile the graph
    return workflow.compile()

if __name__ == "__main__":
    print("Simple CLI Chatbot (type 'exit', 'quit', or 'bye' to end)")
    print("------------------------------------------------------")
    
    # Initialize the state with an empty message history
    initial_state = {
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
        "response": ""
    }
    
    # Create and run the chatbot
    chatbot = create_chatbot()
    chatbot.invoke(initial_state)
