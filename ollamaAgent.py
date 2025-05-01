from typing import Dict, List, Optional, TypedDict, Any
import requests
import json

class OllamaAgent:
    """
    A client for interacting with an Ollama-compatible API server.
    This class provides methods to generate chat completions using the OpenAI-compatible
    endpoints of an Ollama server.
    """
    
    def __init__(self, base_url: str = "http://100.104.93.80:11434", model: str = "llama3"):
        """
        Initialize the Ollama agent.
        
        Args:
            base_url: The base URL of the Ollama API server
            model: The default model to use for completions
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    def chat(self, messages: List[Dict[str, str]], 
             model: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a chat completion using the Ollama API.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            model: The model to use (defaults to the instance model)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The API response as a dictionary
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama API: {str(e)}")
    
    def get_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get just the completion text from a chat request.
        
        Args:
            messages: A list of message dictionaries
            **kwargs: Additional arguments to pass to the chat method
            
        Returns:
            The completion text as a string
        """
        response = self.chat(messages, **kwargs)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def list_models(self) -> List[str]:
        """
        List all available models on the Ollama server.
        
        Returns:
            A list of model names
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error listing models from Ollama API: {str(e)}")
    
    def langgraph_node(self, state_key_in: str = "input", state_key_out: str = "output"):
        """
        Create a LangGraph-compatible node function that uses this agent.
        
        Args:
            state_key_in: The key in the state dictionary to read input from
            state_key_out: The key in the state dictionary to write output to
            
        Returns:
            A function that can be used as a LangGraph node
        """
        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Assume input is either a string or a list of message dicts
            input_data = state[state_key_in]
            
            # Convert string input to a proper message format
            if isinstance(input_data, str):
                messages = [{"role": "user", "content": input_data}]
            else:
                messages = input_data
                
            # Get completion and update state
            completion = self.get_completion(messages)
            state[state_key_out] = completion
            
            return state
        
        return node_function

# Example usage as main
if __name__ == "__main__":
    from langgraph.graph import StateGraph
    from typing import TypedDict
    
    # Define a simple state type
    class State(TypedDict):
        input: str
        output: str
    
    # Create an Ollama agent
    agent = OllamaAgent(model="llama2")
    
    # List available models
    try:
        print("Available models:")
        models = agent.list_models()
        for model in models:
            print(f"- {model}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Create a simple graph
    workflow = StateGraph(State)
    
    # Add the Ollama agent as a node
    workflow.add_node("ollama", agent.langgraph_node())
    
    # Set up the graph
    workflow.set_entry_point("ollama")
    workflow.set_finish_point("ollama")
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the graph with a simple prompt
    state = {"input": "Explain what LangGraph is in one sentence.", "output": ""}
    result = app.invoke(state)
    
    print("Input:", state["input"])
    print("Output:", result["output"])
