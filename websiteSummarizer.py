from typing import Dict, List, TypedDict, Any
import requests
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph
from ollamaAgent import OllamaAgent
import time

class WebsiteSummarizerAgent:
    """
    An agent that fetches website content and summarizes it using an Ollama model.
    """
    
    def __init__(self, ollama_agent: OllamaAgent = None):
        """
        Initialize the website summarizer agent.
        
        Args:
            ollama_agent: An OllamaAgent instance to use for summarization
        """
        self.ollama_agent = ollama_agent or OllamaAgent(model="llama2")
    
    def fetch_website_content(self, url: str) -> str:
        """
        Fetch and extract the main text content from a website.
        
        Args:
            url: The URL of the website to fetch
            
        Returns:
            The extracted text content
        """
        try:
            print("Fetching website content...")
            start_time = time.time()
            
            # Add http:// prefix if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Fetch the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            fetch_time = time.time() - start_time
            print(f"Website fetched in {fetch_time:.2f} seconds")
            
            # Parse the HTML content
            parse_start = time.time()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            parse_time = time.time() - parse_start
            print(f"Content parsed in {parse_time:.2f} seconds")
            
            return text
        except Exception as e:
            return f"Error fetching website content: {str(e)}"
    
    def summarize_content(self, content: str, max_length: int = 8000) -> str:
        """
        Summarize the website content using the Ollama agent.
        
        Args:
            content: The website content to summarize
            max_length: Maximum length of content to send to the model
            
        Returns:
            A summary of the content
        """
        print("Summarizing content...")
        start_time = time.time()
        
        # Truncate content if it's too long
        if len(content) > max_length:
            content = content[:max_length] + "...[content truncated]"
        
        # Further reduce content size for faster processing
        # The summarization is taking ~40 seconds because we're sending too much content
        max_efficient_length = 4000  # Reduced from 8000 for faster processing
        if len(content) > max_efficient_length:
            content = content[:max_efficient_length] + "...[content truncated for efficiency]"
        
        # Create a prompt for summarization with more specific instructions
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes web content accurately and concisely. Focus on extracting key information only."},
            {"role": "user", "content": f"Please provide a brief summary of the following website content. Focus only on the main points and key information:\n\n{content}"}
        ]
        
        # Get the summary from the Ollama agent
        summary = self.ollama_agent.get_completion(messages)
        
        summarize_time = time.time() - start_time
        print(f"Content summarized in {summarize_time:.2f} seconds")
        
        return summary
    
    def langgraph_node(self):
        """
        Create a LangGraph-compatible node function that uses this agent.
        
        Returns:
            A function that can be used as a LangGraph node
        """
        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            url = state["url"]
            
            # Fetch website content
            content = self.fetch_website_content(url)
            state["content"] = content
            
            # Summarize the content
            summary = self.summarize_content(content)
            state["summary"] = summary
            
            return state
        
        return node_function


# Define our state type
class SummarizerState(TypedDict):
    url: str
    content: str
    summary: str


def get_user_input(state: SummarizerState) -> SummarizerState:
    """Get user input for the URL to summarize"""
    print("\nWebsite Summarizer")
    print("Enter a URL to summarize or 'exit' to quit")
    
    user_input = input("\nEnter URL: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        exit(0)
    
    state["url"] = user_input
    return state


def display_results(state: SummarizerState) -> SummarizerState:
    """Display the summary results"""
    print("\n=== Website Summary ===")
    print(f"URL: {state['url']}")
    print("\nSummary:")
    print(state["summary"])
    print("\n" + "="*30)
    return state


# Example usage as main
if __name__ == "__main__":
    # Create agents
    ollama_agent = OllamaAgent(model="llama2")
    summarizer_agent = WebsiteSummarizerAgent(ollama_agent)
    
    # Create a graph
    workflow = StateGraph(SummarizerState)
    
    # Add nodes
    workflow.add_node("get_input", get_user_input)
    workflow.add_node("summarize_website", summarizer_agent.langgraph_node())
    workflow.add_node("display_results", display_results)
    
    # Create edges
    workflow.set_entry_point("get_input")
    workflow.add_edge("get_input", "summarize_website")
    workflow.add_edge("summarize_website", "display_results")
    workflow.add_edge("display_results", "get_input")
    
    # Compile the graph
    app = workflow.compile()
    
    # Initialize the state
    initial_state = {
        "url": "",
        "content": "",
        "summary": ""
    }
    
    # Run the graph
    app.invoke(initial_state)
