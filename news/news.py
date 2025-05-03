from typing import Dict, List, Optional, TypedDict, Any
import requests
from langgraph.graph import StateGraph

class HackerNewsAgent:
    """
    A client for interacting with the Hacker News API.
    This class provides methods to fetch stories and comments from Hacker News.
    """
    
    def __init__(self, base_url: str = "https://hacker-news.firebaseio.com/v0"):
        """
        Initialize the Hacker News agent.
        
        Args:
            base_url: The base URL of the Hacker News API
        """
        self.base_url = base_url.rstrip('/')
    
    def get_top_stories(self, limit: int = 10) -> List[int]:
        """
        Get the IDs of the top stories.
        
        Args:
            limit: Maximum number of stories to return
            
        Returns:
            A list of story IDs
        """
        url = f"{self.base_url}/topstories.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()[:limit]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching top stories: {str(e)}")
    
    def get_new_stories(self, limit: int = 10) -> List[int]:
        """
        Get the IDs of the newest stories.
        
        Args:
            limit: Maximum number of stories to return
            
        Returns:
            A list of story IDs
        """
        url = f"{self.base_url}/newstories.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()[:limit]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching new stories: {str(e)}")
    
    def get_best_stories(self, limit: int = 10) -> List[int]:
        """
        Get the IDs of the best stories.
        
        Args:
            limit: Maximum number of stories to return
            
        Returns:
            A list of story IDs
        """
        url = f"{self.base_url}/beststories.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()[:limit]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching best stories: {str(e)}")
    
    def get_item(self, item_id: int) -> Dict[str, Any]:
        """
        Get an item (story, comment, etc.) by its ID.
        
        Args:
            item_id: The ID of the item to fetch
            
        Returns:
            The item data as a dictionary
        """
        url = f"{self.base_url}/item/{item_id}.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching item {item_id}: {str(e)}")
    
    def get_story_with_comments(self, story_id: int, comment_limit: int = 5) -> Dict[str, Any]:
        """
        Get a story with its top comments.
        
        Args:
            story_id: The ID of the story to fetch
            comment_limit: Maximum number of comments to fetch
            
        Returns:
            A dictionary with the story and its comments
        """
        story = self.get_item(story_id)
        
        # If the story has comments, fetch them
        comments = []
        if "kids" in story and story["kids"]:
            comment_ids = story["kids"][:comment_limit]
            for comment_id in comment_ids:
                comment = self.get_item(comment_id)
                if comment and not comment.get("deleted", False):
                    comments.append(comment)
        
        return {
            "story": story,
            "comments": comments
        }
    
    def langgraph_node(self, state_key_in: str = "query", state_key_out: str = "news_data"):
        """
        Create a LangGraph-compatible node function that uses this agent.
        
        Args:
            state_key_in: The key in the state dictionary to read input from
            state_key_out: The key in the state dictionary to write output to
            
        Returns:
            A function that can be used as a LangGraph node
        """
        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            query = state[state_key_in]
            
            # Default to top stories if no specific query
            if not query or query.lower() == "top":
                story_ids = self.get_top_stories(5)
                stories = [self.get_item(story_id) for story_id in story_ids]
                state[state_key_out] = {
                    "type": "top_stories",
                    "data": stories
                }
            elif query.lower() == "new":
                story_ids = self.get_new_stories(5)
                stories = [self.get_item(story_id) for story_id in story_ids]
                state[state_key_out] = {
                    "type": "new_stories",
                    "data": stories
                }
            elif query.lower() == "best":
                story_ids = self.get_best_stories(5)
                stories = [self.get_item(story_id) for story_id in story_ids]
                state[state_key_out] = {
                    "type": "best_stories",
                    "data": stories
                }
            elif query.isdigit():
                # If the query is a number, treat it as a story ID
                story_id = int(query)
                story_with_comments = self.get_story_with_comments(story_id)
                state[state_key_out] = {
                    "type": "story_with_comments",
                    "data": story_with_comments
                }
            else:
                # Default to top stories with a message
                story_ids = self.get_top_stories(5)
                stories = [self.get_item(story_id) for story_id in story_ids]
                state[state_key_out] = {
                    "type": "top_stories",
                    "data": stories,
                    "message": f"Unrecognized query: '{query}'. Showing top stories instead."
                }
            
            return state
        
        return node_function


# Define our state type for the example
class NewsState(TypedDict):
    query: str
    news_data: Dict[str, Any]
    summary: str


def format_news_data(state: NewsState) -> NewsState:
    """Format the news data into a readable summary"""
    news_data = state["news_data"]
    data_type = news_data["type"]
    formatted_text = ""
    
    if data_type in ["top_stories", "new_stories", "best_stories"]:
        stories = news_data["data"]
        category = data_type.replace("_", " ").title()
        
        formatted_text = f"=== {category} ===\n\n"
        for i, story in enumerate(stories, 1):
            title = story.get("title", "No title")
            url = story.get("url", "No URL")
            score = story.get("score", 0)
            by = story.get("by", "anonymous")
            
            formatted_text += f"{i}. {title}\n"
            formatted_text += f"   Score: {score} | By: {by}\n"
            formatted_text += f"   URL: {url}\n"
            formatted_text += f"   ID: {story.get('id', 'N/A')}\n\n"
    
    elif data_type == "story_with_comments":
        story_data = news_data["data"]
        story = story_data["story"]
        comments = story_data["comments"]
        
        title = story.get("title", "No title")
        url = story.get("url", "No URL")
        score = story.get("score", 0)
        by = story.get("by", "anonymous")
        
        formatted_text = f"=== Story Details ===\n\n"
        formatted_text += f"Title: {title}\n"
        formatted_text += f"Score: {score} | By: {by}\n"
        formatted_text += f"URL: {url}\n\n"
        
        if comments:
            formatted_text += f"=== Top Comments ===\n\n"
            for i, comment in enumerate(comments, 1):
                comment_text = comment.get("text", "No text")
                comment_by = comment.get("by", "anonymous")
                
                formatted_text += f"Comment {i} by {comment_by}:\n"
                formatted_text += f"{comment_text}\n\n"
        else:
            formatted_text += "No comments found for this story.\n"
    
    if "message" in news_data:
        formatted_text += f"\nNote: {news_data['message']}\n"
    
    state["summary"] = formatted_text
    return state


def get_user_input(state: NewsState) -> NewsState:
    """Get user input for the news query"""
    print("\nOptions:")
    print("- 'top' for top stories")
    print("- 'new' for newest stories")
    print("- 'best' for best stories")
    print("- Enter a story ID number for details and comments")
    print("- 'exit' to quit")
    
    user_input = input("\nEnter your query: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        exit(0)
    
    state["query"] = user_input
    return state


def display_results(state: NewsState) -> NewsState:
    """Display the news summary"""
    print("\n" + state["summary"])
    return state


def run_news():
    """Run the news browser workflow"""
    # Create the agent
    news_agent = HackerNewsAgent()
    
    # Create a graph
    workflow = StateGraph(NewsState)
    
    # Add nodes
    workflow.add_node("get_input", get_user_input)
    workflow.add_node("fetch_news", news_agent.langgraph_node())
    workflow.add_node("format_news", format_news_data)
    workflow.add_node("display_results", display_results)
    
    # Create edges
    workflow.set_entry_point("get_input")
    workflow.add_edge("get_input", "fetch_news")
    workflow.add_edge("fetch_news", "format_news")
    workflow.add_edge("format_news", "display_results")
    workflow.add_edge("display_results", "get_input")
    
    # Compile the graph
    app = workflow.compile()
    
    # Initialize the state
    initial_state = {
        "query": "top",
        "news_data": {},
        "summary": ""
    }
    
    # Run the graph
    app.invoke(initial_state)


if __name__ == "__main__":
    run_news() 