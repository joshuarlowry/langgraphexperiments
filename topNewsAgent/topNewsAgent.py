from typing import Dict, List, TypedDict, Any
import requests
from langgraph.graph import StateGraph
from news import HackerNewsAgent
from websiteSummarizer import WebsiteSummarizerAgent
from ollamaAgent import OllamaAgent
import time

class TopNewsState(TypedDict):
    query: str
    news_data: Dict[str, Any]
    summaries: List[Dict[str, Any]]
    markdown_output: str


class TopNewsAgent:
    """
    An agent that fetches top news from Hacker News and summarizes each article.
    """
    
    def __init__(self, news_agent: HackerNewsAgent = None, summarizer_agent: WebsiteSummarizerAgent = None):
        """
        Initialize the top news agent.
        
        Args:
            news_agent: A HackerNewsAgent instance to use for fetching news
            summarizer_agent: A WebsiteSummarizerAgent instance to use for summarizing articles
        """
        self.news_agent = news_agent or HackerNewsAgent()
        self.summarizer_agent = summarizer_agent or WebsiteSummarizerAgent(OllamaAgent(model="llama2"))
    
    def fetch_news(self, state: TopNewsState) -> TopNewsState:
        """Fetch news stories based on the query"""
        print("Fetching news stories...")
        query = state["query"]
        
        # Use the news agent to fetch stories
        news_state = {"query": query, "news_data": {}}
        news_state = self.news_agent.langgraph_node()(news_state)
        
        state["news_data"] = news_state["news_data"]
        return state
    
    def summarize_articles(self, state: TopNewsState) -> TopNewsState:
        """Summarize each article from the news data"""
        print("Summarizing articles...")
        news_data = state["news_data"]
        data_type = news_data["type"]
        summaries = []
        
        if data_type in ["top_stories", "new_stories", "best_stories"]:
            stories = news_data["data"]
            
            for story in stories:
                title = story.get("title", "No title")
                url = story.get("url")
                
                if url:
                    print(f"Summarizing: {title}")
                    
                    # Use the website summarizer to get a summary
                    content = self.summarizer_agent.fetch_website_content(url)
                    summary = self.summarizer_agent.summarize_content(content)
                    
                    summaries.append({
                        "title": title,
                        "url": url,
                        "summary": summary
                    })
                else:
                    # Skip stories without URLs
                    print(f"Skipping story without URL: {title}")
        
        state["summaries"] = summaries
        return state
    
    def format_markdown(self, state: TopNewsState) -> TopNewsState:
        """Format the summaries as Markdown"""
        print("Formatting results as Markdown...")
        summaries = state["summaries"]
        news_data = state["news_data"]
        data_type = news_data["type"].replace("_", " ").title()
        
        markdown = f"# {data_type}\n\n"
        
        for i, item in enumerate(summaries, 1):
            title = item["title"]
            url = item["url"]
            summary = item["summary"]
            
            markdown += f"## [{title}]({url})\n\n"
            markdown += f"{summary}\n\n"
            markdown += "---\n\n"
        
        state["markdown_output"] = markdown
        return state


def get_user_input(state: TopNewsState) -> TopNewsState:
    """Get user input for the news query"""
    print("\nTop News Summarizer")
    print("==================")
    print("Options:")
    print("- 'top' for top stories")
    print("- 'new' for newest stories")
    print("- 'best' for best stories")
    print("- 'exit' to quit")
    
    user_input = input("\nEnter your query: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        exit(0)
    
    state["query"] = user_input
    return state


def display_results(state: TopNewsState) -> TopNewsState:
    """Display the Markdown results"""
    print("\n" + state["markdown_output"])
    return state


def run_news_workflow():
    """Run the top news workflow"""
    # Create agents
    news_agent = HackerNewsAgent()
    ollama_agent = OllamaAgent(model="llama2")
    summarizer_agent = WebsiteSummarizerAgent(ollama_agent)
    top_news_agent = TopNewsAgent(news_agent, summarizer_agent)
    
    # Create a graph
    workflow = StateGraph(TopNewsState)
    
    # Add nodes
    workflow.add_node("get_input", get_user_input)
    workflow.add_node("fetch_news", top_news_agent.fetch_news)
    workflow.add_node("summarize_articles", top_news_agent.summarize_articles)
    workflow.add_node("format_markdown", top_news_agent.format_markdown)
    workflow.add_node("display_results", display_results)
    
    # Create edges
    workflow.set_entry_point("get_input")
    workflow.add_edge("get_input", "fetch_news")
    workflow.add_edge("fetch_news", "summarize_articles")
    workflow.add_edge("summarize_articles", "format_markdown")
    workflow.add_edge("format_markdown", "display_results")
    workflow.add_edge("display_results", "get_input")
    
    # Compile the graph
    app = workflow.compile()
    
    # Initialize the state
    initial_state = {
        "query": "top",
        "news_data": {},
        "summaries": [],
        "markdown_output": ""
    }
    
    # Run the graph
    app.invoke(initial_state)


if __name__ == "__main__":
    run_news_workflow() 