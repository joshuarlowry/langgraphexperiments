"""
Main entry point for LangGraph Experiments.
"""
from ollamaAgent import OllamaAgent
from dnddice import DiceTools, DiceRollState
from websiteSummarizer import WebsiteSummarizerAgent
from topNewsAgent import run_news_workflow
from topNewsAgentCache import run_cached_news_workflow
from news.news import run_news

def main():
    print("LangGraph Experiments")
    print("=====================")
    print("1. Roll Dice (dnddice)")
    print("2. Summarize Website (websiteSummarizer)")
    print("3. Top News (topNewsAgent)")
    print("4. Hacker News Browser (news)")
    print("5. Top News with Caching (topNewsAgentCache)")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        # Example of using DiceTools directly
        result = DiceTools.roll_d20()
        import json
        print(f"D20 roll result: {json.loads(result)['rolls'][0]}")
    elif choice == "2":
        # Run the website summarizer
        from websiteSummarizer.websiteSummarizer import run_summarizer
        run_summarizer()
    elif choice == "3":
        # Run the top news workflow
        run_news_workflow()
    elif choice == "4":
        # Run the hacker news browser
        run_news()
    elif choice == "5":
        # Run the cached top news workflow
        run_cached_news_workflow()
    elif choice == "6":
        print("Goodbye!")
        return
    else:
        print("Invalid choice. Please try again.")
        
    # Recursive call for menu
    main()

if __name__ == "__main__":
    main()
