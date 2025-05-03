# LangGraph Experiments

A collection of experimental agents using langchain and langgraph.

## Project Structure

The project is organized into the following modules:

- **dnddice**: A D&D dice roller with LLM-powered context detection
- **ollamaAgent**: A client for interacting with Ollama API
- **topNewsAgent**: An agent that fetches and summarizes top news stories
- **topNewsAgentCache**: An enhanced version of topNewsAgent with SQLite caching for improved performance
- **websiteSummarizer**: An agent that summarizes website content
- **news**: A client for interacting with Hacker News API

## Dependencies

To run these experiments, you'll need the following dependencies:

- `uv` - A fast Python package installer and resolver
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- `ollama` - Local LLM hosting platform
  ```bash
  curl https://ollama.ai/install.sh | sh
  ```

- Python packages (install with uv):
  ```bash
  uv pip install langgraph langchain-core langchain-community beautifulsoup4
  ```

## Running the Project

You can run the main application with:

```bash
python main.py
```

This will present a menu with options to try each module.

## Running Tests

To run tests:

```bash
python -m pytest
```
