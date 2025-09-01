# Multi-Agent Round-Table Discussion

This project implements a multi-agent round-table discussion system using LangChain. It features multiple AI agents with distinct personalities who engage in a conversation moderated by a controller. The discussion is enhanced by a shared RAG (Retrieval-Augmented Generation) database, allowing the agents to access and reference a common knowledge base.

## Features

- **Multiple AI Agents**: The system supports several AI agents, each with a unique personality (e.g., an optimist, a pessimist, a neutral analyst).
- **Dynamic Conversation Flow**: A controller manages the conversation, randomizing the speaking order of the agents in each round.
- **User Interaction**: The user can initiate each round of conversation and can also mention specific agents to have them respond first.
- **Shared RAG Database**: All agents share a RAG database, enabling them to pull in relevant information and provide more context-aware responses.
- **Rich Output**: The conversation is displayed with rich formatting, using different colors and styles for each agent to improve readability.
- **Flexible LLM Integration**: The project is configured to use the DeepSeek API but can be easily adapted to other language models like OpenAI's GPT series.

## Requirements

- Python 3.8+
- `pip` for package management

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    -   Rename the `.env.example` file to `.env`.
    -   Open the `.env` file and add your DeepSeek API key:
        ```
        DEEPSEEK_API_KEY="your_deepseek_api_key_here"
        ```

## Usage

To start the round-table discussion, run the following command:

```bash
python main.py
```

You can then start typing your messages to begin the conversation. To mention an agent and have them speak first, use the `@` symbol followed by their role name (e.g., `@optimist`).

To exit the program, type `exit` or `quit`.

## Project Structure

-   `main.py`: The main script that runs the multi-agent discussion.
-   `.env`: The file where you store your API keys.
-   `.gitignore`: Specifies which files and directories to ignore in the Git repository.
-   `requirements.txt`: A list of all the Python packages required for the project.
-   `README.md`: This file.
