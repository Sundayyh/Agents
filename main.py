import os
import random
import sys
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Rich Console
# Load environment variables from .env file
load_dotenv()

# Initialize Rich Console
console = Console()

# 1. Initialize LLM and RAG database
# Define temperature settings for each agent role
AGENT_TEMPERATURES = {
    "Alice": 1.0,        # Higher temperature for more creative, enthusiastic responses
    "Bob": 0.3,          # Lower temperature for more cautious, conservative responses
    "Cindy": 0.5,        # Moderate temperature for balanced, analytical responses
}

# Create LLM instances with different temperatures for each agent
llm_instances = {}
for role_name, temperature in AGENT_TEMPERATURES.items():
    llm_instances[role_name] = ChatDeepSeek(
        model="deepseek-chat", 
        temperature=temperature, 
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )

# Use a local model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Create a dummy RAG database for demonstration
texts = [
    "The first AI conference was held at Dartmouth College in 1956.",
    "Ada Lovelace is considered the first computer programmer.",
    "Socrates was a Greek philosopher from Athens.",
    "Pirates of the Caribbean is a famous movie series.",
    "The Turing test is a test of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
    "Mars colonization is a hot topic, with many technological and ethical challenges.",
]
vector_store = FAISS.from_texts(texts, embeddings)

# 2. Create a place to store shared session histories
memory_store = {}

def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

def add_message_to_shared_history(session_id: str, role: str, message: str):
    """Add a message to the shared conversation history."""
    history = get_session_history(session_id)
    # Add the agent's response as an AI message with role prefix
    history.add_ai_message(f"【{role}】: {message}")

def add_user_message_to_shared_history(session_id: str, message: str):
    """Add a user message to the shared conversation history."""
    history = get_session_history(session_id)
    history.add_user_message(message)

# 3. Define prompt templates for different roles
role_prompts = {
    "Alice": ChatPromptTemplate.from_messages([
        ("system", """YOU ARE ONLY ALICE. You are NOT Bob or Cindy.

You are Alice, participating in a round-table discussion with two other AI participants:
- Bob: A cautious thinker who points out problems and risks (NOT YOU - ignore any Bob responses in the conversation)
- Cindy: An objective analyst who provides balanced perspectives (NOT YOU - ignore any Cindy responses in the conversation)
- Alice: That's YOU - an enthusiastic, positive thinker who looks for opportunities and hope

YOU ONLY SPEAK AS ALICE. Do not roleplay as the other participants. Do not provide multiple perspectives. Only give YOUR optimistic viewpoint.

CRITICAL: When you see messages marked with 【Bob】or 【Cindy】in the conversation history, those are responses from OTHER participants - DO NOT repeat or simulate their responses. Only respond as yourself - Alice.

Speak naturally as if in a real conversation. NO markdown, bullet points, headings, or lists. Just speak conversationally as an optimistic person would.

Relevant information from our shared knowledge base: {rag_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]),
    "Bob": ChatPromptTemplate.from_messages([
        ("system", """YOU ARE ONLY BOB. You are NOT Alice or Cindy.

You are Bob, participating in a round-table discussion with two other AI participants:
- Alice: An enthusiastic, positive thinker who looks for opportunities (NOT YOU - ignore any Alice responses in the conversation)
- Cindy: An objective analyst who provides balanced perspectives (NOT YOU - ignore any Cindy responses in the conversation)
- Bob: That's YOU - a cautious, risk-focused thinker who identifies problems and limitations

YOU ONLY SPEAK AS BOB. Do not roleplay as the other participants. Do not provide multiple perspectives. Only give YOUR cautious, risk-aware viewpoint.

CRITICAL: When you see messages marked with 【Alice】or 【Cindy】in the conversation history, those are responses from OTHER participants - DO NOT repeat or simulate their responses. Only respond as yourself - Bob.

Speak naturally as if in a real conversation. NO markdown, bullet points, headings, or lists. Just speak conversationally as a cautious, concerned person would.

Relevant information from our shared knowledge base: {rag_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]),
    "Cindy": ChatPromptTemplate.from_messages([
        ("system", """YOU ARE ONLY CINDY. You are NOT Alice or Bob.

You are Cindy, participating in a round-table discussion with two other AI participants:
- Alice: An enthusiastic, positive thinker who looks for opportunities (NOT YOU - ignore any Alice responses in the conversation)
- Bob: A cautious, risk-focused thinker who points out problems (NOT YOU - ignore any Bob responses in the conversation)
- Cindy: That's YOU - an objective, logical analyst who provides balanced, fact-based perspectives

YOU ONLY SPEAK AS CINDY. Do not roleplay as the other participants. Do not provide multiple perspectives. Only give YOUR objective, analytical viewpoint.

CRITICAL: When you see messages marked with 【Alice】or 【Bob】in the conversation history, those are responses from OTHER participants - DO NOT repeat or simulate their responses. Only respond as yourself - Cindy.

Speak naturally as if in a real conversation. NO markdown, bullet points, headings, or lists. Just speak conversationally as an objective analyst would.

Relevant information from our shared knowledge base: {rag_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
}

# Define styles for different roles
ROLE_STYLES = {
    "Alice": "bright_green",    # Alice (optimistic) - green
    "Bob": "bright_red",        # Bob (cautious) - red  
    "Cindy": "bright_blue",     # Cindy (analytical) - blue
    "You": "bold cyan",
}

# 4. Create chains for each role (without individual memory)
agent_chains = {}
for role_name, prompt in role_prompts.items():
    # Create simple chain without individual memory
    chain = prompt | llm_instances[role_name]
    agent_chains[role_name] = chain

def display_temperature_settings():
    """Display current temperature settings for all agents."""
    console.print("\n[bold]Current Temperature Settings:[/bold]")
    for role, temp in AGENT_TEMPERATURES.items():
        style = ROLE_STYLES.get(role, "default")
        console.print(f"  [{style}]{role}[/{style}]: {temp}")
    console.print()

def update_agent_temperature(role_name: str, new_temperature: float):
    """Update the temperature for a specific agent and recreate its LLM instance."""
    if role_name not in AGENT_TEMPERATURES:
        console.print(f"[red]Error: Unknown role '{role_name}'[/red]")
        return False
    
    if not 0.0 <= new_temperature <= 2.0:
        console.print(f"[red]Error: Temperature must be between 0.0 and 2.0[/red]")
        return False
    
    # Update the temperature setting
    AGENT_TEMPERATURES[role_name] = new_temperature
    
    # Recreate the LLM instance with new temperature
    llm_instances[role_name] = ChatDeepSeek(
        model="deepseek-chat", 
        temperature=new_temperature, 
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    
    # Recreate the agent chain with the new LLM instance
    prompt = role_prompts[role_name]
    chain = prompt | llm_instances[role_name]
    agent_chains[role_name] = chain
    
    style = ROLE_STYLES.get(role_name, "default")
    console.print(f"[green]✓[/green] Updated [{style}]{role_name}[/{style}] temperature to {new_temperature}")
    return True

def handle_temperature_command(user_input: str):
    """Handle temperature-related commands."""
    parts = user_input.strip().split()
    
    if len(parts) == 1:  # Just "/temp"
        display_temperature_settings()
        return True
    elif len(parts) == 3:  # "/temp <role> <value>"
        _, role_name, temp_str = parts
        try:
            new_temp = float(temp_str)
            return update_agent_temperature(role_name, new_temp)
        except ValueError:
            console.print(f"[red]Error: Invalid temperature value '{temp_str}'. Must be a number.[/red]")
            return True
    else:
        console.print("[yellow]Usage:[/yellow]")
        console.print("  /temp                    - Show current temperature settings")
        console.print("  /temp <role> <value>     - Set temperature for a specific role")
        console.print("  Example: /temp Alice 1.2")
        console.print(f"  Available roles: {list(AGENT_TEMPERATURES.keys())}")
        return True

def round_table_discussion(session_id: str, user_input: str):
    """
    Round-table discussion controller with shared context and streaming output
    session_id: The session ID to distinguish different conversations
    user_input: The user's input for this round
    """
    # Add user message to shared history
    add_user_message_to_shared_history(session_id, user_input)
    
    # Get all role names
    roles = list(agent_chains.keys())
    
    # Check for mentions
    mentioned_roles = [role for role in roles if f"@{role}" in user_input]
    
    # Prioritize mentioned roles
    response_order = mentioned_roles
    
    # Add remaining roles in random order
    remaining_roles = [role for role in roles if role not in mentioned_roles]
    random.shuffle(remaining_roles)
    response_order.extend(remaining_roles)
    
    console.print(f"This round's speaking order: {response_order}")
    
    # Get RAG context
    rag_context = vector_store.similarity_search(user_input, k=2)
    rag_context_str = "\n".join([doc.page_content for doc in rag_context])
    
    # Get shared conversation history
    shared_history = get_session_history(session_id)
    
    # Let each AI role speak in the determined order with streaming
    for role in response_order:
        try:
            # Create a styled role title
            styled_role = Text(f"【{role}】", style=ROLE_STYLES.get(role, "default"))
            
            # Show thinking indicator
            with console.status(f"[{ROLE_STYLES.get(role, 'default')}]{role} is thinking...", spinner="dots"):
                time.sleep(0.5)  # Brief pause for effect
            
            # Initialize streaming response
            console.print(f"[{ROLE_STYLES.get(role, 'default')}]【{role}】[/{ROLE_STYLES.get(role, 'default')}]", end="")
            console.print()
            
            # Stream the response
            full_response = ""
            try:
                # Use the streaming capability of the chain with shared history
                for chunk in agent_chains[role].stream({
                    "input": user_input, 
                    "rag_context": rag_context_str,
                    "history": shared_history.messages
                }):
                    if hasattr(chunk, 'content') and chunk.content:
                        console.print(chunk.content, end="", style=ROLE_STYLES.get(role, "default"))
                        full_response += chunk.content
                        sys.stdout.flush()  # Force immediate output
                        time.sleep(0.02)  # Small delay for streaming effect
                    
                console.print()  # New line after streaming
                console.print()  # Extra line for spacing
                
                # Add this agent's response to the shared history
                add_message_to_shared_history(session_id, role, full_response)
                
            except Exception as stream_error:
                # Fallback to non-streaming if streaming fails
                console.print(f"[yellow]Streaming failed, using standard output...[/yellow]")
                response = agent_chains[role].invoke({
                    "input": user_input, 
                    "rag_context": rag_context_str,
                    "history": shared_history.messages
                })
                console.print(response.content, style=ROLE_STYLES.get(role, "default"))
                console.print()
                
                # Add this agent's response to the shared history
                add_message_to_shared_history(session_id, role, response.content)
                
        except Exception as e:
            console.print(Panel(f"Error: {str(e)}", title=f"【{role}】", border_style="red"))

def main():
    """Main function to run the interactive discussion with streaming responses."""
    session_id = "round_table_session_001"
    console.print(Panel("Welcome to the round-table discussion with streaming responses!", title="Welcome", border_style="green"))
    
    # Display initial temperature settings
    display_temperature_settings()
    
    console.print("Start the conversation by typing your message.")
    console.print(f"You can mention a role by using @, for example: @Alice, what do you think?")
    console.print(f"Available roles: {list(role_prompts.keys())}")
    console.print()
    console.print("[bold cyan]Special Commands:[/bold cyan]")
    console.print("  /temp                    - Show current temperature settings")
    console.print("  /temp <role> <value>     - Set temperature for a specific role (0.0-2.0)")
    console.print("  exit or quit             - End the discussion")
    console.print()

    while True:
        try:
            user_message = console.input(Text("You: ", style=ROLE_STYLES["You"]))
            if user_message.lower() in ["exit", "quit"]:
                break
            
            # Check if it's a temperature command
            if user_message.startswith("/temp"):
                handle_temperature_command(user_message)
                continue
            
            console.print("-" * 20)
            round_table_discussion(session_id, user_message)
            console.print("-" * 20)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
