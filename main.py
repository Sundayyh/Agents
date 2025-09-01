import os
import random
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

# Load environment variables from .env file
load_dotenv()

# Initialize Rich Console
console = Console()

# 1. Initialize LLM and RAG database
# Use DeepSeek for the language model
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.8, api_key=os.getenv("DEEPSEEK_API_KEY"))

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

# 2. Create a place to store all session histories
memory_store = {}

def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# 3. Define prompt templates for different roles
role_prompts = {
    "optimist": ChatPromptTemplate.from_messages([
        ("system", """你是一个乐观主义者。你总是从积极、阳光的角度看待任何问题，努力寻找希望和机会。说话充满热情。
Relevant information from our shared knowledge base: {rag_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]),
    "pessimist": ChatPromptTemplate.from_messages([
        ("system", """你是一个悲观主义者。你习惯于指出任何计划或想法中的风险、缺陷和潜在问题。说话谨慎而严肃。
Relevant information from our shared knowledge base: {rag_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]),
    "neutral_analyst": ChatPromptTemplate.from_messages([
        ("system", """你是一个冷静的分析师。你力求客观、中立，基于逻辑和事实进行分析，不掺杂个人感情。说话条理清晰。
Relevant information from our shared knowledge base: {rag_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
}

# Define styles for different roles
ROLE_STYLES = {
    "optimist": "bright_green",
    "pessimist": "bright_red",
    "neutral_analyst": "bright_blue",
    "You": "bold cyan",
}

# 4. Create chains for each role and wrap them with memory functionality
agent_chains = {}
for role_name, prompt in role_prompts.items():
    chain = prompt | llm
    agent_chains[role_name] = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

def round_table_discussion(session_id: str, user_input: str):
    """
    Round-table discussion controller
    session_id: The session ID to distinguish different conversations
    user_input: The user's input for this round
    """
    # Configuration object, including the session ID
    config = {"configurable": {"session_id": session_id}}
    
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
    
    # Let each AI role speak in the determined order
    for role in response_order:
        try:
            # Invoke the chain for the specific role
            response = agent_chains[role].invoke(
                {"input": user_input, "rag_context": rag_context_str},
                config=config
            )
            # Use Rich to print the output
            styled_role = Text(f"【{role}】", style=ROLE_STYLES.get(role, "default"))
            panel = Panel(response.content, title=styled_role, border_style=ROLE_STYLES.get(role, "default"))
            console.print(panel)
        except Exception as e:
            console.print(Panel(f"Error: {str(e)}", title=f"【{role}】", border_style="red"))

def main():
    """Main function to run the interactive discussion."""
    session_id = "round_table_session_001"
    console.print(Panel("Welcome to the round-table discussion!", title="Welcome", border_style="green"))
    console.print("Start the conversation by typing your message.")
    console.print(f"You can mention a role by using @, for example: @optimist, what do you think?")
    console.print(f"Available roles: {list(role_prompts.keys())}")
    console.print("Type 'exit' or 'quit' to end the discussion.")

    while True:
        try:
            user_message = console.input(Text("You: ", style=ROLE_STYLES["You"]))
            if user_message.lower() in ["exit", "quit"]:
                break
            
            console.print("-" * 20)
            round_table_discussion(session_id, user_message)
            console.print("-" * 20)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
