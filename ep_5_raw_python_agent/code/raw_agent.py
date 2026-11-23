import os
from dotenv import load_dotenv
from pathlib import Path
from rich import print
from rich.console import Console
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model

load_dotenv(override=True)
MODEL_NAME = "gemini-3.8-flash"
MODEL_PROVIDER = "google_genai"
MAX_STEPS = 5

console = Console()

@tool
def search(query: str) -> str:
    """Look up a fact (mocked for the demo).
    
    Args:
        query: The search keywords or question.
    """
    facts = {
        "population of japan": "Japan's population is about 124 million.",
        "population of india": "India's population is about 1.43 billion.",
    }
    for k, v in facts.items():
        if k in query.lower():
            return v
    return "No result found (mocked search)."

@tool
def calculator(expression: str) -> str:
    """Evaluate basic arithmetic, e.g. '1240000000/2'.
    
    Args:
        expression: The mathematical expression string to evaluate.
    """
    allowed = set("0123456789+-*/().e")
    if not set(expression) <= allowed:
        return "Error: only basic arithmetic allowed."
    try:
        return str(eval(expression))  # safe: restricted charset
    except Exception:
        return "Error: invalid expression."

TOOLS = {t.name: t for t in [search, calculator]}

def run_tool(name: str, args: dict) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'."
    try:
        return fn.invoke(args)
    except Exception as exc:
        return f"Error running {name}: {type(exc).__name__}."

# Initialize model once globally so each message responds quickly without re-connecting
llm = init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=0.3).bind_tools(list(TOOLS.values()))

def run_agent(question: str, messages: list) -> str:
    messages.append(HumanMessage(content=question))

    for step in range(1, MAX_STEPS + 1):
        ai = llm.invoke(messages)
        messages.append(ai)

        # If the model gives a final text answer without needing more tools:
        if not ai.tool_calls:
            return ai.content

        # If the model chose to call tools:
        for tc in ai.tool_calls:
            print(f"[dim yellow]⚡ Tool used:[/dim yellow] [cyan]{tc['name']}[/cyan]({tc['args']})")
            result = run_tool(tc['name'], tc['args'])
            print(f"[dim green]✔ Observation:[/dim green] {result}")
            messages.append(ToolMessage(content=result, tool_call_id=tc['id']))

    return "Stopped: reached the max step limit."

def main():
    print("[bold cyan]═════════════════════════════════════════════════[/bold cyan]")
    print("[bold green]🤖 AI Chatbot Ready! (Type 'exit' or 'quit' to stop)[/bold green]")
    print("[bold cyan]═════════════════════════════════════════════════[/bold cyan]\n")

    # Maintains conversation history across multiple turns like ChatGPT
    messages = [
        SystemMessage(content="You are a helpful and conversational AI assistant like ChatGPT. Answer user questions clearly and concisely. When you need facts or calculations, use your available tools, and then provide a friendly final answer.")
    ]

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("[dim]Goodbye![/dim]")
                break

            response = run_agent(user_input, messages)
            print(f"\n[bold magenta]AI:[/bold magenta] {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n[dim]Session closed.[/dim]")
            break

if __name__ == "__main__":
    main()
