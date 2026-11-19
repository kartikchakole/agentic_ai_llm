import os
from pathlib import Path  ## used to read path of file 
from dotenv import load_dotenv  ## load environment variable from .env file
from rich import print   ## display beautiful and format output 
from langchain_core.tools import tool ## Convert a normal Python function into a LangChain tool. 
from langchain_core.messages import HumanMessage, ToolMessage ## Represent different types of messages in a LangChain conversation.
from langchain.chat_models import init_chat_model 

load_dotenv()
MODEL = os.getenv("gemini model", "gemini-3.5-flash")

RESUME_DIR = Path(__file__).parent.parent / "resume" 

@tool
def add(a: float, b: float) -> float:
    """add two number together and return the addition result"""
    return a + b

@tool
def get_weather(city: str) -> str:
    """get the current weather of city name"""
    weather = {"mumbai": "32c, humid", "delhi": "34c, cloudy", "nagpur": "23c, rainy"}
    return f"{city} - {weather.get(city.lower(), f'no weather data for {city}')}"  

@tool
def lookup_user(user_id: str) -> str:
    """Look up a user's name and plan by their id (e.g. 'u1')."""
    users = {"u1": "Aarav (Free Plan)", "u2": "Diya (Free Plan)"}
    return users.get(user_id, f"No user with id '{user_id}'.")      

@tool
def get_resume(name: str) -> str:
    """Read and return stored resume for candidate name (e.g., 'arav')."""
    first_name = name.lower().strip().split()[0]
    file_path = RESUME_DIR / f"{first_name}.txt"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return f"no resume found for {name}"    

TOOLS = {t.name: t for t in [add, get_weather, lookup_user, get_resume]}

def runtool_safely(name: str, args: dict) -> str:
    tool_fn = TOOLS.get(name, None)

    if tool_fn is None:
        return f"error : unknown tool named {name}"

    try:
        return tool_fn.invoke(args)
    except Exception as e:
        return f"error running {name} : {e}"    

def main() -> None:
    llm = init_chat_model(model=MODEL, model_provider="google_genai", temperature=0)
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    question = "can we select arav for python developer in interview"
    message = [HumanMessage(content=question)]

    ai = llm_with_tools.invoke(message)
    message.append(ai)

    if not ai.tool_calls:
        print(f"[magenta]AI:[/magenta] {ai.content}")
        return None

    for tc in ai.tool_calls:
        print(f"[yellow]Tool call:[/yellow] {tc['name']}({tc['args']})")
        result = runtool_safely(tc["name"], tc["args"])
        print(f"[green]Result:[/green] {result}")
        message.append(ToolMessage(content=result, tool_call_id=tc["id"]))    

    final = llm_with_tools.invoke(message)
    print(f"\n[bold magenta]AI:[/bold magenta] {final.content}")

if __name__ == "__main__":
    main()
