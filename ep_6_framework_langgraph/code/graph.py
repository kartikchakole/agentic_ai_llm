import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from rich import print
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
# Using the model confirmed available on your account
groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file! Please check your .env.")

# Initialize ChatGroq with your active model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=groq_model,
    temperature=0.2
)

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content.strip()}

def build_graph():
    graph = StateGraph(State)

    graph.add_node("answer", answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)

    return graph.compile()

def main():
    agent = build_graph()
    result = agent.invoke({"question": "what is python explain in single sentence"})
    print(f"[bold cyan]Que:[/bold cyan] {result['question']}")
    print(f"[bold magenta]Ans:[/bold magenta] {result['answer']}")

if __name__ == "__main__":
    main()
