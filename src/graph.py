"""LangGraph state, router functions, and graph edges for the agentic RAG pipeline."""

from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from src import nodes

class GraphState(TypedDict, total=False):
    """
    Defines the state structure passed between nodes in the graph.
    Every node will take in this state, edit specific values (like the context or source), and return the updated state
    """
    query: str           # The original user query
    source: str          # The route chosen by the router or the name of the retrieval source
    context: str         # The retrieved documents or web search results
    is_relevant: str     # Binary "Yes" or "No" verdict from the relevance checker
    iteration_count: int # Tracks how many relevance checks have occurred
    prompt: str          # The final augmented prompt for the LLM
    response: str        # The final generated answer


def route_by_source(state: GraphState) -> Literal["Retrieve_QnA", "Retrieve_Device", "Web_Search"]:
    """
    Conditional edge evaluated after the initial Router node.
    This dictates whether the workflow should query the medical Q&A database, the medical devices database, or execute a web search.
    """
    return state["source"]


def route_by_relevance(state: GraphState) -> Literal["Augment", "Web_Search"]:
    """
    Conditional edge evaluated after the relevance checker reviews the retrieved data.
    If the context is relevant, the agent is happy to proceed to generation. 
    If it is not relevant, the agent falls back to the web search node to find better data from the internet[.
    """
    return "Augment" if state["is_relevant"] == "Yes" else "Web_Search"


def build_graph():
    """
    Constructs and compiles the StateGraph for the agentic RAG pipeline.
    """
    workflow = StateGraph(GraphState)

    # Add all the required nodes to the graph
    # These represent the individual functions we defined to interact with databases, APIs, and LLMs
    workflow.add_node("Router", nodes.route_query)
    workflow.add_node("Retrieve_QnA", nodes.retrieve_qna)
    workflow.add_node("Retrieve_Device", nodes.retrieve_device)
    workflow.add_node("Web_Search", nodes.search_web)
    workflow.add_node("Relevance_Checker", nodes.check_relevance)
    workflow.add_node("Augment", nodes.build_prompt)
    workflow.add_node("Generate", nodes.generate_answer)

    # Define the exact flow  between the nodes
    # The workflow will always start at the Router node
    workflow.add_edge(START, "Router")
    
    # The router dynamically decides the next step based on the LLM's routing evaluation.
    workflow.add_conditional_edges(
        "Router",
        route_by_source,
        {"Retrieve_QnA": "Retrieve_QnA", "Retrieve_Device": "Retrieve_Device", "Web_Search": "Web_Search"},
    )
    
    # After a local database retrieves information, the graph routes the output directly to the relevance checker.
    workflow.add_edge("Retrieve_QnA", "Relevance_Checker")
    workflow.add_edge("Retrieve_Device", "Relevance_Checker")
    
    # Web search results also flow into the relevance checker to ensure quality before generation
    workflow.add_edge("Web_Search", "Relevance_Checker")
    
    # The relevance checker dictates whether to proceed to generation or route back to a web search.
    workflow.add_conditional_edges(
        "Relevance_Checker",
        route_by_relevance,
        {"Augment": "Augment", "Web_Search": "Web_Search"},
    )
    
    # Once relevant context is confirmed, the graph augments the prompt and moves to the final generation node.
    workflow.add_edge("Augment", "Generate")
    
    # The graph successfully concludes after the final answer is generated
    workflow.add_edge("Generate", END)

    # Compile the graph into an executable application 
    return workflow.compile()