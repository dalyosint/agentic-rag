
import os
from functools import lru_cache

from langchain_tavily import TavilySearch
from openai import OpenAI

from src.data_loader import get_collections

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") # Standardized to a widely available model
N_RESULTS = 3
MAX_RELEVANCE_CHECKS = 3
ROUTES = ("Retrieve_QnA", "Retrieve_Device", "Web_Search")




@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    """Caches the OpenAI client so it is only initialized once per session."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def get_llm_response(prompt: str) -> str:
    """Send a single prompt to the OpenAI chat model and return the text."""
    response = _openai_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def query_qna_db(query: str) -> str:
    """Return the top medical Q&A passages for a query from ChromaDB."""
    results = get_collections()["qna"].query(query_texts=[query], n_results=N_RESULTS)
    return "\n".join(results["documents"][0])


def query_device_db(query: str) -> str:
    """Return the top medical device manual passages for a query from ChromaDB."""
    results = get_collections()["device"].query(query_texts=[query], n_results=N_RESULTS)
    return "\n".join(results["documents"][0])


@lru_cache(maxsize=1)
def _tavily() -> TavilySearch:
    """Caches the Tavily client."""
    return TavilySearch(topic="general", max_results=3)


def web_search(query: str) -> str:
    """Search the live web with Tavily and concatenate the results."""
    result = _tavily().invoke({"query": query})
    items = result.get("results", [])
    return "\n".join(item.get("content", "") for item in items)




def route_query(state: dict) -> dict:
    """
    Analyzes the initial user query and decides which data source is most likely to contain the answer.
    """
    prompt = f"""
    You are a routing agent. Based on the user query, decide where to look for information.

    Options:
    - Retrieve_QnA: if it's about general medical knowledge, symptoms, or treatment.
    - Retrieve_Device: if it's about medical devices, manuals, or instructions.
    - Web_Search: if it's about recent news, brand names, or external data.

    Query: "{state['query']}"

    Respond ONLY with one of: Retrieve_QnA, Retrieve_Device, Web_Search
    """
    raw = get_llm_response(prompt)
    # Safely extract the decision, defaulting to Web Search if the LLM output is malformed
    decision = next((r for r in ROUTES if r.lower() in raw.lower()), "Web_Search")
    print(f"---ROUTER DECISION: {decision}---")
    
    # Updates the state with the chosen route
    return {"source": decision}


def retrieve_qna(state: dict) -> dict:
    """Executes the Q&A database tool and updates the state's context."""
    print("---RETRIEVING FROM MEDICAL Q&A---")
    return {"context": query_qna_db(state["query"]), "source": "Medical Q&A Collection"}


def retrieve_device(state: dict) -> dict:
    """Executes the Device database tool and updates the state's context."""
    print("---RETRIEVING FROM MEDICAL DEVICE MANUALS---")
    return {"context": query_device_db(state["query"]), "source": "Medical Device Manual"}


def search_web(state: dict) -> dict:
    """Executes the Web Search tool and updates the state's context."""
    print("---PERFORMING TAVILY WEB SEARCH---")
    return {"context": web_search(state["query"]), "source": "Web Search"}


def check_relevance(state: dict) -> dict:
    """
    The agentic safety loop. Evaluates whether the context pulled by the 
    retrieval nodes actually contains the answer to the user's query.
    """
    print("---CONTEXT RELEVANCE CHECKER---")
    iteration = state.get("iteration_count", 0) + 1

    prompt = f"""
    Check the context below to see if the context is relevant to the user query or not.
    ####
    Context:
    {state.get('context', '')}
    ####
    User Query: {state['query']}

    Options:
    - Yes: if the context is relevant.
    - No: if the context is not relevant.

    Please answer with only 'Yes' or 'No'.
    """
    verdict = "Yes" if get_llm_response(prompt).lower().startswith("yes") else "No"

    # Prevent the agent from infinitely looping if it can't find good context
    if verdict == "No" and iteration >= MAX_RELEVANCE_CHECKS:
        print("---MAX ITERATIONS REACHED, FORCING 'Yes' to proceed to Generation---")
        verdict = "Yes"

    print(f"---RELEVANCE DECISION: {verdict}---")
    
    # Update state with the verdict and increment the loop counter
    return {"is_relevant": verdict, "iteration_count": iteration}


def build_prompt(state: dict) -> dict:
    """Combines the validated context and the original query into a final prompt."""
    print("---AUGMENT (BUILDING PROMPT)---")
    prompt = f"""
    Answer the following question using the context below.
    Context:
    {state.get('context', '')}
    Question: {state['query']}
    please limit your answer in 50 words.
    """
    return {"prompt": prompt}


def generate_answer(state: dict) -> dict:
    """Passes the fully augmented prompt to the LLM to generate the final response."""
    print("---GENERATE (CALLING LLM)---")
    return {"response": get_llm_response(state["prompt"])}