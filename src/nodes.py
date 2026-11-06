import os
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb

# Initialize LLM for routing, relevance checks, and generation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path="./chroma_db")
qa_collection = client.get_collection(name="medical_Q_and_A")
md_collection = client.get_collection(name="medical_devices")

# Initialize Tavily web search tool
tavily_search = TavilyAnswer(max_results=3, topic="general")

def retrieve_qa_node(state: dict) -> dict:
    """Retrieves top documents from the Medical Q&A ChromaDB collection."""
    print("LOG: Node -> Retrieving from Medical Q&A DB...")
    query = state["query"]
    results = qa_collection.query(query_texts=[query], n_results=3)
    documents = results.get("documents", [[]])[0]
    
    context = "\n".join(documents)
    state["context"] = context
    state["source"] = "medical_Q_and_A"
    return state

def retrieve_device_node(state: dict) -> dict:
    """Retrieves top documents from the Medical Devices ChromaDB collection."""
    print("LOG: Node -> Retrieving from Medical Devices DB...")
    query = state["query"]
    results = md_collection.query(query_texts=[query], n_results=3)
    documents = results.get("documents", [[]])[0]
    
    context = "\n".join(documents)
    state["context"] = context
    state["source"] = "medical_devices"
    return state

def web_search_node(state: dict) -> dict:
    """Performs an external web search using Tavily."""
    print("LOG: Node -> Performing Web Search...")
    query = state["query"]
    search_result = tavily_search.invoke({"query": query})
    
    state["context"] = str(search_result)
    state["source"] = "web_search"
    return state

def router_node(state: dict) -> str:
    """Decides which path/tool the agent should use first based on the query."""
    print("LOG: Node -> Routing Query...")
    query = state["query"]
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a routing agent. Based on the user query, decide where to look for information:\n"
                   "1. 'medical_qa' if it's general medical knowledge, symptoms, or treatments.\n"
                   "2. 'medical_device' if it's about medical device manuals, model numbers, or instructions.\n"
                   "3. 'web_search' if it's about recent news, specific brand names, or external data not in standard manuals.\n"
                   "Respond ONLY with one of those three exact strings."),
        ("human", "{query}")
    ])
    
    chain = router_prompt | llm
    decision = chain.invoke({"query": query}).content.strip().lower()
    
    print(f"LOG: Router decided path -> {decision}")
    if "device" in decision:
        return "retrieve_device"
    elif "web" in decision:
        return "web_search"
    else:
        return "retrieve_qa"

def relevance_checker_node(state: dict) -> dict:
    """Checks if the retrieved context is relevant enough to answer the user's query."""
    print("LOG: Node -> Checking Context Relevance...")
    query = state["query"]
    context = state["context"]
    
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine whether the provided context is relevant to the user query.\n"
                   "Respond ONLY with 'yes' or 'no'."),
        ("human", "Query: {query}\n\nContext: {context}")
    ])
    
    chain = relevance_prompt | llm
    result = chain.invoke({"query": query, "context": context}).content.strip().lower()
    
    state["is_relevant"] = "yes" if "yes" in result else "no"
    print(f"LOG: Is context relevant? -> {state['is_relevant']}")
    return state

def generate_response_node(state: dict) -> dict:
    """Generates the final response using the validated context and LLM."""
    print("LOG: Node -> Generating Final Response...")
    query = state["query"]
    context = state["context"]
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert medical assistant. Answer the user query accurately using ONLY the provided context."),
        ("human", "Context:\n{context}\n\nQuery: {query}")
    ])
    
    chain = gen_prompt | llm
    response = chain.invoke({"context": context, "query": query}).content
    
    state["response"] = response
    return state 