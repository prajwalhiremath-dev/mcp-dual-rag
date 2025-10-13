

import os
import json
from mcp.server.fastmcp import FastMCP
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Dict, List

# Import libraries for vector search
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import libraries for hybrid search
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Load environment variables
load_dotenv()

mcp = FastMCP(name="RAGAgent")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY1"))

# --- Configuration ---
# VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store")
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = os.path.join(os.path.dirname(__file__), "hybrid_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Load all components of the Hybrid Index on Server Start ---
print("Loading hybrid search components...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(os.path.join(INDEX_PATH, "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    
    with open(os.path.join(INDEX_PATH, "bm25_index.pkl"), "rb") as f:
        bm25_index = pickle.load(f)
    with open(os.path.join(INDEX_PATH, "documents.pkl"), "rb") as f:
        documents = pickle.load(f)
        
    stop_words = set(stopwords.words('english'))
    print("Hybrid search components loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load hybrid index from '{INDEX_PATH}'.")
    print(f"Please run 'python build_hybrid_index.py' first. Error: {e}")
    vector_store, bm25_index, documents = None, None, None

def hybrid_search(query: str, top_k: int = 3) -> List[Dict]:
    """Performs a weighted hybrid search using BM25 and FAISS."""
    if not all([vector_store, bm25_index, documents]):
        return []

    # 1. BM25 Search (Keyword)
    tokenized_query = [word for word in word_tokenize(query.lower()) if word.isalpha() and word not in stop_words]
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # 2. Vector Search (Semantic)
    vector_results = vector_store.similarity_search_with_relevance_scores(query, k=len(documents))

    # 3. Combine and Re-rank
    combined_scores = []
    vector_scores_map = {res[0].metadata["chunk_id"]: res[1] for res in vector_results}

    for i, doc in enumerate(documents):
        bm25_score = bm25_scores[i]
        vector_score = vector_scores_map.get(doc["metadata"]["chunk_id"], 0)
        
        # Simple weighted combination (can be tuned)
        combined_score = (0.4 * bm25_score) + (0.6 * vector_score)
        combined_scores.append({"doc": doc, "score": combined_score})

    # Sort by the final combined score
    sorted_results = sorted(combined_scores, key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]

# --- Tool 1: NER Agent for RAG ---
@mcp.tool()
def ner_agent_rag(question: str) -> str:
    """Analyzes a question to extract the main subject or entity (e.g., company name, topic)."""
    prompt = f"""
    You are an expert entity extractor. From the user's question, identify the main subject.
    Your output MUST be a single JSON object with one key: "entity".
    User Question: "{question}"
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return json.dumps({"error": f"LLM Error in ner_agent_rag: {e}"})

# --- Tool 2: The Retriever Agent ---
@mcp.tool()
def retriever_agent(search_query: str) -> str:
    """
    Performs a powerful hybrid search to find the most relevant document chunks.
    """
    print(f"Performing hybrid search for query: '{search_query}'")
    try:
        retrieved_docs = hybrid_search(search_query, top_k=3)
        
        if not retrieved_docs:
            return json.dumps({"source": "Hybrid Search", "content": "No relevant information found."})

        context = "\n---\n".join([res["doc"]["text"] for res in retrieved_docs])
        sources = list(set([res["doc"]["metadata"].get("source", "Unknown") for res in retrieved_docs]))
        
        print(f"Found {len(retrieved_docs)} relevant chunks from sources: {sources}")
        return json.dumps({"source": ", ".join(sources), "content": context})
    except Exception as e:
        return json.dumps({"error": f"Error during hybrid search: {e}"})

# --- Tool 3: Grader Agent ---
@mcp.tool()
def grader_agent(question: str, context: str) -> str:
    """Evaluates if the retrieved context is relevant to the user's question."""
    prompt = f"""
    Is the provided context relevant to answer the user's question? Answer with a single word: 'yes' or 'no'.
    User Question: "{question}"
    Context: "{context[:2000]}"
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip().lower()
    except Exception:
        return "no"

# --- Tool 4: Answer Generation Agent ---
@mcp.tool()
def answer_generation_agent(question: str, context: str, source: str) -> str:
    """Generates a final answer based on the provided context."""
    prompt = f"""
    Answer the user's question based *only* on the provided context. Cite your source(s).
    User Question: "{question}"
    Context: "{context}"
    Source(s): {source}
    
    The response shouldn't contain the phrases like "Based on the provided data" or "According to the context" od "According to the provided context"
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error in answer_generation_agent: {e}"

# --- Run Server ---
if __name__ == "__main__":
    print("RAG MCP server is starting...")
    mcp.run(transport="stdio")


