from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from master_client import MasterOrchestrator # <-- Import the new MasterOrchestrator

# Load environment variables from .env file at the very beginning
load_dotenv()

app = FastAPI(title="Unified AI Agent API", root_path="/api/v1")



# Allow all origins for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the master client once.
master_client = MasterOrchestrator()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest) -> Dict[str, List[Dict[str, Any]]]:
    """
    Receives a query, routes it through the MasterOrchestrator, and returns the conversation.
    """
    try:
        # The MasterOrchestrator now handles everything.
        messages = await master_client.process_query(request.query)
        return {"messages": messages}
    except Exception as e:
        # Print the full error for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for load balancer.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)