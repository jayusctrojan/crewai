from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI(title="CrewAI API", version="1.0.0")
security = HTTPBearer()

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")

def verify_api_key(authorization: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key in the Authorization header"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    if authorization.credentials != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key"
        )
    return authorization.credentials

class TaskRequest(BaseModel):
    description: str
    role: str = "AI Assistant"
    goal: str = "Complete the given task effectively"

@app.get("/")
async def root():
    return {"message": "CrewAI API is running on Render!", "note": "Authentication required for protected endpoints"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CrewAI API"}

@app.post("/run-crew")
async def run_crew(request: TaskRequest, api_key: str = Depends(verify_api_key)):
    try:
        # Create agent
        agent = Agent(
            role=request.role,
            goal=request.goal,
            backstory=f"You are an expert {request.role} focused on {request.goal}",
            verbose=True
        )
        
        # Create task
        task = Task(
            description=request.description,
            agent=agent,
            expected_output="A detailed response completing the requested task"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "success": True,
            "result": str(result),
            "agent_role": request.role
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
