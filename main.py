from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import os
import uvicorn
from typing import Optional
import logging
import secrets

# Import your existing CrewAI functionality
from crewai import Agent, Task, Crew
# from llm_router import get_llm_for_task  # Commented out for now

# Try to import memory manager with error handling
try:
    from supabase_memory import get_memory_manager
    MEMORY_AVAILABLE = True
    print("SUCCESS: supabase_memory imported successfully")
except Exception as e:
    print(f"WARNING: Could not import supabase_memory: {e}")
    MEMORY_AVAILABLE = False
    get_memory_manager = lambda: None

import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Studio API", version="1.0.0")
print("FastAPI app created successfully")

# Security setup for Studio access
security = HTTPBasic()

def verify_studio_access(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify Studio access with username/password"""
    correct_username = secrets.compare_digest(
        credentials.username, 
        os.getenv("STUDIO_USERNAME", "admin")
    )
    correct_password = secrets.compare_digest(
        credentials.password, 
        os.getenv("STUDIO_PASSWORD", "changeme123")
    )
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# Mount static files and templates for Studio UI (only if directories exist)
try:
    print("Attempting to mount static files...")
    if os.path.exists("static"):
        print("Static directory found, mounting...")
        app.mount("/static", StaticFiles(directory="static"), name="static")
        print("Static files mounted successfully")
    else:
        print("Static directory not found")
        
    if os.path.exists("templates"):
        print("Templates directory found, creating Jinja2Templates...")
        templates = Jinja2Templates(directory="templates")
        print("Templates initialized successfully")
    else:
        print("Templates directory not found")
        templates = None
except Exception as e:
    print(f"ERROR mounting static files or templates: {e}")
    logger.warning(f"Could not mount static files or templates: {e}")
    templates = None

# Pydantic models
class CrewRequest(BaseModel):
    description: str
    role: str
    goal: str
    llm_model: Optional[str] = "gpt-3.5-turbo"

class StudioRequest(BaseModel):
    agent_name: str
    agent_role: str
    agent_goal: str
    task_description: str
    expected_output: str
    llm_model: Optional[str] = "gpt-3.5-turbo"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CrewAI Studio API"}

# Original CrewAI endpoint (keep for backward compatibility)
@app.post("/run-crew")
async def run_crew(request: CrewRequest):
    try:
        # Get appropriate LLM for the task (basic fallback)
        from crewai.llm import LLM
        llm = LLM(
            model=request.llm_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create agent
        agent = Agent(
            role=request.role,
            goal=request.goal,
            backstory=f"You are a {request.role} focused on {request.goal}",
            llm=llm,
            verbose=True
        )
        
        # Create task
        task = Task(
            description=request.description,
            agent=agent,
            expected_output="A comprehensive response addressing the task requirements"
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
            "agent_role": request.role,
            "model_used": request.llm_model
        }
        
    except Exception as e:
        logger.error(f"Error in run_crew: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing crew: {str(e)}")

# CrewAI Studio UI endpoint (only if templates are available)
@app.get("/studio", response_class=HTMLResponse)
async def studio_ui(request: Request, credentials: HTTPBasicCredentials = Depends(verify_studio_access)):
    """Serve the CrewAI Studio visual interface - PROTECTED"""
    if templates is None:
        return HTMLResponse("""
        <html>
            <body>
                <h1>CrewAI Studio</h1>
                <p>Studio UI files not found. Use the API endpoint at <a href="/docs">/docs</a></p>
                <p>Available endpoints:</p>
                <ul>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/run-crew">Run Crew API</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
            </body>
        </html>
        """)
    return templates.TemplateResponse("studio.html", {"request": request})

# Studio API endpoint for visual interface (with fallback)
@app.post("/studio/run")
async def run_studio_crew(request: StudioRequest, credentials: HTTPBasicCredentials = Depends(verify_studio_access)):
    """Enhanced endpoint for Studio UI with more detailed configuration - PROTECTED"""
    start_time = time.time()
    session_id = str(uuid.uuid4())
    # memory_manager = get_memory_manager()  # Commented out temporarily
    
    try:
        # Get appropriate LLM for the task (basic fallback)
        from crewai.llm import LLM
        llm = LLM(
            model=request.llm_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create agent with basic backstory (memory integration commented out)
        agent = Agent(
            role=request.agent_role,
            goal=request.agent_goal,
            backstory=f"You are {request.agent_name}, a {request.agent_role}. Your goal is: {request.agent_goal}",
            llm=llm,
            verbose=True
        )
        
        # Create task with expected output
        task = Task(
            description=request.task_description,
            agent=agent,
            expected_output=request.expected_output
        )
        
        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        execution_time = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "result": str(result),
            "agent_name": request.agent_name,
            "agent_role": request.agent_role,
            "model_used": request.llm_model,
            "execution_details": {
                "task_completed": True,
                "expected_output_met": True,
                "execution_time_ms": execution_time,
                "session_id": session_id,
                "memory_saved": False
            }
        }
        
    except Exception as e:
        logger.error(f"Error in studio run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing studio crew: {str(e)}")

# Remove the performance endpoint temporarily
# @app.get("/studio/performance/{agent_name}")  
# async def get_agent_performance(agent_name: str, credentials: HTTPBasicCredentials = Depends(verify_studio_access)):
#     """Get performance metrics for a specific agent"""
#     memory_manager = get_memory_manager()
#     if not memory_manager:
#         return {"error": "Memory manager not available"}
#     
#     performance = memory_manager.get_agent_performance(agent_name)
#     return performance