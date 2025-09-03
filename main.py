from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import os
import uvicorn
from typing import Optional, List, Dict, Any
import logging
import secrets

# Import your existing CrewAI functionality
from crewai import Agent, Task, Crew

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

# ADD MONITORING IMPORTS
import psutil
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Studio API", version="1.0.0")
print("FastAPI app created successfully")

# Initialize Prometheus metrics
task_counter = Counter('crewai_tasks_total', 'Total tasks processed', ['status', 'role', 'task_type', 'endpoint'])
task_duration = Histogram('crewai_task_duration_seconds', 'Task execution duration', buckets=[1, 5, 10, 30, 60, 300])
active_tasks = Gauge('crewai_active_tasks', 'Currently active tasks')
system_memory = Gauge('crewai_memory_usage_bytes', 'Memory usage in bytes')
system_cpu = Gauge('crewai_cpu_usage_percent', 'CPU usage percentage')

# Track service start time
start_time = time.time()
print("Monitoring metrics initialized successfully")

# Security setup
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

# Mount static files and templates
try:
    print("Attempting to mount static files...")
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        print("Static files mounted successfully")
        
    if os.path.exists("templates"):
        templates = Jinja2Templates(directory="templates")
        print("Templates initialized successfully")
    else:
        templates = None
except Exception as e:
    print(f"ERROR mounting static files or templates: {e}")
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
    return {
        "status": "healthy", 
        "service": "CrewAI Studio API",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(time.time() - start_time, 2),
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "features": {
            "memory_system": MEMORY_AVAILABLE,
            "monitoring": True
        }
    }

# Studio API endpoint
@app.post("/studio/run")
async def run_studio_crew(request: StudioRequest, credentials: HTTPBasicCredentials = Depends(security)):
    """Studio endpoint with memory integration"""
    start_time_request = time.time()
    active_tasks.inc()
    session_id = str(uuid.uuid4())
    memory_manager_instance = get_memory_manager() if MEMORY_AVAILABLE else None
    
    try:
        # Save task start to memory
        if memory_manager_instance:
            memory_manager_instance.save_agent_memory(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                content=f"Starting task: {request.task_description}",
                memory_type="task_start",
                session_id=session_id
            )

        # Get previous knowledge for this agent type
        agent_knowledge = []
        if memory_manager_instance:
            knowledge = memory_manager_instance.get_agent_knowledge(request.agent_name)
            agent_knowledge = [k["knowledge_content"] for k in knowledge[:3]]

        # Enhanced backstory with memory
        backstory = f"You are {request.agent_name}, a {request.agent_role}. Your goal is: {request.agent_goal}"
        
        if agent_knowledge:
            backstory += f"\n\nYour previous knowledge includes:\n" + "\n".join(f"- {k}" for k in agent_knowledge)

        # Get appropriate LLM for the task
        from crewai.llm import LLM
        llm = LLM(
            model=request.llm_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create agent with enhanced backstory
        agent = Agent(
            role=request.agent_role,
            goal=request.agent_goal,
            backstory=backstory,
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
        execution_time = int((time.time() - start_time_request) * 1000)

        # Save results to memory
        if memory_manager_instance:
            memory_manager_instance.save_agent_memory(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                content=str(result),
                memory_type="task_result",
                session_id=session_id,
                metadata={"task": request.task_description, "model": request.llm_model}
            )
            
            memory_manager_instance.log_task_execution(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                task_description=request.task_description,
                expected_output=request.expected_output,
                actual_output=str(result),
                execution_time_ms=execution_time,
                model_used=request.llm_model,
                success=True
            )

        active_tasks.dec()

        return {
            "success": True,
            "result": str(result),
            "agent_name": request.agent_name,
            "agent_role": request.agent_role,
            "model_used": request.llm_model,
            "execution_details": {
                "execution_time_ms": execution_time,
                "session_id": session_id,
                "memory_saved": memory_manager_instance is not None
            }
        }
        
    except Exception as e:
        active_tasks.dec()
        logger.error(f"Error in studio run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing studio crew: {str(e)}")

# Original CrewAI endpoint
@app.post("/run-crew")
async def run_crew(request: CrewRequest):
    try:
        from crewai.llm import LLM
        llm = LLM(
            model=request.llm_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        agent = Agent(
            role=request.role,
            goal=request.goal,
            backstory=f"You are a {request.role} focused on {request.goal}",
            llm=llm,
            verbose=True
        )
        
        task = Task(
            description=request.description,
            agent=agent,
            expected_output="A comprehensive response addressing the task requirements"
        )
        
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "CrewAI Studio API is running on Render!",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "run_crew": "/run-crew", 
            "studio_api": "/studio/run",
            "api_docs": "/docs"
        },
        "features": {
            "memory_system": MEMORY_AVAILABLE,
            "monitoring": True
        },
        "archon_service": "https://jb-archon.onrender.com (separate service for agent prototyping)"
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print("🚀 CrewAI Production API")
        print("🏛️ Archon prototyping available at separate service")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise