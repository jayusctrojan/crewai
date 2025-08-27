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

print("Health endpoint defined")

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

print("Basic crew endpoint defined")

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

print("Studio UI endpoint defined")

# FULL Studio API endpoint with memory integration and manual auth
print("About to define studio/run endpoint...")

@app.post("/studio/run")
async def run_studio_crew(request: StudioRequest, credentials: HTTPBasicCredentials = Depends(security)):
    """Full endpoint with memory integration and manual auth checking"""
    # Manual authentication check instead of using verify_studio_access dependency
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

    start_time = time.time()
    session_id = str(uuid.uuid4())
    memory_manager = get_memory_manager() if MEMORY_AVAILABLE else None
    
    try:
        # Save task start to memory
        if memory_manager:
            memory_manager.save_agent_memory(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                content=f"Starting task: {request.task_description}",
                memory_type="task_start",
                session_id=session_id
            )

        # Get previous knowledge for this agent type
        agent_knowledge = []
        if memory_manager:
            knowledge = memory_manager.get_agent_knowledge(request.agent_name)
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
        execution_time = int((time.time() - start_time) * 1000)

        # Save results to memory
        if memory_manager:
            # Save task result
            memory_manager.save_agent_memory(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                content=str(result),
                memory_type="task_result",
                session_id=session_id,
                metadata={"task": request.task_description, "model": request.llm_model}
            )
            
            # Log execution
            memory_manager.log_task_execution(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                task_description=request.task_description,
                expected_output=request.expected_output,
                actual_output=str(result),
                execution_time_ms=execution_time,
                model_used=request.llm_model,
                success=True
            )

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
                "memory_saved": memory_manager is not None
            }
        }
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        
        # Log error to memory
        if memory_manager:
            memory_manager.log_task_execution(
                agent_name=request.agent_name,
                agent_role=request.agent_role,  
                task_description=request.task_description,
                expected_output=request.expected_output,
                actual_output="",
                execution_time_ms=execution_time,
                model_used=request.llm_model,
                success=False,
                error_message=str(e)
            )

        logger.error(f"Error in studio run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing studio crew: {str(e)}")

print("Studio/run endpoint defined successfully")

# Performance endpoint with authentication
@app.get("/studio/performance/{agent_name}")
async def get_agent_performance(agent_name: str, credentials: HTTPBasicCredentials = Depends(security)):
    """Get performance metrics for a specific agent - PROTECTED"""
    # Manual auth check
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
    
    memory_manager = get_memory_manager()
    if not memory_manager:
        return {"error": "Memory manager not available"}
    
    performance = memory_manager.get_agent_performance(agent_name)
    return performance

print("Performance endpoint defined")

print("Studio endpoints defined")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "CrewAI Studio API is running on Render!",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "run_crew": "/run-crew", 
            "studio_ui": "/studio",
            "studio_api": "/studio/run",
            "api_docs": "/docs"
        },
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "docker_deployment": True
        }
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print("Initializing uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise