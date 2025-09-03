from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response  # Added Response for metrics
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import os
import uvicorn
from typing import Optional, List, Dict, Any
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

# Try to import real Archon integration
try:
    from real_archon import real_archon_setup
    REAL_ARCHON_AVAILABLE = True
    print("SUCCESS: Real Archon integration imported successfully")
except Exception as e:
    print(f"WARNING: Could not import Real Archon integration: {e}")
    REAL_ARCHON_AVAILABLE = False

import time
import uuid

# ADD MONITORING IMPORTS
import psutil
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ADD LAKERA GUARD IMPORTS
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Studio API with Real Archon MCP", version="1.0.0")
print("FastAPI app created successfully")

# Initialize Real Archon integration
archon_integrator = None
if REAL_ARCHON_AVAILABLE:
    try:
        archon_integrator = real_archon_setup(app)
        print("🏛️ Real Archon integration activated")
    except Exception as e:
        print(f"Failed to activate Real Archon integration: {e}")
        archon_integrator = None
else:
    print("Real Archon integration not available")

# ADD MONITORING SETUP AFTER APP CREATION
# Initialize Prometheus metrics
task_counter = Counter('crewai_tasks_total', 'Total tasks processed', ['status', 'role', 'task_type', 'endpoint'])
task_duration = Histogram('crewai_task_duration_seconds', 'Task execution duration', buckets=[1, 5, 10, 30, 60, 300])
request_duration = Histogram('crewai_request_duration_seconds', 'HTTP request duration')
active_tasks = Gauge('crewai_active_tasks', 'Currently active tasks')
system_memory = Gauge('crewai_memory_usage_bytes', 'Memory usage in bytes')
system_cpu = Gauge('crewai_cpu_usage_percent', 'CPU usage percentage')

# Track service start time
start_time = time.time()
print("Monitoring metrics initialized successfully")

# ADD LAKERA GUARD CLASS
class LakeraGuard:
    """Lakera Guard integration for AI security"""
    
    def __init__(self):
        self.api_key = os.getenv("LAKERA_API_KEY")
        self.project_id = os.getenv("LAKERA_PROJECT_ID") 
        self.base_url = "https://api.lakera.ai/v1/guard"
        self.available = bool(self.api_key)
        
        if self.available:
            print("SUCCESS: Lakera Guard initialized successfully")
        else:
            print("WARNING: Lakera Guard not available - API key missing")
    
    def screen_content(self, user_input: str, llm_output: str = None) -> Dict[str, Any]:
        """Screen content with Lakera Guard for security threats"""
        if not self.available:
            return {"flagged": False, "categories": [], "lakera_available": False}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": user_input,
        }
        
        if self.project_id:
            payload["project_id"] = self.project_id
            
        if llm_output:
            payload["output"] = llm_output
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "flagged": result.get("flagged", False),
                    "categories": result.get("categories", []),
                    "lakera_available": True,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                logger.warning(f"Lakera Guard API returned status {response.status_code}: {response.text}")
                return {"flagged": False, "categories": [], "lakera_available": False, "error": "API error"}
                
        except requests.RequestException as e:
            logger.error(f"Lakera Guard request failed: {e}")
            return {"flagged": False, "categories": [], "lakera_available": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Lakera Guard error: {e}")
            return {"flagged": False, "categories": [], "lakera_available": False, "error": str(e)}

# Initialize Lakera Guard
try:
    lakera_guard = LakeraGuard()
    print("Global Lakera Guard initialized")
except Exception as e:
    print(f"Failed to initialize Lakera Guard: {e}")
    lakera_guard = None

# ADD HELPER FUNCTIONS
def record_task_metrics(task_type: str, role: str, status: str, duration: float, endpoint: str = "unknown"):
    """Record task execution metrics"""
    task_counter.labels(status=status, role=role, task_type=task_type, endpoint=endpoint).inc()
    task_duration.observe(duration)

def update_system_metrics():
    """Update system resource metrics"""
    try:
        system_memory.set(psutil.Process().memory_info().rss)
        system_cpu.set(psutil.cpu_percent())
    except:
        pass  # Silently handle any system metric errors

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

# Health check endpoint - ENHANCED WITH REAL ARCHON STATUS
@app.get("/health")
async def health_check():
    lakera_status = "available" if (lakera_guard and lakera_guard.available) else "unavailable"
    archon_status = "real_archon_integrated" if archon_integrator else "unavailable"
    
    return {
        "status": "healthy", 
        "service": "CrewAI Studio API with Real Archon MCP",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(time.time() - start_time, 2),
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "cpu_usage_percent": round(psutil.cpu_percent(), 1),
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "lakera_security": lakera_status,
            "real_archon_mcp": archon_status
        }
    }

print("Health endpoint defined")

# Studio API endpoint with memory integration and Lakera Guard protection
@app.post("/studio/run")
async def run_studio_crew(request: StudioRequest, credentials: HTTPBasicCredentials = Depends(security)):
    """Studio endpoint with memory integration and Lakera Guard security"""
    # Manual authentication check
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

    start_time_request = time.time()
    active_tasks.inc()
    session_id = str(uuid.uuid4())
    memory_manager_instance = get_memory_manager() if MEMORY_AVAILABLE else None
    
    try:
        # SCREEN INPUT WITH LAKERA GUARD
        if lakera_guard and lakera_guard.available:
            guard_result = lakera_guard.screen_content(request.task_description)
            
            if guard_result.get("flagged"):
                active_tasks.dec()
                record_task_metrics(request.task_description[:50], request.agent_role, 'blocked_by_security', 
                                  time.time() - start_time_request, 'studio-run')
                raise HTTPException(
                    status_code=400, 
                    detail=f"Task description blocked by security policy. Detected threats: {guard_result.get('categories', [])}"
                )
        
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

        # SCREEN OUTPUT WITH LAKERA GUARD
        output_blocked = False
        if lakera_guard and lakera_guard.available:
            output_guard = lakera_guard.screen_content("", str(result))
            if output_guard.get("flagged"):
                output_blocked = True
                result = f"Response blocked by security policy. Detected threats: {output_guard.get('categories', [])}"

        # Save results to memory
        if memory_manager_instance:
            # Save task result
            memory_manager_instance.save_agent_memory(
                agent_name=request.agent_name,
                agent_role=request.agent_role,
                content=str(result),
                memory_type="task_result",
                session_id=session_id,
                metadata={"task": request.task_description, "model": request.llm_model}
            )
            
            # Log execution
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

        # Record success metrics
        duration = time.time() - start_time_request
        record_task_metrics(request.task_description[:50], request.agent_role, 'success', duration, 'studio-run')
        active_tasks.dec()

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
                "memory_saved": memory_manager_instance is not None
            },
            "security_screening": {
                "input_blocked": False,
                "output_blocked": output_blocked,
                "lakera_enabled": lakera_guard and lakera_guard.available
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like security blocks)
    except Exception as e:
        execution_time = int((time.time() - start_time_request) * 1000)
        
        # Log error to memory
        if memory_manager_instance:
            memory_manager_instance.log_task_execution(
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

        # Record error metrics
        duration = time.time() - start_time_request
        record_task_metrics(request.task_description[:50], request.agent_role, 'error', duration, 'studio-run')
        active_tasks.dec()

        logger.error(f"Error in studio run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing studio crew: {str(e)}")

print("Studio endpoint defined")

# Original CrewAI endpoint (keep for backward compatibility)
@app.post("/run-crew")
async def run_crew(request: CrewRequest):
    request_start_time = time.time()
    active_tasks.inc()
    
    try:
        # SCREEN INPUT WITH LAKERA GUARD
        if lakera_guard and lakera_guard.available:
            guard_result = lakera_guard.screen_content(request.description)
            
            if guard_result.get("flagged"):
                active_tasks.dec()
                record_task_metrics(request.description[:50], request.role, 'blocked_by_security', 
                                  time.time() - request_start_time, 'run-crew')
                raise HTTPException(
                    status_code=400, 
                    detail=f"Content blocked by security policy. Detected threats: {guard_result.get('categories', [])}"
                )
        
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
        
        # SCREEN OUTPUT WITH LAKERA GUARD
        output_blocked = False
        if lakera_guard and lakera_guard.available:
            output_guard = lakera_guard.screen_content("", str(result))
            if output_guard.get("flagged"):
                output_blocked = True
                result = f"Response blocked by security policy. Detected threats: {output_guard.get('categories', [])}"
        
        # Record success metrics
        duration = time.time() - request_start_time
        record_task_metrics(request.description[:50], request.role, 'success', duration, 'run-crew')
        active_tasks.dec()
        
        return {
            "success": True,
            "result": str(result),
            "agent_role": request.role,
            "model_used": request.llm_model,
            "execution_time": round(duration, 2),
            "security_screening": {
                "input_blocked": False,
                "output_blocked": output_blocked,
                "lakera_enabled": lakera_guard and lakera_guard.available
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - request_start_time
        record_task_metrics(request.description[:50] if hasattr(request, 'description') else 'unknown', 
                          request.role if hasattr(request, 'role') else 'unknown', 
                          'error', duration, 'run-crew')
        active_tasks.dec()
        
        logger.error(f"Error in run_crew: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing crew: {str(e)}")

print("Basic crew endpoint defined")

# Root endpoint - ENHANCED WITH REAL ARCHON STATUS
@app.get("/")
async def root():
    lakera_status = "available" if (lakera_guard and lakera_guard.available) else "unavailable"
    archon_status = "real_archon_integrated" if archon_integrator else "unavailable"
    
    endpoints = {
        "health": "/health",
        "run_crew": "/run-crew", 
        "studio_api": "/studio/run",
        "api_docs": "/docs"
    }
    
    # Add Real Archon endpoints if available
    if archon_integrator:
        endpoints.update({
            "archon_interface": "/archon",
            "archon_status": "/archon/status",
            "archon_setup": "/archon/setup",
            "archon_start": "/archon/start",
            "archon_stop": "/archon/stop"
        })
    
    return {
        "message": "CrewAI Studio API with Real Archon MCP Integration is running on Render!",
        "status": "healthy",
        "endpoints": endpoints,
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "lakera_security": lakera_status,
            "real_archon_mcp": archon_status,
            "docker_deployment": True
        },
        "archon_info": {
            "repository": "https://github.com/coleam00/Archon",
            "version": "v6-tool-library-integration",
            "description": "The world's first 'Agenteer' - an AI agent that builds other AI agents"
        } if archon_integrator else None
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API with Real Archon MCP Integration...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print(f"Lakera Guard ready: {getattr(lakera_guard, 'available', False) if lakera_guard else False}")
        print(f"Real Archon integration: {'Active' if archon_integrator else 'Inactive'}")
        if archon_integrator:
            print(f"🏛️ Real Archon will be available at: http://localhost:{port}/archon")
            print(f"📚 Repository: https://github.com/coleam00/Archon")
            print(f"🎯 Version: v6-tool-library-integration")
        print("Initializing uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise