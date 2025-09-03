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

# Add Archon integration import
try:
    from archon_integration import setup_archon_integration
    ARCHON_INTEGRATION_AVAILABLE = True
    print("SUCCESS: Archon integration imported successfully")
except Exception as e:
    print(f"WARNING: Could not import Archon integration: {e}")
    ARCHON_INTEGRATION_AVAILABLE = False

import time
import uuid

# ADD MONITORING IMPORTS
import psutil
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ADD PINECONE IMPORTS
try:
    from pinecone import Pinecone, ServerlessSpec
    import openai
    PINECONE_AVAILABLE = True
    print("SUCCESS: Pinecone imported successfully")
except Exception as e:
    print(f"WARNING: Could not import Pinecone: {e}")
    PINECONE_AVAILABLE = False

# ADD LAKERA GUARD IMPORTS
import requests
import json

# ADD PLUGIN SYSTEM IMPORTS
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Studio API with Archon MCP", version="1.0.0")
print("FastAPI app created successfully")

# Initialize Archon integration
if ARCHON_INTEGRATION_AVAILABLE:
    try:
        archon_integrator = setup_archon_integration(app)
        if archon_integrator:
            print("🏛️ Archon MCP integration activated")
    except Exception as e:
        print(f"Failed to activate Archon integration: {e}")
        archon_integrator = None
else:
    archon_integrator = None

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

# ADD PINECONE KNOWLEDGE MANAGER
class CourseKnowledgeManager:
    """Manages course knowledge storage and retrieval via Pinecone"""
    
    def __init__(self):
        self.available = False  # Initialize available attribute first
        
        if not PINECONE_AVAILABLE:
            print("WARNING: Pinecone not available, knowledge features disabled")
            return
            
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index_name = "crewai-course-knowledge"
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Create index if it doesn't exist
            self._ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            self.available = True
            print("Pinecone CourseKnowledgeManager initialized successfully")
        except Exception as e:
            print(f"ERROR initializing Pinecone: {e}")
            self.available = False
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI ada-002 embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Created Pinecone index: {self.index_name}")
                time.sleep(10)  # Wait for index to be ready
        except Exception as e:
            print(f"Error ensuring index exists: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        if not self.available:
            return []
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def store_course_content(self, 
                           content: str, 
                           course_name: str,
                           section: str,
                           metadata: Dict[str, Any] = None) -> str:
        """Store course content in Pinecone"""
        if not self.available:
            return "pinecone_unavailable"
            
        try:
            # Generate embedding
            embedding = self.embed_text(content)
            if not embedding:
                return "embedding_failed"
            
            # Prepare metadata
            vector_metadata = {
                "course_name": course_name,
                "section": section,
                "content": content[:1000],  # Store partial content in metadata
                "content_length": len(content),
                "timestamp": str(datetime.now())
            }
            
            if metadata:
                vector_metadata.update(metadata)
            
            # Generate unique ID
            vector_id = f"{course_name}_{section}_{hash(content[:100])}"
            
            # Store in Pinecone
            self.index.upsert([(vector_id, embedding, vector_metadata)])
            
            return vector_id
        except Exception as e:
            print(f"Error storing course content: {e}")
            return "storage_failed"
    
    def search_knowledge(self, 
                        query: str, 
                        course_filter: str = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant course knowledge"""
        if not self.available:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            if not query_embedding:
                return []
            
            # Build filter
            filter_dict = {}
            if course_filter:
                filter_dict["course_name"] = course_filter
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            knowledge_chunks = []
            for match in results.matches:
                knowledge_chunks.append({
                    "content": match.metadata.get("content", ""),
                    "course": match.metadata.get("course_name", ""),
                    "section": match.metadata.get("section", ""),
                    "relevance_score": match.score,
                    "id": match.id
                })
            
            return knowledge_chunks
        except Exception as e:
            print(f"Error searching knowledge: {e}")
            return []
    
    def get_agent_context(self, 
                         task_description: str, 
                         agent_role: str,
                         max_context: int = 3) -> str:
        """Get relevant context for CrewAI agents"""
        if not self.available:
            return ""
            
        try:
            # Create search query combining task and role
            search_query = f"{agent_role}: {task_description}"
            
            # Search for relevant knowledge
            knowledge = self.search_knowledge(search_query, top_k=max_context)
            
            if not knowledge:
                return ""
            
            # Format context for agent
            context_parts = []
            for item in knowledge:
                context_parts.append(
                    f"[{item['course']} - {item['section']}]: {item['content']}"
                )
            
            return "\n\nRelevant Course Knowledge:\n" + "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error getting agent context: {e}")
            return ""

# Initialize global knowledge manager
try:
    knowledge_manager = CourseKnowledgeManager()
    print("Global knowledge manager initialized")
except Exception as e:
    print(f"Failed to initialize knowledge manager: {e}")
    knowledge_manager = None

# ADD PLUGIN AGENT LOADER CLASS
class AgentLoader:
    """Dynamic agent plugin loader"""
    
    def __init__(self):
        self.agents = {}
        self.load_all_agents()
    
    def load_all_agents(self):
        """Auto-discover and load all agent plugins"""
        agents_dir = Path("plugins/agents")
        
        # Create plugins directory if it doesn't exist
        agents_dir.mkdir(parents=True, exist_ok=True)
        
        if not agents_dir.exists():
            print("Plugins directory not found, creating...")
            return
            
        for file in agents_dir.glob("*.py"):
            if file.name.startswith("__"):
                continue
                
            module_name = f"plugins.agents.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'AGENT_CLASS') and hasattr(module, 'AGENT_NAME'):
                    self.agents[module.AGENT_NAME] = module.AGENT_CLASS
                    print(f"Loaded agent: {module.AGENT_NAME}")
            except Exception as e:
                print(f"Failed to load agent from {file}: {e}")
    
    def get_agent_class(self, name: str):
        """Get agent class by name"""
        return self.agents.get(name)
    
    def list_agents(self):
        """List all available agents"""
        return list(self.agents.keys())
    
    def reload_agents(self):
        """Reload all agents (for development)"""
        self.agents.clear()
        self.load_all_agents()

# Initialize Agent Loader
print("Initializing Agent Loader...")
try:
    agent_loader = AgentLoader()
    print(f"Agent loader initialized with {len(agent_loader.agents)} agents")
except Exception as e:
    print(f"Failed to initialize agent loader: {e}")
    agent_loader = None

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

# NEW PINECONE MODELS
class CourseUploadRequest(BaseModel):
    course_name: str
    file_name: str
    content: str
    section: Optional[str] = None

class KnowledgeSearchRequest(BaseModel):
    query: str
    course_filter: Optional[str] = None
    limit: Optional[int] = 5

# ADD PLUGIN AGENT MODEL
class AgentExecuteRequest(BaseModel):
    task_description: str
    context_data: Optional[Dict[str, Any]] = None
    source_info: Optional[Dict[str, Any]] = None
    llm_model: Optional[str] = "gpt-4"

# Health check endpoint - ENHANCED WITH ARCHON STATUS
@app.get("/health")
async def health_check():
    pinecone_status = "available" if (knowledge_manager and knowledge_manager.available) else "unavailable"
    lakera_status = "available" if (lakera_guard and lakera_guard.available) else "unavailable"
    archon_status = "available" if archon_integrator else "unavailable"
    
    return {
        "status": "healthy", 
        "service": "CrewAI Studio API with Archon MCP",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(time.time() - start_time, 2),
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "cpu_usage_percent": round(psutil.cpu_percent(), 1),
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "monitoring_dashboard": True,
            "pinecone_knowledge": pinecone_status,
            "lakera_security": lakera_status,
            "plugin_agents": len(agent_loader.agents) if agent_loader else 0,
            "archon_mcp": archon_status
        }
    }

print("Health endpoint defined")

# Original CrewAI endpoint (keep for backward compatibility) - ENHANCED WITH MONITORING AND LAKERA GUARD
@app.post("/run-crew")
async def run_crew(request: CrewRequest):
    request_start_time = time.time()
    active_tasks.inc()  # Increment active task counter
    
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
        raise  # Re-raise HTTP exceptions (like security blocks)
    except Exception as e:
        # Record error metrics
        duration = time.time() - request_start_time
        record_task_metrics(request.description[:50] if hasattr(request, 'description') else 'unknown', 
                          request.role if hasattr(request, 'role') else 'unknown', 
                          'error', duration, 'run-crew')
        active_tasks.dec()
        
        logger.error(f"Error in run_crew: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing crew: {str(e)}")

print("Basic crew endpoint defined")

# Root endpoint - ENHANCED WITH ARCHON STATUS
@app.get("/")
async def root():
    pinecone_status = "available" if (knowledge_manager and knowledge_manager.available) else "unavailable"
    lakera_status = "available" if (lakera_guard and lakera_guard.available) else "unavailable"
    archon_status = "integrated" if archon_integrator else "unavailable"
    
    endpoints = {
        "health": "/health",
        "run_crew": "/run-crew", 
        "monitoring_dashboard": "/dashboard",
        "prometheus_metrics": "/metrics",
        "api_docs": "/docs"
    }
    
    # Add Archon endpoints if available
    if archon_integrator:
        endpoints.update({
            "archon_interface": "/archon",
            "archon_status": "/archon/status",
            "archon_start": "/archon/start",
            "archon_stop": "/archon/stop",
            "archon_health": "/health/archon"
        })
    
    return {
        "message": "CrewAI Studio API with Archon MCP Integration is running on Render!",
        "status": "healthy",
        "endpoints": endpoints,
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "monitoring_dashboard": True,
            "pinecone_knowledge": pinecone_status,
            "lakera_security": lakera_status,
            "plugin_agents": len(agent_loader.agents) if agent_loader else 0,
            "archon_mcp_integration": archon_status,
            "docker_deployment": True
        }
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API with Archon MCP Integration...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print(f"Pinecone available: {PINECONE_AVAILABLE}")
        print(f"Knowledge manager ready: {getattr(knowledge_manager, 'available', False) if knowledge_manager else False}")
        print(f"Lakera Guard ready: {getattr(lakera_guard, 'available', False) if lakera_guard else False}")
        print(f"Plugin agents loaded: {len(agent_loader.agents) if agent_loader else 0}")
        print(f"Archon MCP integration: {'Active' if archon_integrator else 'Inactive'}")
        print(f"Dashboard will be available at: http://localhost:{port}/dashboard")
        if archon_integrator:
            print(f"🏛️ Archon MCP will be available at: http://localhost:{port}/archon")
        print("Initializing uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise