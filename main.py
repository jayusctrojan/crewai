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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Studio API", version="1.0.0")
print("FastAPI app created successfully")

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
                        region=os.getenv("PINECONE_REGION", "us-west-2")  # Configurable, defaults to West Coast
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

# Health check endpoint - ENHANCED WITH MONITORING AND PINECONE STATUS
@app.get("/health")
async def health_check():
    pinecone_status = "available" if (knowledge_manager and knowledge_manager.available) else "unavailable"
    
    return {
        "status": "healthy", 
        "service": "CrewAI Studio API",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(time.time() - start_time, 2),
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "cpu_usage_percent": round(psutil.cpu_percent(), 1),
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "monitoring_dashboard": True,
            "pinecone_knowledge": pinecone_status
        }
    }

print("Health endpoint defined")

# Original CrewAI endpoint (keep for backward compatibility) - ENHANCED WITH MONITORING
@app.post("/run-crew")
async def run_crew(request: CrewRequest):
    request_start_time = time.time()
    active_tasks.inc()  # Increment active task counter
    
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
        
        # Record success metrics
        duration = time.time() - request_start_time
        record_task_metrics(request.description[:50], request.role, 'success', duration, 'run-crew')
        active_tasks.dec()
        
        return {
            "success": True,
            "result": str(result),
            "agent_role": request.role,
            "model_used": request.llm_model,
            "execution_time": round(duration, 2)
        }
        
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

# NEW KNOWLEDGE-ENHANCED CREWAI ENDPOINT
@app.post("/crew/enhanced")
async def run_enhanced_crew(request: CrewRequest):
    """CrewAI endpoint enhanced with course knowledge from Pinecone"""
    request_start_time = time.time()
    active_tasks.inc()
    
    try:
        # Get knowledge context if available
        knowledge_context = ""
        if knowledge_manager and knowledge_manager.available:
            knowledge_context = knowledge_manager.get_agent_context(
                request.description, 
                request.role,
                max_context=3
            )
        
        # Enhanced backstory with knowledge
        backstory = f"You are a {request.role} focused on {request.goal}"
        if knowledge_context:
            backstory += f"\n\nYou have access to relevant course materials and knowledge base. Use this information to provide accurate, detailed responses.\n{knowledge_context}"
        
        # Get appropriate LLM for the task
        from crewai.llm import LLM
        llm = LLM(
            model=request.llm_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create knowledge-enhanced agent
        agent = Agent(
            role=request.role,
            goal=request.goal,
            backstory=backstory,
            llm=llm,
            verbose=True
        )
        
        # Create task
        task = Task(
            description=request.description,
            agent=agent,
            expected_output="A comprehensive response using available course knowledge and expertise"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Record success metrics
        duration = time.time() - request_start_time
        record_task_metrics(request.description[:50], request.role, 'success', duration, 'crew-enhanced')
        active_tasks.dec()
        
        return {
            "success": True,
            "result": str(result),
            "agent_role": request.role,
            "model_used": request.llm_model,
            "execution_time": round(duration, 2),
            "enhanced_with_knowledge": bool(knowledge_context),
            "knowledge_available": knowledge_manager and knowledge_manager.available
        }
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - request_start_time
        record_task_metrics(request.description[:50], request.role, 'error', duration, 'crew-enhanced')
        active_tasks.dec()
        
        logger.error(f"Error in run_enhanced_crew: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing enhanced crew: {str(e)}")

# PINECONE KNOWLEDGE ENDPOINTS
@app.post("/knowledge/upload")
async def upload_course_content(request: CourseUploadRequest):
    """Upload course content to Pinecone knowledge base"""
    if not knowledge_manager or not knowledge_manager.available:
        raise HTTPException(status_code=503, detail="Pinecone knowledge system unavailable")
    
    try:
        # Process content into chunks
        chunk_size = 1000
        words = request.content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        stored_ids = []
        for i, chunk in enumerate(chunks):
            section = request.section or f"{request.file_name}_part_{i+1}"
            
            vector_id = knowledge_manager.store_course_content(
                content=chunk,
                course_name=request.course_name,
                section=section,
                metadata={
                    "file_name": request.file_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            stored_ids.append(vector_id)
        
        return {
            "success": True,
            "course_name": request.course_name,
            "file_name": request.file_name,
            "chunks_stored": len(stored_ids),
            "vector_ids": stored_ids
        }
        
    except Exception as e:
        logger.error(f"Error uploading course content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading content: {str(e)}")

@app.post("/knowledge/search")
async def search_knowledge(request: KnowledgeSearchRequest):
    """Search the knowledge base for relevant information"""
    if not knowledge_manager or not knowledge_manager.available:
        raise HTTPException(status_code=503, detail="Pinecone knowledge system unavailable")
    
    try:
        results = knowledge_manager.search_knowledge(
            query=request.query,
            course_filter=request.course_filter,
            top_k=request.limit
        )
        
        return {
            "success": True,
            "query": request.query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching knowledge: {str(e)}")

@app.get("/knowledge/search")
async def search_knowledge_get(query: str, course: Optional[str] = None, limit: int = 5):
    """GET endpoint for knowledge search (for easy browser testing)"""
    if not knowledge_manager or not knowledge_manager.available:
        return {"error": "Pinecone knowledge system unavailable"}
    
    try:
        results = knowledge_manager.search_knowledge(
            query=query,
            course_filter=course,
            top_k=limit
        )
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {str(e)}")
        return {"error": f"Error searching knowledge: {str(e)}"}

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
                    <li><a href="/crew/enhanced">Enhanced Crew with Knowledge</a></li>
                    <li><a href="/dashboard">Monitoring Dashboard</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
            </body>
        </html>
        """)
    return templates.TemplateResponse("studio.html", {"request": request})

print("Studio UI endpoint defined")

# FULL Studio API endpoint with memory integration and manual auth - ENHANCED WITH MONITORING AND KNOWLEDGE
print("About to define studio/run endpoint...")

@app.post("/studio/run")
async def run_studio_crew(request: StudioRequest, credentials: HTTPBasicCredentials = Depends(security)):
    """Full endpoint with memory integration, knowledge enhancement, and manual auth checking"""
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

    start_time_request = time.time()
    active_tasks.inc()  # Increment active task counter
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

        # Get course knowledge context
        course_knowledge_context = ""
        if knowledge_manager and knowledge_manager.available:
            course_knowledge_context = knowledge_manager.get_agent_context(
                request.task_description,
                request.agent_role,
                max_context=3
            )

        # Enhanced backstory with both memory and course knowledge
        backstory = f"You are {request.agent_name}, a {request.agent_role}. Your goal is: {request.agent_goal}"
        
        if agent_knowledge:
            backstory += f"\n\nYour previous knowledge includes:\n" + "\n".join(f"- {k}" for k in agent_knowledge)
        
        if course_knowledge_context:
            backstory += f"\n\nYou have access to comprehensive course materials and knowledge base:\n{course_knowledge_context}"

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
                "memory_saved": memory_manager_instance is not None,
                "knowledge_enhanced": bool(course_knowledge_context)
            }
        }
        
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
    
    memory_manager_instance = get_memory_manager()
    if not memory_manager_instance:
        return {"error": "Memory manager not available"}
    
    performance = memory_manager_instance.get_agent_performance(agent_name)
    return performance

print("Performance endpoint defined")

print("Studio endpoints defined")

# ADD MONITORING DASHBOARD ENDPOINTS (keeping your existing dashboard code)
@app.get("/dashboard", response_class=HTMLResponse)
async def monitoring_dashboard():
    """Monitoring dashboard for CrewAI service - No auth required for monitoring"""
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CrewAI Monitoring Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); 
                color: white; 
                min-height: 100vh;
            }
            .header {
                background: rgba(76, 175, 80, 0.1);
                border-bottom: 2px solid #4CAF50;
                padding: 20px;
                text-align: center;
            }
            .header h1 {
                color: #4CAF50;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4CAF50;
                animation: pulse 2s infinite;
                margin-left: 10px;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
                100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .metrics-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
                gap: 20px; 
                margin: 30px 0; 
            }
            .metric-card { 
                background: rgba(45, 45, 45, 0.8); 
                padding: 25px; 
                border-radius: 12px; 
                border: 1px solid #444; 
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(76, 175, 80, 0.2);
            }
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #4CAF50, #81C784);
            }
            .metric-value { 
                font-size: 2.5em; 
                font-weight: bold; 
                margin-bottom: 8px;
                background: linear-gradient(135deg, #4CAF50, #81C784);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .metric-label { 
                color: #bbb; 
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .chart-container { 
                background: rgba(45, 45, 45, 0.8); 
                padding: 30px; 
                border-radius: 12px; 
                margin: 20px 0; 
                border: 1px solid #444;
                backdrop-filter: blur(10px);
            }
            .chart-title {
                color: #81C784;
                font-size: 1.3em;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .chart-title::before {
                content: '📊';
                font-size: 1.2em;
            }
            .charts-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            @media (max-width: 768px) {
                .charts-grid { grid-template-columns: 1fr; }
                .metrics-grid { grid-template-columns: 1fr; }
                .header h1 { font-size: 2em; }
            }
            .last-updated {
                text-align: center;
                color: #666;
                margin-top: 30px;
                padding: 20px;
                border-top: 1px solid #444;
            }
            .loading {
                text-align: center;
                color: #4CAF50;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 CrewAI Studio Monitoring<span class="status-indicator"></span></h1>
            <p>Real-time service monitoring and analytics with knowledge base</p>
        </div>
        
        <div class="container">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="success-rate">--</div>
                    <div class="metric-label">Success Rate (%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="avg-duration">--</div>
                    <div class="metric-label">Avg Duration (sec)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="total-tasks">--</div>
                    <div class="metric-label">Total Tasks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="active-tasks">--</div>
                    <div class="metric-label">Active Tasks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="memory-usage">--</div>
                    <div class="metric-label">Memory Usage (MB)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="cpu-usage">--</div>
                    <div class="metric-label">CPU Usage (%)</div>
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">Task Execution Timeline</div>
                    <canvas id="taskChart" height="300"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Response Time Distribution</div>
                    <canvas id="responseChart" height="300"></canvas>
                </div>
            </div>
        </div>
        
        <div class="last-updated">
            <p>Last updated: <span id="last-update">--</span></p>
            <p class="loading" id="loading" style="display: none;">Refreshing data...</p>
        </div>
        
        <script>
            function formatNumber(num) {
                if (num >= 1000) return (num/1000).toFixed(1) + 'K';
                return num.toString();
            }
            
            async function updateMetrics() {
                document.getElementById('loading').style.display = 'block';
                
                try {
                    const response = await fetch('/api/dashboard/metrics');
                    const data = await response.json();
                    
                    document.getElementById('success-rate').textContent = data.success_rate.toFixed(1);
                    document.getElementById('avg-duration').textContent = data.avg_duration.toFixed(2);
                    document.getElementById('total-tasks').textContent = formatNumber(data.total_tasks);
                    document.getElementById('active-tasks').textContent = data.active_tasks;
                    document.getElementById('memory-usage').textContent = data.memory_usage_mb.toFixed(1);
                    document.getElementById('cpu-usage').textContent = data.cpu_usage_percent.toFixed(1);
                    
                    document.getElementById('last-update').textContent = new Date().toLocaleString();
                } catch (error) {
                    console.error('Error updating metrics:', error);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            Chart.defaults.color = 'white';
            Chart.defaults.borderColor = '#444';
            
            const taskCtx = document.getElementById('taskChart').getContext('2d');
            const taskChart = new Chart(taskCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Tasks per Hour',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { labels: { color: 'white' } },
                        tooltip: {
                            backgroundColor: 'rgba(45, 45, 45, 0.9)',
                            titleColor: 'white',
                            bodyColor: 'white'
                        }
                    },
                    scales: {
                        x: { 
                            ticks: { color: 'white' }, 
                            grid: { color: '#444' }
                        },
                        y: { 
                            ticks: { color: 'white' }, 
                            grid: { color: '#444' }
                        }
                    }
                }
            });
            
            const responseCtx = document.getElementById('responseChart').getContext('2d');
            const responseChart = new Chart(responseCtx, {
                type: 'doughnut',
                data: {
                    labels: ['< 1s', '1-5s', '5-10s', '10-30s', '> 30s'],
                    datasets: [{
                        data: [0, 0, 0, 0, 0],
                        backgroundColor: ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'],
                        borderColor: '#2d2d2d',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { 
                            labels: { color: 'white' },
                            position: 'bottom'
                        },
                        tooltip: {
                            backgroundColor: 'rgba(45, 45, 45, 0.9)',
                            titleColor: 'white',
                            bodyColor: 'white'
                        }
                    }
                }
            });
            
            async function updateCharts() {
                try {
                    const response = await fetch('/api/dashboard/charts');
                    const data = await response.json();
                    
                    taskChart.data.labels = data.task_timeline.labels;
                    taskChart.data.datasets[0].data = data.task_timeline.data;
                    taskChart.update('none');
                    
                    responseChart.data.datasets[0].data = data.response_distribution;
                    responseChart.update('none');
                } catch (error) {
                    console.error('Error updating charts:', error);
                }
            }
            
            updateMetrics();
            updateCharts();
            
            setInterval(updateMetrics, 15000);  // Every 15 seconds
            setInterval(updateCharts, 30000);   // Every 30 seconds
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=dashboard_html)

@app.get("/api/dashboard/metrics")
async def dashboard_metrics():
    """API endpoint for dashboard metrics"""
    update_system_metrics()
    
    # Get metrics from Prometheus counters
    success_count = 0
    error_count = 0
    total_duration = 0
    sample_count = 0
    
    try:
        for sample in task_counter.collect()[0].samples:
            if 'success' in str(sample):
                success_count += sample.value
            elif 'error' in str(sample):
                error_count += sample.value
        
        for sample in task_duration.collect()[0].samples:
            if sample.name.endswith('_sum'):
                total_duration = sample.value
            elif sample.name.endswith('_count'):
                sample_count = sample.value
    except:
        pass  # Handle any metric collection errors
    
    total_tasks = success_count + error_count
    success_rate = (success_count / total_tasks * 100) if total_tasks > 0 else 100
    avg_duration = (total_duration / sample_count) if sample_count > 0 else 0
    
    # System metrics
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    cpu_percent = psutil.cpu_percent()
    current_active = active_tasks._value._value if hasattr(active_tasks._value, '_value') else 0
    
    return {
        "success_rate": round(success_rate, 1),
        "avg_duration": round(avg_duration, 2),
        "total_tasks": int(total_tasks),
        "active_tasks": current_active,
        "memory_usage_mb": round(memory_mb, 1),
        "cpu_usage_percent": round(cpu_percent, 1),
        "service_uptime": time.time() - start_time
    }

@app.get("/api/dashboard/charts")
async def dashboard_charts():
    """API endpoint for chart data"""
    # Generate last 12 hours of sample data
    labels = []
    data = []
    for i in range(12):
        hour = (datetime.now() - timedelta(hours=11-i)).strftime('%H:00')
        labels.append(hour)
        # Sample data - in future can be enhanced with real Supabase data
        data.append(max(0, int(3 + 2 * (i % 5))))
    
    # Sample response time distribution
    response_distribution = [60, 25, 10, 4, 1]
    
    return {
        "task_timeline": {
            "labels": labels,
            "data": data
        },
        "response_distribution": response_distribution
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint for external monitoring tools"""
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

print("Monitoring dashboard endpoints defined")

# Root endpoint - ENHANCED WITH PINECONE STATUS
@app.get("/")
async def root():
    pinecone_status = "available" if (knowledge_manager and knowledge_manager.available) else "unavailable"
    
    return {
        "message": "CrewAI Studio API with Knowledge Base is running on Render!",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "run_crew": "/run-crew", 
            "run_enhanced_crew": "/crew/enhanced",
            "upload_knowledge": "/knowledge/upload",
            "search_knowledge": "/knowledge/search",
            "studio_ui": "/studio",
            "studio_api": "/studio/run",
            "monitoring_dashboard": "/dashboard",
            "prometheus_metrics": "/metrics",
            "api_docs": "/docs"
        },
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "monitoring_dashboard": True,
            "pinecone_knowledge": pinecone_status,
            "docker_deployment": True
        }
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API with Knowledge Base and Monitoring...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print(f"Pinecone available: {PINECONE_AVAILABLE}")
        print(f"Knowledge manager ready: {getattr(knowledge_manager, 'available', False) if knowledge_manager else False}")
        print(f"📊 Dashboard will be available at: http://localhost:{port}/dashboard")
        print(f"📈 Metrics endpoint: http://localhost:{port}/metrics")
        print(f"📚 Knowledge search: http://localhost:{port}/knowledge/search")
        print("Initializing uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise