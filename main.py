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

# ADD LAKERA GUARD IMPORTS
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Studio API with Archon MCP", version="1.0.0")
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

# ARCHON WEB INTERFACE - EMBEDDED DIRECTLY
@app.get("/archon", response_class=HTMLResponse)
async def archon_interface():
    """Archon Agent Builder Web Interface - Embedded directly"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏛️ Archon - AI Agent Builder</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-bottom: 20px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        .form-group input,
        .form-group textarea,
        .form-group select {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 16px;
        }
        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        .btn {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .agent-list {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .agent-item {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #4CAF50;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .agent-item:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateX(5px);
        }
        .agent-item h4 {
            margin-bottom: 5px;
        }
        .agent-item p {
            opacity: 0.8;
            font-size: 0.9rem;
        }
        .chat-container {
            grid-column: 1 / -1;
            max-height: 400px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 20px;
        }
        .chat-message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 80%;
        }
        .chat-message.user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            margin-left: auto;
            text-align: right;
        }
        .chat-message.assistant {
            background: rgba(255, 255, 255, 0.1);
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            opacity: 0.8;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
            margin-right: 10px;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }
        .success-msg {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        .error-msg {
            background: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ Archon</h1>
            <p>AI Agent Builder & MCP Interface</p>
            <div class="status-indicator"></div>
            <span>Connected to CrewAI Studio</span>
        </div>

        <div id="successMsg" class="success-msg"></div>
        <div id="errorMsg" class="error-msg"></div>

        <div class="main-grid">
            <!-- Agent Creation Form -->
            <div class="card">
                <h2>🤖 Create AI Agent</h2>
                <form id="agentForm">
                    <div class="form-group">
                        <label for="agentName">Agent Name</label>
                        <input type="text" id="agentName" name="agentName" placeholder="e.g., ResearchBot" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="agentRole">Agent Role</label>
                        <input type="text" id="agentRole" name="agentRole" placeholder="e.g., Research Assistant" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="agentGoal">Agent Goal</label>
                        <textarea id="agentGoal" name="agentGoal" placeholder="What is this agent's primary objective?" required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="llmModel">LLM Model</label>
                        <select id="llmModel" name="llmModel">
                            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                            <option value="gpt-4">GPT-4</option>
                            <option value="gpt-4-turbo">GPT-4 Turbo</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn" id="createBtn">Create & Test Agent</button>
                </form>
            </div>

            <!-- Agent Management -->
            <div class="card">
                <h2>🎯 Created Agents</h2>
                <div id="agentList">
                    <div class="loading">No agents created yet. Create your first agent!</div>
                </div>
            </div>

            <!-- Chat Interface -->
            <div class="card chat-container">
                <h2>💬 Chat with Agent</h2>
                <div id="chatMessages">
                    <div class="loading">Create an agent to start testing</div>
                </div>
                <div style="margin-top: 15px;">
                    <div class="form-group">
                        <input type="text" id="chatInput" placeholder="Type your message..." disabled>
                    </div>
                    <button class="btn" onclick="sendMessage()" id="sendBtn" disabled>Send Message</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let agents = [];
        let selectedAgent = null;
        let chatHistory = [];

        function showMessage(text, type = 'success') {
            const msgEl = document.getElementById(type === 'success' ? 'successMsg' : 'errorMsg');
            msgEl.textContent = text;
            msgEl.style.display = 'block';
            setTimeout(() => { msgEl.style.display = 'none'; }, 5000);
        }

        // Create and Test Agent
        document.getElementById('agentForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const agentData = {
                agent_name: formData.get('agentName'),
                agent_role: formData.get('agentRole'),
                agent_goal: formData.get('agentGoal'),
                task_description: `Hello, I am testing you. Please introduce yourself and explain what you can help with.`,
                expected_output: "A friendly introduction explaining your role and capabilities",
                llm_model: formData.get('llmModel')
            };

            const createBtn = document.getElementById('createBtn');
            createBtn.disabled = true;
            createBtn.textContent = 'Creating & Testing...';

            try {
                const response = await fetch('/studio/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic ' + btoa('admin:changeme123')
                    },
                    body: JSON.stringify(agentData)
                });

                if (response.ok) {
                    const result = await response.json();
                    
                    // Add agent to list
                    const agent = {
                        name: agentData.agent_name,
                        role: agentData.agent_role,
                        goal: agentData.agent_goal,
                        model: agentData.llm_model,
                        created: new Date().toLocaleString(),
                        firstResponse: result.result
                    };
                    
                    agents.push(agent);
                    updateAgentList();
                    
                    // Auto-select and show test result
                    selectAgent(agents.length - 1);
                    chatHistory = [
                        { type: 'user', content: 'Hello, please introduce yourself.' },
                        { type: 'assistant', content: result.result }
                    ];
                    updateChatInterface();
                    
                    e.target.reset();
                    showMessage(`Agent "${agent.name}" created and tested successfully!`, 'success');
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to create agent');
                }
            } catch (error) {
                console.error('Error creating agent:', error);
                showMessage(`Failed to create agent: ${error.message}`, 'error');
            } finally {
                createBtn.disabled = false;
                createBtn.textContent = 'Create & Test Agent';
            }
        });

        // Update Agent List
        function updateAgentList() {
            const agentList = document.getElementById('agentList');
            
            if (agents.length === 0) {
                agentList.innerHTML = '<div class="loading">No agents created yet. Create your first agent!</div>';
                return;
            }

            agentList.innerHTML = agents.map((agent, index) => `
                <div class="agent-item" onclick="selectAgent(${index})" ${selectedAgent === agent ? 'style="background: rgba(76, 175, 80, 0.2);"' : ''}>
                    <h4>🤖 ${agent.name}</h4>
                    <p><strong>Role:</strong> ${agent.role}</p>
                    <p><strong>Model:</strong> ${agent.model}</p>
                    <p><strong>Created:</strong> ${agent.created}</p>
                </div>
            `).join('');
        }

        // Select Agent
        function selectAgent(index) {
            selectedAgent = agents[index];
            updateAgentList(); // Refresh to show selection
            
            // Show agent's first response if available
            if (selectedAgent.firstResponse) {
                chatHistory = [
                    { type: 'user', content: 'Hello, please introduce yourself.' },
                    { type: 'assistant', content: selectedAgent.firstResponse }
                ];
            } else {
                chatHistory = [];
            }
            
            updateChatInterface();
            
            document.getElementById('chatInput').disabled = false;
            document.getElementById('sendBtn').disabled = false;
            
            showMessage(`Selected agent: ${selectedAgent.name}`, 'success');
        }

        // Send Message
        async function sendMessage() {
            const chatInput = document.getElementById('chatInput');
            const message = chatInput.value.trim();
            
            if (!message || !selectedAgent) return;

            chatHistory.push({ type: 'user', content: message });
            updateChatInterface();
            chatInput.value = '';

            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            sendBtn.textContent = 'Agent is thinking...';

            try {
                const response = await fetch('/studio/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic ' + btoa('admin:changeme123')
                    },
                    body: JSON.stringify({
                        agent_name: selectedAgent.name,
                        agent_role: selectedAgent.role,
                        agent_goal: selectedAgent.goal,
                        task_description: message,
                        expected_output: "A helpful response to the user's message",
                        llm_model: selectedAgent.model
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    chatHistory.push({ type: 'assistant', content: result.result });
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to get response');
                }
            } catch (error) {
                console.error('Error sending message:', error);
                chatHistory.push({ type: 'assistant', content: 'Sorry, I encountered an error. Please try again.' });
            } finally {
                updateChatInterface();
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send Message';
            }
        }

        // Update Chat Interface
        function updateChatInterface() {
            const chatMessages = document.getElementById('chatMessages');
            
            if (chatHistory.length === 0) {
                chatMessages.innerHTML = selectedAgent 
                    ? `<div class="loading">Start chatting with ${selectedAgent.name}</div>`
                    : '<div class="loading">Create an agent to start testing</div>';
                return;
            }

            chatMessages.innerHTML = chatHistory.map(msg => `
                <div class="chat-message ${msg.type}">
                    ${msg.content}
                </div>
            `).join('');
            
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Enter key support for chat
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Initialize
        updateAgentList();
        updateChatInterface();
    </script>
</body>
</html>
    """)

@app.get("/archon/status")
async def archon_status():
    """Check Archon status"""
    return {
        "status": "active",
        "type": "embedded_web_interface",
        "message": "Archon web interface is running",
        "endpoints": {
            "interface": "/archon",
            "status": "/archon/status"
        }
    }

# Health check endpoint - ENHANCED WITH ARCHON STATUS
@app.get("/health")
async def health_check():
    lakera_status = "available" if (lakera_guard and lakera_guard.available) else "unavailable"
    
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
            "lakera_security": lakera_status,
            "archon_mcp": "embedded_web_interface_active"
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

# Root endpoint - ENHANCED WITH ARCHON STATUS
@app.get("/")
async def root():
    lakera_status = "available" if (lakera_guard and lakera_guard.available) else "unavailable"
    
    endpoints = {
        "health": "/health",
        "run_crew": "/run-crew", 
        "studio_api": "/studio/run",
        "archon_interface": "/archon",
        "archon_status": "/archon/status",
        "api_docs": "/docs"
    }
    
    return {
        "message": "CrewAI Studio API with Archon MCP Integration is running on Render!",
        "status": "healthy",
        "endpoints": endpoints,
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "lakera_security": lakera_status,
            "archon_mcp_web_interface": "embedded_active",
            "docker_deployment": True
        }
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API with Embedded Archon MCP Interface...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print(f"Lakera Guard ready: {getattr(lakera_guard, 'available', False) if lakera_guard else False}")
        print(f"🏛️ Archon MCP embedded interface will be available at: http://localhost:{port}/archon")
        print("Initializing uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise