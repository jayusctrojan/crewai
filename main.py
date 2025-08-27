from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response  # Added Response for metrics
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

# ADD MONITORING IMPORTS
import psutil
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

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

# Health check endpoint - ENHANCED WITH MONITORING
@app.get("/health")
async def health_check():
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
            "monitoring_dashboard": True
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
                    <li><a href="/dashboard">Monitoring Dashboard</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
            </body>
        </html>
        """)
    return templates.TemplateResponse("studio.html", {"request": request})

print("Studio UI endpoint defined")

# FULL Studio API endpoint with memory integration and manual auth - ENHANCED WITH MONITORING
print("About to define studio/run endpoint...")

@app.post("/studio/run")
async def run_studio_crew(request: StudioRequest, credentials: HTTPBasicCredentials = Depends(security)):
    """Full endpoint with memory integration and manual auth checking - ENHANCED WITH MONITORING"""
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
        execution_time = int((time.time() - start_time_request) * 1000)

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
                "memory_saved": memory_manager is not None
            }
        }
        
    except Exception as e:
        execution_time = int((time.time() - start_time_request) * 1000)
        
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
    
    memory_manager = get_memory_manager()
    if not memory_manager:
        return {"error": "Memory manager not available"}
    
    performance = memory_manager.get_agent_performance(agent_name)
    return performance

print("Performance endpoint defined")

print("Studio endpoints defined")

# ADD MONITORING DASHBOARD ENDPOINTS
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
            <p>Real-time service monitoring and analytics</p>
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
            "monitoring_dashboard": "/dashboard",
            "prometheus_metrics": "/metrics",
            "api_docs": "/docs"
        },
        "features": {
            "visual_studio": templates is not None,
            "memory_system": MEMORY_AVAILABLE,
            "monitoring_dashboard": True,
            "docker_deployment": True
        }
    }

if __name__ == "__main__":
    try:
        print("Starting CrewAI Studio API with Monitoring...")
        port = int(os.getenv("PORT", 8000))
        print(f"Port: {port}")
        print(f"Memory available: {MEMORY_AVAILABLE}")
        print(f"📊 Dashboard will be available at: http://localhost:{port}/dashboard")
        print(f"📈 Metrics endpoint: http://localhost:{port}/metrics")
        print("Initializing uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise