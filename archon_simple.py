from fastapi.responses import HTMLResponse

def get_archon_html():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>🏛️ Archon - AI Agent Builder</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
            color: white; min-height: 100vh;
        }
        .header {
            background: rgba(76, 175, 80, 0.1);
            border-bottom: 2px solid #4CAF50;
            padding: 2rem; text-align: center;
        }
        .header h1 {
            font-size: 3rem; margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #4CAF50, #81C784);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 2rem; }
        .card {
            background: rgba(45, 45, 45, 0.9);
            border-radius: 12px; padding: 2rem;
            border: 1px solid #444; margin-bottom: 2rem;
        }
        .card h3 { color: #4CAF50; margin-bottom: 1rem; font-size: 1.5rem; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; color: #bbb; }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%; padding: 0.75rem; border-radius: 6px;
            border: 1px solid #555; background: rgba(0, 0, 0, 0.5);
            color: white; font-size: 1rem;
        }
        .form-group textarea { height: 120px; resize: vertical; }
        .btn {
            background: linear-gradient(135deg, #4CAF50, #81C784);
            color: white; border: none; padding: 0.75rem 2rem;
            border-radius: 6px; cursor: pointer; font-size: 1rem;
            font-weight: 600; margin-top: 1rem;
        }
        .btn:hover { transform: translateY(-2px); }
        .status { 
            padding: 0.5rem 1rem; border-radius: 20px; 
            font-weight: 600; text-align: center; margin: 1rem 0;
        }
        .status.success { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
        .status.error { background: rgba(244, 67, 54, 0.2); color: #f44336; }
        .status.loading { background: rgba(255, 193, 7, 0.2); color: #FFC107; }
        .response-area {
            margin-top: 1rem; padding: 1rem;
            background: rgba(0, 0, 0, 0.3); border-radius: 6px;
            border: 1px solid #333; min-height: 100px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ Archon</h1>
        <p>AI Agent Builder & Prototyping Platform</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h3>🤖 Create & Test AI Agent</h3>
            <form id="agent-form">
                <div class="form-group">
                    <label>Agent Name</label>
                    <input type="text" id="agent-name" placeholder="e.g., ResearchBot" required>
                </div>
                <div class="form-group">
                    <label>Agent Role</label>
                    <input type="text" id="agent-role" placeholder="e.g., Research Assistant" required>
                </div>
                <div class="form-group">
                    <label>Agent Goal</label>
                    <textarea id="agent-goal" placeholder="What is this agent's primary objective?" required></textarea>
                </div>
                <div class="form-group">
                    <label>Test Task</label>
                    <textarea id="test-task" placeholder="Give your agent a task to test its capabilities..." required></textarea>
                </div>
                <div class="form-group">
                    <label>AI Model</label>
                    <select id="ai-model">
                        <option value="gpt-4">GPT-4 (Recommended)</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    </select>
                </div>
                <button type="submit" class="btn">Create & Test Agent</button>
            </form>
            <div id="status"></div>
            <div id="response" class="response-area" style="display: none;"></div>
        </div>
        
        <div class="card">
            <h3>🎯 How It Works</h3>
            <p>1. Enter your agent's name, role, and primary goal</p>
            <p>2. Provide a test task to see how your agent performs</p>
            <p>3. Click "Create & Test Agent" to see it in action</p>
            <p>4. Use the results to refine your agent design</p>
            <br>
            <p><strong>Integration:</strong> This connects to your CrewAI Studio API with knowledge base, security screening, and memory system.</p>
        </div>
    </div>

    <script>
        document.getElementById('agent-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const statusDiv = document.getElementById('status');
            const responseDiv = document.getElementById('response');
            
            statusDiv.innerHTML = '<div class="status loading">🤖 Creating and testing agent...</div>';
            responseDiv.style.display = 'none';
            
            const formData = {
                agent_name: document.getElementById('agent-name').value,
                agent_role: document.getElementById('agent-role').value,
                agent_goal: document.getElementById('agent-goal').value,
                task_description: document.getElementById('test-task').value,
                expected_output: "A comprehensive response demonstrating the agent's capabilities",
                llm_model: document.getElementById('ai-model').value
            };
            
            try {
                const response = await fetch('/studio/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic ' + btoa('admin:changeme123')
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.innerHTML = '<div class="status success">✅ Agent created and tested successfully!</div>';
                    responseDiv.innerHTML = `🤖 ${formData.agent_name} (${formData.agent_role}):\n\n${result.result}\n\n⚡ Execution time: ${result.execution_details?.execution_time_ms || 'N/A'}ms\n🧠 Model used: ${result.model_used}`;
                    responseDiv.style.display = 'block';
                } else {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + (result.detail || 'Unknown error') + '</div>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="status error">❌ Connection Error: ' + error.message + '</div>';
            }
        });
    </script>
</body>
</html>
    """

def setup_archon_routes(app):
    """Add Archon routes to FastAPI"""
    
    @app.get("/archon", response_class=HTMLResponse)
    async def archon_interface():
        return HTMLResponse(content=get_archon_html())
    
    @app.get("/archon/status") 
    async def archon_status():
        return {
            "status": "active",
            "interface": "html_web_interface", 
            "message": "Archon MCP running via FastAPI routes",
            "endpoints": {
                "interface": "/archon",
                "status": "/archon/status"
            }
        }
    
    print("✅ Archon routes added successfully")