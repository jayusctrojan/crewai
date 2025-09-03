"""
Simple Archon Web Interface - Embedded in FastAPI
Direct HTML interface for agent building without Streamlit complexity
"""

from fastapi import Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
import json

def setup_archon_web_interface(app):
    """Add Archon web interface directly to FastAPI"""
    
    @app.get("/archon", response_class=HTMLResponse)
    async def archon_interface(request: Request):
        """Archon Agent Builder Web Interface"""
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
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            .header h1 {
                font-size: 2rem;
            }
            .container {
                padding: 15px;
            }
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
                        <label for="agentBackstory">Agent Backstory</label>
                        <textarea id="agentBackstory" name="agentBackstory" placeholder="Describe the agent's background and expertise..." required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="llmModel">LLM Model</label>
                        <select id="llmModel" name="llmModel">
                            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                            <option value="gpt-4">GPT-4</option>
                            <option value="gpt-4-turbo">GPT-4 Turbo</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn" id="createBtn">Create Agent</button>
                </form>
            </div>

            <!-- Agent Management -->
            <div class="card">
                <h2>🎯 Agent Management</h2>
                <div id="agentList">
                    <div class="loading">No agents created yet. Create your first agent!</div>
                </div>
                <button class="btn" onclick="loadAgents()" id="refreshBtn">Refresh Agents</button>
            </div>

            <!-- Chat Interface -->
            <div class="card chat-container">
                <h2>💬 Chat with Agent</h2>
                <div id="chatMessages">
                    <div class="loading">Select an agent to start chatting</div>
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

        // Create Agent
        document.getElementById('agentForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const agentData = {
                agent_name: formData.get('agentName'),
                agent_role: formData.get('agentRole'),
                agent_goal: formData.get('agentGoal'),
                task_description: `You are ${formData.get('agentName')}, a ${formData.get('agentRole')}. ${formData.get('agentBackstory')}`,
                expected_output: "A helpful and informative response based on your expertise",
                llm_model: formData.get('llmModel')
            };

            const createBtn = document.getElementById('createBtn');
            createBtn.disabled = true;
            createBtn.textContent = 'Creating...';

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
                    agents.push({
                        name: agentData.agent_name,
                        role: agentData.agent_role,
                        goal: agentData.agent_goal,
                        backstory: formData.get('agentBackstory'),
                        model: agentData.llm_model,
                        created: new Date().toLocaleString()
                    });
                    
                    updateAgentList();
                    e.target.reset();
                    showNotification('Agent created successfully!', 'success');
                } else {
                    throw new Error('Failed to create agent');
                }
            } catch (error) {
                console.error('Error creating agent:', error);
                showNotification('Failed to create agent', 'error');
            } finally {
                createBtn.disabled = false;
                createBtn.textContent = 'Create Agent';
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
                <div class="agent-item" onclick="selectAgent(${index})">
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
            chatHistory = [];
            updateChatInterface();
            
            document.getElementById('chatInput').disabled = false;
            document.getElementById('sendBtn').disabled = false;
            
            showNotification(`Selected agent: ${selectedAgent.name}`, 'info');
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
            sendBtn.textContent = 'Thinking...';

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
                    throw new Error('Failed to get response');
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
                    : '<div class="loading">Select an agent to start chatting</div>';
                return;
            }

            chatMessages.innerHTML = chatHistory.map(msg => `
                <div class="chat-message ${msg.type}">
                    ${msg.content}
                </div>
            `).join('');
            
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Show Notification
        function showNotification(message, type = 'info') {
            // Simple notification system
            const notification = document.createElement('div');
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
                color: white;
                padding: 15px 20px;
                border-radius: 5px;
                z-index: 1000;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 3000);
        }

        // Load agents on page load
        function loadAgents() {
            updateAgentList();
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
            "endpoints": {
                "interface": "/archon",
                "status": "/archon/status"
            }
        }

    print("✅ Archon web interface setup complete")

# Export setup function
archon_web_setup = setup_archon_web_interface