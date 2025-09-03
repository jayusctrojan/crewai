import streamlit as st
import json
import asyncio
import os
from typing import Dict, List, Optional
import anthropic
from datetime import datetime
import concurrent.futures
import threading

# Configure Streamlit
st.set_page_config(
    page_title="🏛️ Archon - AI Agent Builder",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1E88E5;
    }
    .agent-card {
        border: 2px solid #E3F2FD;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-stopped {
        color: #F44336;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class ArchonMCP:
    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.anthropic_client = None
        
        # Initialize Anthropic client if API key is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
    
    def create_agent(self, name: str, role: str, goal: str, backstory: str, tools: List[str] = None) -> Dict:
        """Create a new AI agent"""
        agent_id = f"agent_{len(self.agents) + 1}"
        agent = {
            "id": agent_id,
            "name": name,
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "tools": tools or [],
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "conversations": []
        }
        self.agents[agent_id] = agent
        return agent
    
    def list_agents(self) -> List[Dict]:
        """List all agents"""
        return list(self.agents.values())
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get specific agent by ID"""
        return self.agents.get(agent_id)
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = status
    
    def chat_with_agent(self, agent_id: str, message: str) -> str:
        """Chat with a specific agent using Anthropic Claude"""
        agent = self.get_agent(agent_id)
        if not agent:
            return "Agent not found"
        
        if not self.anthropic_client:
            return "Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
        
        try:
            # Create a prompt that embodies the agent's characteristics
            system_prompt = f"""
            You are {agent['name']}, an AI agent with the following characteristics:
            
            Role: {agent['role']}
            Goal: {agent['goal']}
            Backstory: {agent['backstory']}
            Tools Available: {', '.join(agent['tools']) if agent['tools'] else 'None'}
            
            Respond to the user's message while staying in character and working towards your goal.
            Be helpful, professional, and true to your role and backstory.
            """
            
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": message}]
            )
            
            agent_response = response.content[0].text
            
            # Log the conversation
            conversation = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "agent_response": agent_response
            }
            agent["conversations"].append(conversation)
            
            return agent_response
            
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"

# Initialize Archon MCP
@st.cache_resource
def get_archon():
    return ArchonMCP()

archon = get_archon()

# Main App
def main():
    st.markdown('<h1 class="main-header">🏛️ Archon - AI Agent Builder</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🏛️ Archon Control Panel")
        
        page = st.selectbox(
            "Navigate",
            ["🏠 Dashboard", "🤖 Create Agent", "💬 Chat with Agents", "📊 Analytics", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Quick stats
        total_agents = len(archon.list_agents())
        active_agents = sum(1 for agent in archon.list_agents() if agent["status"] == "running")
        
        st.metric("Total Agents", total_agents)
        st.metric("Active Agents", active_agents)
        st.metric("Conversations", sum(len(agent.get("conversations", [])) for agent in archon.list_agents()))
    
    # Main content
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🤖 Create Agent":
        create_agent_page()
    elif page == "💬 Chat with Agents":
        chat_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

def dashboard_page():
    st.header("🏠 Dashboard")
    
    # Agent overview
    agents = archon.list_agents()
    
    if not agents:
        st.info("👋 Welcome to Archon! Create your first AI agent to get started.")
        if st.button("Create Your First Agent"):
            st.session_state.page = "🤖 Create Agent"
            st.experimental_rerun()
    else:
        st.subheader("Your AI Agents")
        
        cols = st.columns(min(3, len(agents)))
        for i, agent in enumerate(agents):
            with cols[i % 3]:
                status_class = "status-running" if agent["status"] == "running" else "status-stopped"
                st.markdown(f"""
                <div class="agent-card">
                    <h3>🤖 {agent['name']}</h3>
                    <p><strong>Role:</strong> {agent['role']}</p>
                    <p><strong>Status:</strong> <span class="{status_class}">{agent['status'].title()}</span></p>
                    <p><strong>Conversations:</strong> {len(agent.get('conversations', []))}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Chat with {agent['name']}", key=f"chat_{agent['id']}"):
                    st.session_state.selected_agent = agent['id']
                    st.session_state.page = "💬 Chat with Agents"
                    st.experimental_rerun()

def create_agent_page():
    st.header("🤖 Create New AI Agent")
    
    with st.form("create_agent_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Agent Name", placeholder="e.g., ResearchBot")
            role = st.text_input("Agent Role", placeholder="e.g., Research Assistant")
        
        with col2:
            goal = st.text_area("Agent Goal", placeholder="e.g., Help users find accurate information and citations")
            backstory = st.text_area("Agent Backstory", placeholder="e.g., You are an experienced researcher with expertise in...")
        
        # Tools selection
        st.subheader("Available Tools")
        tools = st.multiselect(
            "Select tools for your agent",
            ["web_search", "code_executor", "file_analyzer", "data_processor", "image_generator"],
            default=["web_search"]
        )
        
        submitted = st.form_submit_button("Create Agent", type="primary")
        
        if submitted:
            if name and role and goal and backstory:
                agent = archon.create_agent(name, role, goal, backstory, tools)
                st.success(f"✅ Agent '{name}' created successfully!")
                st.json(agent)
                
                # Auto-start the agent
                archon.update_agent_status(agent['id'], "running")
                st.info(f"🚀 Agent '{name}' is now running and ready to chat!")
            else:
                st.error("Please fill in all required fields.")

def chat_page():
    st.header("💬 Chat with Your AI Agents")
    
    agents = archon.list_agents()
    if not agents:
        st.warning("No agents available. Create an agent first!")
        return
    
    # Agent selection
    agent_options = {f"{agent['name']} ({agent['role']})": agent['id'] for agent in agents}
    selected_agent_display = st.selectbox("Select an agent to chat with", list(agent_options.keys()))
    
    if selected_agent_display:
        selected_agent_id = agent_options[selected_agent_display]
        agent = archon.get_agent(selected_agent_id)
        
        # Display agent info
        with st.expander("Agent Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {agent['name']}")
                st.write(f"**Role:** {agent['role']}")
            with col2:
                st.write(f"**Goal:** {agent['goal']}")
                st.write(f"**Tools:** {', '.join(agent['tools']) if agent['tools'] else 'None'}")
        
        # Chat interface
        st.subheader(f"Chat with {agent['name']}")
        
        # Display conversation history
        conversations = agent.get('conversations', [])
        for conv in conversations[-5:]:  # Show last 5 conversations
            with st.chat_message("user"):
                st.write(conv['user_message'])
            with st.chat_message("assistant"):
                st.write(conv['agent_response'])
        
        # Chat input
        user_message = st.chat_input("Type your message here...")
        
        if user_message:
            # Display user message
            with st.chat_message("user"):
                st.write(user_message)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner(f"{agent['name']} is thinking..."):
                    response = archon.chat_with_agent(selected_agent_id, user_message)
                    st.write(response)

def analytics_page():
    st.header("📊 Analytics")
    
    agents = archon.list_agents()
    if not agents:
        st.info("No data available. Create and interact with agents to see analytics.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", len(agents))
    
    with col2:
        running_agents = sum(1 for agent in agents if agent["status"] == "running")
        st.metric("Running Agents", running_agents)
    
    with col3:
        total_conversations = sum(len(agent.get("conversations", [])) for agent in agents)
        st.metric("Total Conversations", total_conversations)
    
    # Agent performance
    st.subheader("Agent Activity")
    
    agent_data = []
    for agent in agents:
        agent_data.append({
            "Name": agent["name"],
            "Role": agent["role"],
            "Status": agent["status"],
            "Conversations": len(agent.get("conversations", [])),
            "Created": agent["created_at"][:10]
        })
    
    if agent_data:
        st.dataframe(agent_data)

def settings_page():
    st.header("⚙️ Settings")
    
    st.subheader("Environment Configuration")
    
    # Check API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        st.success("✅ Anthropic API Key configured")
    else:
        st.error("❌ Anthropic API Key not found")
        st.info("Set the ANTHROPIC_API_KEY environment variable to enable agent conversations.")
    
    st.subheader("Agent Settings")
    
    with st.expander("Default Agent Configuration"):
        st.text_input("Default Model", value="claude-3-5-sonnet-20241022", disabled=True)
        st.slider("Max Tokens", min_value=100, max_value=4000, value=1000, disabled=True)
        st.selectbox("Temperature", options=[0.1, 0.3, 0.7, 0.9], index=2, disabled=True)
    
    st.info("More settings coming soon! This is an early preview of Archon.")

if __name__ == "__main__":
    main()
