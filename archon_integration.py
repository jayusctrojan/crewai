"""
FastAPI-Streamlit Integration for Archon MCP
Adds Streamlit endpoints to existing FastAPI application
"""

import asyncio
import subprocess
import threading
import os
import time
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import streamlit.web.bootstrap as bootstrap
from streamlit.web.server import Server
from streamlit import config as st_config

class StreamlitIntegrator:
    def __init__(self, app, main_port=8000):
        self.app = app
        self.main_port = main_port
        self.streamlit_port = 8501
        self.streamlit_process = None
        self.is_running = False
        
    def setup_streamlit_routes(self):
        """Add Streamlit routes to FastAPI app"""
        
        @self.app.get("/archon", response_class=HTMLResponse)
        async def archon_redirect():
            """Redirect to Archon Streamlit interface"""
            return RedirectResponse(f"http://localhost:{self.streamlit_port}", status_code=302)
        
        @self.app.get("/archon/status")
        async def archon_status():
            """Check if Archon Streamlit is running"""
            return {
                "streamlit_running": self.is_running,
                "streamlit_port": self.streamlit_port,
                "archon_url": f"http://localhost:{self.streamlit_port}",
                "integration_active": True
            }
        
        @self.app.post("/archon/start")
        async def start_archon():
            """Start Archon Streamlit interface"""
            if self.is_running:
                return {"message": "Archon is already running", "status": "running"}
            
            try:
                await self.start_streamlit()
                return {
                    "message": "Archon started successfully",
                    "status": "running",
                    "url": f"http://localhost:{self.streamlit_port}"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start Archon: {str(e)}")
        
        @self.app.post("/archon/stop")
        async def stop_archon():
            """Stop Archon Streamlit interface"""
            if not self.is_running:
                return {"message": "Archon is not running", "status": "stopped"}
            
            try:
                await self.stop_streamlit()
                return {"message": "Archon stopped successfully", "status": "stopped"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to stop Archon: {str(e)}")
        
        # Add Archon info to your existing health endpoint
        @self.app.get("/health/archon")
        async def archon_health():
            """Detailed health check for Archon integration"""
            return {
                "archon_integration": "active",
                "streamlit_available": True,
                "streamlit_running": self.is_running,
                "streamlit_port": self.streamlit_port,
                "archon_endpoints": {
                    "redirect": "/archon",
                    "status": "/archon/status", 
                    "start": "/archon/start",
                    "stop": "/archon/stop",
                    "health": "/health/archon"
                }
            }
        
    async def start_streamlit(self):
        """Start Streamlit in a separate thread"""
        if self.is_running:
            return
            
        def run_streamlit():
            try:
                # Configure Streamlit
                os.environ['STREAMLIT_SERVER_PORT'] = str(self.streamlit_port)
                os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
                os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
                os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
                
                # Start Streamlit process
                cmd = [
                    'streamlit', 'run', 'archon_app.py',
                    '--server.port', str(self.streamlit_port),
                    '--server.address', '0.0.0.0',
                    '--server.headless', 'true',
                    '--server.enableCORS', 'false',
                    '--server.fileWatcherType', 'none'
                ]
                
                self.streamlit_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.is_running = True
                print(f"🏛️ Archon Streamlit started on port {self.streamlit_port}")
                
                # Keep the process running
                self.streamlit_process.wait()
                
            except Exception as e:
                print(f"❌ Error running Streamlit: {e}")
                self.is_running = False
        
        # Start in background thread
        streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        streamlit_thread.start()
        
        # Give it time to start
        await asyncio.sleep(3)
        
    async def stop_streamlit(self):
        """Stop Streamlit process"""
        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
            
            self.streamlit_process = None
            self.is_running = False
            print("🏛️ Archon Streamlit stopped")

# Global integrator instance
streamlit_integrator = None

def setup_archon_integration(app):
    """Setup Archon-Streamlit integration with FastAPI app"""
    global streamlit_integrator
    
    try:
        streamlit_integrator = StreamlitIntegrator(app)
        streamlit_integrator.setup_streamlit_routes()
        print("✅ Archon integration setup complete")
        return streamlit_integrator
    except Exception as e:
        print(f"❌ Failed to setup Archon integration: {e}")
        return None
