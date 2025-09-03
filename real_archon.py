"""
Real Archon MCP Integration
Integrates the official Archon agent builder from https://github.com/coleam00/Archon
"""

import asyncio
import subprocess
import os
import json
import shutil
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import RedirectResponse
import threading
import time

class RealArchonIntegrator:
    def __init__(self):
        self.archon_dir = Path("archon")
        self.is_installed = False
        self.is_running = False
        self.streamlit_process = None
        
    async def setup_archon(self):
        """Setup real Archon from GitHub repository"""
        try:
            # Check if already cloned
            if self.archon_dir.exists():
                print("🏛️ Archon directory exists, pulling latest updates...")
                result = subprocess.run(
                    ["git", "pull"], 
                    cwd=self.archon_dir, 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    print("✅ Archon updated successfully")
                else:
                    print(f"⚠️ Git pull failed: {result.stderr}")
            else:
                print("🏛️ Cloning Archon repository...")
                result = subprocess.run([
                    "git", "clone", 
                    "https://github.com/coleam00/Archon.git", 
                    str(self.archon_dir)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"Failed to clone Archon: {result.stderr}")
                print("✅ Archon repository cloned successfully")
            
            # Install Archon dependencies
            requirements_path = self.archon_dir / "requirements.txt"
            if requirements_path.exists():
                print("📦 Installing Archon dependencies...")
                result = subprocess.run([
                    "pip", "install", "-r", str(requirements_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"⚠️ Some dependencies may have failed: {result.stderr}")
                else:
                    print("✅ Archon dependencies installed")
            
            # Setup workbench directory
            workbench_dir = self.archon_dir / "workbench"
            workbench_dir.mkdir(exist_ok=True)
            
            # Create basic environment configuration
            env_vars_path = workbench_dir / "env_vars.json"
            if not env_vars_path.exists():
                env_config = {
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                    "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
                    "SUPABASE_KEY": os.getenv("SUPABASE_KEY", ""),
                    "MODEL_PROVIDER": "openai",
                    "PRIMARY_MODEL": "gpt-4",
                    "REASONING_MODEL": "gpt-4"
                }
                
                with open(env_vars_path, 'w') as f:
                    json.dump(env_config, f, indent=2)
                print("✅ Environment configuration created")
            
            self.is_installed = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup Archon: {e}")
            return False
    
    async def start_archon_service(self):
        """Start the real Archon Streamlit interface"""
        if not self.is_installed:
            setup_success = await self.setup_archon()
            if not setup_success:
                return False
        
        if self.is_running:
            return True
            
        try:
            # Start Archon Streamlit UI in background
            def run_archon():
                try:
                    streamlit_script = self.archon_dir / "streamlit_ui.py"
                    if not streamlit_script.exists():
                        print(f"❌ Streamlit script not found at {streamlit_script}")
                        return
                    
                    # Change to Archon directory and run
                    os.chdir(self.archon_dir)
                    
                    self.streamlit_process = subprocess.Popen([
                        "streamlit", "run", "streamlit_ui.py",
                        "--server.port", "8502",
                        "--server.address", "0.0.0.0",
                        "--server.headless", "true"
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    self.is_running = True
                    print("🏛️ Real Archon started on port 8502")
                    
                    # Wait for process
                    self.streamlit_process.wait()
                    
                except Exception as e:
                    print(f"❌ Error starting Archon: {e}")
                    self.is_running = False
            
            # Start in background thread
            archon_thread = threading.Thread(target=run_archon, daemon=True)
            archon_thread.start()
            
            # Give it time to start
            await asyncio.sleep(5)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Archon service: {e}")
            return False
    
    async def stop_archon_service(self):
        """Stop the Archon service"""
        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
            
            self.streamlit_process = None
            self.is_running = False
            print("🏛️ Archon service stopped")
        
        return True
    
    def get_status(self):
        """Get current Archon status"""
        return {
            "installed": self.is_installed,
            "running": self.is_running,
            "archon_url": "http://localhost:8502" if self.is_running else None,
            "repository": "https://github.com/coleam00/Archon",
            "version": "v6-tool-library-integration"
        }

# Global Archon integrator
archon_integrator = RealArchonIntegrator()

def setup_real_archon_routes(app):
    """Setup routes for real Archon integration"""
    
    @app.get("/archon")
    async def archon_interface():
        """Redirect to real Archon interface"""
        if not archon_integrator.is_running:
            # Try to start Archon
            started = await archon_integrator.start_archon_service()
            if not started:
                raise HTTPException(
                    status_code=503, 
                    detail="Archon service is not available. Please check logs and try again."
                )
        
        return RedirectResponse(url="http://localhost:8502", status_code=302)
    
    @app.get("/archon/status")
    async def archon_status():
        """Get real Archon status"""
        return archon_integrator.get_status()
    
    @app.post("/archon/setup")
    async def setup_archon():
        """Setup real Archon from GitHub"""
        try:
            success = await archon_integrator.setup_archon()
            if success:
                return {
                    "success": True,
                    "message": "Archon setup completed successfully",
                    "status": archon_integrator.get_status()
                }
            else:
                raise HTTPException(status_code=500, detail="Archon setup failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Setup error: {str(e)}")
    
    @app.post("/archon/start")
    async def start_archon():
        """Start real Archon service"""
        try:
            success = await archon_integrator.start_archon_service()
            if success:
                return {
                    "success": True,
                    "message": "Archon service started successfully",
                    "url": "http://localhost:8502",
                    "status": archon_integrator.get_status()
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to start Archon service")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Start error: {str(e)}")
    
    @app.post("/archon/stop")
    async def stop_archon():
        """Stop real Archon service"""
        try:
            success = await archon_integrator.stop_archon_service()
            return {
                "success": True,
                "message": "Archon service stopped successfully",
                "status": archon_integrator.get_status()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stop error: {str(e)}")
    
    print("✅ Real Archon routes setup complete")
    return archon_integrator

# Export the setup function
real_archon_setup = setup_real_archon_routes