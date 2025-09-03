#!/usr/bin/env python3
"""
Multi-Service Startup Script for CrewAI + Archon Integration
Runs both the CrewAI FastAPI server and Archon Streamlit app simultaneously
"""

import subprocess
import threading
import time
import signal
import sys
import os
from concurrent.futures import ThreadPoolExecutor

class ServiceManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def run_crewai_server(self):
        """Run the CrewAI FastAPI server on port 8000"""
        try:
            print("🚀 Starting CrewAI FastAPI server on port 8000...")
            
            # Import and run your existing main.py
            import main
            
        except Exception as e:
            print(f"❌ Error starting CrewAI server: {e}")
            return False
    
    def run_archon_streamlit(self):
        """Run the Archon Streamlit app on port 8501"""
        try:
            print("🏛️ Starting Archon Streamlit app on port 8501...")
            
            # Set Streamlit configuration
            os.environ['STREAMLIT_SERVER_PORT'] = '8501'
            os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
            os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
            os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
            os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
            
            # Run streamlit
            cmd = [
                'streamlit', 'run', 'archon_app.py',
                '--server.port', '8501',
                '--server.address', '0.0.0.0',
                '--server.headless', 'true',
                '--server.enableCORS', 'false',
                '--server.enableXsrfProtection', 'false'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            
            # Monitor the process
            while self.running and process.poll() is None:
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error starting Archon Streamlit: {e}")
            return False
    
    def start_services(self):
        """Start both services concurrently"""
        print("🔄 Starting multi-service application...")
        print("📡 CrewAI API will be available on: http://0.0.0.0:8000")
        print("🏛️ Archon UI will be available on: http://0.0.0.0:8501")
        print("-" * 60)
        
        # Use ThreadPoolExecutor to run services concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start CrewAI server
            crewai_future = executor.submit(self.run_crewai_server)
            
            # Give CrewAI a moment to start
            time.sleep(2)
            
            # Start Archon Streamlit
            archon_future = executor.submit(self.run_archon_streamlit)
            
            try:
                # Keep the main thread alive
                while self.running:
                    time.sleep(1)
                    
                    # Check if either service failed
                    if crewai_future.done() and not crewai_future.result():
                        print("❌ CrewAI server failed")
                        break
                        
                    if archon_future.done() and not archon_future.result():
                        print("❌ Archon Streamlit failed")
                        break
                        
            except KeyboardInterrupt:
                print("\n⏹️ Shutting down services...")
                self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all services"""
        self.running = False
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"Error terminating process: {e}")
        
        print("✅ All services stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🔄 Received signal {signum}, shutting down...")
    service_manager.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    service_manager = ServiceManager()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        service_manager.start_services()
    except Exception as e:
        print(f"❌ Startup error: {e}")
        service_manager.shutdown()
        sys.exit(1)
