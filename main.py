import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=== Starting CrewAI Application ===")

try:
    logger.info("Loading environment variables...")
    load_dotenv()
    
    logger.info("Importing CrewAI...")
    from crewai import Agent, Task, Crew
    logger.info("CrewAI imported successfully")
    
    logger.info("Creating FastAPI app...")
    app = FastAPI(title="CrewAI API", version="1.0.0")
    logger.info("FastAPI app created")
    
    class TaskRequest(BaseModel):
        description: str
        role: str = "AI Assistant"
        goal: str = "Complete the given task effectively"

    @app.get("/")
    async def root():
        return {"message": "CrewAI API is running on Render!"}

    @app.post("/run-crew")
    async def run_crew(request: TaskRequest):
        try:
            logger.info(f"Creating agent with role: {request.role}")
            
            # Create agent
            agent = Agent(
                role=request.role,
                goal=request.goal,
                backstory=f"You are an expert {request.role} focused on {request.goal}",
                verbose=True
            )
            
            # Create task
            task = Task(
                description=request.description,
                agent=agent,
                expected_output="A detailed response completing the requested task"
            )
            
            # Create and run crew
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True
            )
            
            logger.info("Running crew...")
            result = crew.kickoff()
            logger.info("Crew execution completed")
            
            return {
                "success": True,
                "result": str(result),
                "agent_role": request.role
            }
            
        except Exception as e:
            logger.error(f"Error in run_crew: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "CrewAI API"}

    logger.info("Routes defined successfully")

except Exception as e:
    logger.error(f"Fatal error during startup: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        
        logger.info(f"Starting uvicorn server on port {port}")
        logger.info(f"OpenAI API Key exists: {bool(os.getenv('OPENAI_API_KEY'))}")
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start uvicorn: {str(e)}")
        sys.exit(1)