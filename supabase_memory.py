"""
Supabase Agent Memory Manager
Handles agent memory, knowledge, and performance tracking
"""
import os
import json
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

class AgentMemoryManager:
    def __init__(self):
        """Initialize Supabase connection for agent memory"""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        self.supabase: Client = create_client(url, key)
        logger.info("AgentMemoryManager initialized with Supabase")

    def save_agent_memory(
        self, 
        agent_name: str, 
        agent_role: str, 
        content: str, 
        memory_type: str = "context",
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save agent memory to Supabase"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())

            data = {
                "agent_name": agent_name,
                "agent_role": agent_role,
                "session_id": session_id,
                "memory_type": memory_type,
                "content": content,
                "metadata": metadata or {}
            }

            result = self.supabase.table("agent_memory").insert(data).execute()
            logger.info(f"Saved memory for agent {agent_name}: {memory_type}")
            return True

        except Exception as e:
            logger.error(f"Error saving agent memory: {str(e)}")
            return False

    def get_agent_memory(
        self, 
        agent_name: str, 
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Retrieve agent memory from Supabase"""
        try:
            query = self.supabase.table("agent_memory").select("*")
            
            # Filter by agent name
            query = query.eq("agent_name", agent_name)
            
            # Optional filters
            if memory_type:
                query = query.eq("memory_type", memory_type)
            if session_id:
                query = query.eq("session_id", session_id)
            
            # Order and limit
            query = query.order("created_at", desc=True).limit(limit)
            
            result = query.execute()
            return result.data

        except Exception as e:
            logger.error(f"Error retrieving agent memory: {str(e)}")
            return []

    def save_agent_knowledge(
        self,
        agent_name: str,
        knowledge_content: str,
        knowledge_type: str = "fact",
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence_score: float = 1.0
    ) -> bool:
        """Save reusable knowledge for an agent"""
        try:
            data = {
                "agent_name": agent_name,
                "knowledge_type": knowledge_type,
                "knowledge_content": knowledge_content,
                "source": source,
                "confidence_score": confidence_score,
                "tags": tags or []
            }

            result = self.supabase.table("agent_knowledge").insert(data).execute()
            logger.info(f"Saved knowledge for agent {agent_name}: {knowledge_type}")
            return True

        except Exception as e:
            logger.error(f"Error saving agent knowledge: {str(e)}")
            return False

    def get_agent_knowledge(
        self,
        agent_name: str,
        knowledge_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """Retrieve agent knowledge from Supabase"""
        try:
            query = self.supabase.table("agent_knowledge").select("*")
            query = query.eq("agent_name", agent_name).eq("is_active", True)
            
            if knowledge_type:
                query = query.eq("knowledge_type", knowledge_type)
            
            if tags:
                for tag in tags:
                    query = query.contains("tags", [tag])
            
            result = query.order("confidence_score", desc=True).execute()
            return result.data

        except Exception as e:
            logger.error(f"Error retrieving agent knowledge: {str(e)}")
            return []

    def log_task_execution(
        self,
        agent_name: str,
        agent_role: str,
        task_description: str,
        actual_output: str,
        execution_time_ms: int,
        model_used: str,
        success: bool = True,
        error_message: Optional[str] = None,
        expected_output: Optional[str] = None
    ) -> bool:
        """Log task execution for performance tracking"""
        try:
            data = {
                "agent_name": agent_name,
                "agent_role": agent_role,
                "task_description": task_description,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "execution_time_ms": execution_time_ms,
                "model_used": model_used,
                "success": success,
                "error_message": error_message
            }

            result = self.supabase.table("task_executions").insert(data).execute()
            logger.info(f"Logged task execution for {agent_name}")
            return True

        except Exception as e:
            logger.error(f"Error logging task execution: {str(e)}")
            return False

    def get_agent_performance(self, agent_name: str, days: int = 7) -> Dict[str, Any]:
        """Get agent performance metrics for the last N days"""
        try:
            # Get recent executions
            result = self.supabase.table("task_executions")\
                .select("*")\
                .eq("agent_name", agent_name)\
                .gte("created_at", f"now() - interval '{days} days'")\
                .execute()

            executions = result.data
            
            if not executions:
                return {"total_tasks": 0, "success_rate": 0, "avg_time": 0}

            # Calculate metrics
            total_tasks = len(executions)
            successful_tasks = len([e for e in executions if e["success"]])
            success_rate = (successful_tasks / total_tasks) * 100
            
            execution_times = [e["execution_time_ms"] for e in executions if e["execution_time_ms"]]
            avg_time = sum(execution_times) / len(execution_times) if execution_times else 0

            return {
                "total_tasks": total_tasks,
                "success_rate": round(success_rate, 2),
                "avg_execution_time_ms": round(avg_time, 2),
                "avg_execution_time_seconds": round(avg_time / 1000, 2),
                "last_7_days": True
            }

        except Exception as e:
            logger.error(f"Error getting agent performance: {str(e)}")
            return {"error": str(e)}

# Global instance
memory_manager = None

def get_memory_manager() -> Optional[AgentMemoryManager]:
    """Get global memory manager instance"""
    global memory_manager
    try:
        if memory_manager is None:
            memory_manager = AgentMemoryManager()
        return memory_manager
    except Exception as e:
        logger.error(f"Could not initialize memory manager: {str(e)}")
        return None