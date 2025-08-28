# plugins/agents/content_analyzer.py
from crewai import Agent, Task
from typing import Dict, Any

class ContentAnalyzer:
    """Educational Content Analysis Agent"""
    
    @staticmethod
    def create_agent(llm=None, knowledge_context=""):
        backstory = """You are an expert educational content analyst with 15+ years experience in 
                    curriculum design and learning outcome assessment. You excel at identifying the 
                    core learning objectives, difficulty levels, prerequisites, and practical applications 
                    of educational material across various industries and domains."""
        
        if knowledge_context:
            backstory += f"\n\nYou have access to relevant course materials:\n{knowledge_context}"
        
        return Agent(
            role="Educational Content Analyst",
            goal="Extract learning objectives, key concepts, and actionable insights from processed educational content",
            backstory=backstory,
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
    
    @staticmethod
    def create_tasks(agent, request_data: Dict[str, Any]):
        task_description = request_data.get('task_description', '')
        context_data = request_data.get('context_data', {})
        source_info = request_data.get('source_info', {})
        
        return [
            Task(
                description=f"""
                Analyze this educational content and extract:
                
                1. **Learning Objectives**: What specific skills or knowledge will learners gain?
                2. **Key Concepts**: Main ideas, frameworks, and theories presented
                3. **Difficulty Level**: Beginner, Intermediate, Advanced, or Expert
                4. **Prerequisites**: What knowledge/skills are assumed?
                5. **Practical Applications**: How can this be applied in real-world scenarios?
                6. **Industry Relevance**: Which business sectors would benefit most?
                7. **Action Items**: Specific steps learners should take after consuming this content
                
                **Content to Analyze**: {task_description}
                **Context Data**: {context_data}
                **Source Information**: {source_info}
                
                Provide a structured analysis that makes this content searchable and actionable for business professionals.
                """,
                agent=agent,
                expected_output="""
                Structured analysis containing:
                - Learning objectives (3-5 clear, measurable outcomes)
                - Key concepts list with brief explanations
                - Difficulty assessment with justification
                - Prerequisites checklist
                - 3-5 practical application scenarios
                - Industry relevance mapping
                - Actionable next steps for learners
                """
            )
        ]

# Register this agent
AGENT_CLASS = ContentAnalyzer
AGENT_NAME = "content_analyzer"