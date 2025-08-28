# plugins/agents/youtube_processor.py
from crewai import Agent, Task
from typing import Dict, Any

class YouTubeProcessor:
    """YouTube Educational Content Specialist Agent"""
    
    @staticmethod
    def create_agent(llm=None, knowledge_context=""):
        backstory = """You are a specialist in online educational content who excels at extracting 
                    maximum value from video-based learning materials. You identify key teaching moments, 
                    practical examples, and actionable business insights from YouTube educational videos.
                    
                    You understand video structure, can identify timestamps for important sections,
                    and excel at converting video content into searchable, actionable business knowledge."""
        
        if knowledge_context:
            backstory += f"\n\nYou have access to relevant course materials:\n{knowledge_context}"
        
        return Agent(
            role="YouTube Educational Content Specialist",
            goal="Analyze YouTube educational content and extract structured learning materials for business applications",
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
        
        # Extract YouTube URL and metadata from task description or context
        youtube_url = context_data.get('youtube_url', 'URL not provided')
        course_name = context_data.get('course_name', source_info.get('courseName', 'Unknown Course'))
        module_name = context_data.get('module_name', source_info.get('moduleName', 'Unknown Module'))
        
        return [
            Task(
                description=f"""
                Process this YouTube educational video and extract comprehensive business-focused analysis:
                
                1. **Content Structure**: Break down main topics and subtopics with approximate timestamps
                2. **Key Learning Points**: Most important takeaways and insights
                3. **Business Applications**: How can this content be applied in real business contexts?
                4. **Tools & Resources**: Any tools, software, frameworks, or resources mentioned
                5. **Case Studies**: Real examples, success stories, or case studies presented
                6. **Action Items**: Specific steps viewers should take after watching
                7. **Target Audience**: Who would benefit most from this content?
                8. **Industry Relevance**: Which business sectors/departments would find this valuable?
                9. **Follow-up Content**: What additional learning might be needed?
                10. **Searchable Tags**: Keywords for easy discovery in knowledge base
                
                **YouTube URL**: {youtube_url}
                **Course Context**: {course_name} - {module_name}
                **Analysis Request**: {task_description}
                **Additional Context**: {context_data}
                
                Focus on extracting business value and making this content discoverable for professionals.
                """,
                agent=agent,
                expected_output="""
                Comprehensive YouTube video analysis containing:
                - Structured content breakdown with timestamps
                - 5-7 key business takeaways
                - Practical application scenarios by industry
                - Tools/resources inventory
                - Case study summaries
                - Actionable next steps checklist
                - Target audience profile
                - Industry/department relevance mapping
                - Related content suggestions
                - 10-15 searchable keywords/tags
                """
            )
        ]

# Register this agent
AGENT_CLASS = YouTubeProcessor
AGENT_NAME = "youtube_processor"