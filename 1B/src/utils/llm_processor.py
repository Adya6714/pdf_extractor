# src/utils/llm_processor.py
"""
Enhanced LLM processor for different types of tasks
"""

import logging
from llama_cpp import Llama
from src.models.document_models import PersonaProfile

logger = logging.getLogger(__name__)

class LLMProcessor:
    """Handles LLM operations for content generation"""
    
    def __init__(self, model_path: str):
        logger.info(f"Loading LLM from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0
        )
        
        # Task-specific prompts
        self.task_prompts = {
            "travel": self._travel_prompt,
            "research": self._research_prompt,
            "analysis": self._analysis_prompt,
            "hr": self._hr_prompt,
            "default": self._default_prompt
        }
    
    def generate_task_response(self, context: str, persona: PersonaProfile, task: str) -> str:
        """Generate response based on task type"""
        # Determine task type
        task_type = self._identify_task_type(task)
        prompt_func = self.task_prompts.get(task_type, self.task_prompts["default"])
        
        # Generate prompt
        prompt = prompt_func(context, persona, task)
        
        # Generate response
        return self.generate_response(prompt, max_tokens=1500)
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from LLM"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            echo=False
        )
        return response['choices'][0]['text'].strip()
    
    def _identify_task_type(self, task: str) -> str:
        """Identify the type of task"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["travel", "trip", "itinerary", "visit"]):
            return "travel"
        elif any(word in task_lower for word in ["research", "study", "investigate", "analyze"]):
            return "research"
        elif any(word in task_lower for word in ["business", "market", "competitive", "strategy"]):
            return "analysis"
        elif any(word in task_lower for word in ["hr", "employee", "onboarding", "policy"]):
            return "hr"
        else:
            return "default"
    
    def _travel_prompt(self, context: str, persona: PersonaProfile, task: str) -> str:
        return f"""You are a {persona.role} with the task: {task}

Based on the following information, create a detailed travel plan.

Context Information:
{context}

Please create a comprehensive response that includes:
1. Day-by-day itinerary with specific times
2. Accommodation recommendations with price ranges
3. Restaurant suggestions for each meal
4. Activities and attractions with estimated costs
5. Transportation details
6. Total budget estimate
7. Important tips and cultural notes

Format your response clearly with headers and bullet points.

Response:"""
    
    def _research_prompt(self, context: str, persona: PersonaProfile, task: str) -> str:
        return f"""You are a {persona.role} with the task: {task}

Based on the following research materials:

{context}

Please provide:
1. Executive Summary
2. Key Findings
3. Methodology Overview
4. Detailed Analysis
5. Conclusions
6. Recommendations
7. Areas for Further Research

Use academic style and cite information where relevant.

Response:"""
    
    def _analysis_prompt(self, context: str, persona: PersonaProfile, task: str) -> str:
        return f"""You are a {persona.role} with the task: {task}

Analyze the following business information:

{context}

Provide:
1. Market Overview
2. Competitive Analysis
3. SWOT Analysis
4. Key Metrics and Trends
5. Strategic Recommendations
6. Risk Assessment
7. Implementation Roadmap

Use business terminology and data-driven insights.

Response:"""
    
    def _hr_prompt(self, context: str, persona: PersonaProfile, task: str) -> str:
        return f"""You are a {persona.role} with the task: {task}

Based on the HR documentation:

{context}

Create:
1. Process Overview
2. Step-by-Step Guidelines
3. Required Documentation
4. Compliance Checklist
5. Timeline and Milestones
6. Key Contacts and Resources
7. FAQs

Ensure clarity and compliance focus.

Response:"""
    
    def _default_prompt(self, context: str, persona: PersonaProfile, task: str) -> str:
        return f"""You are a {persona.role} with the task: {task}

Based on the following information:

{context}

Please provide a comprehensive response that addresses all aspects of the task.
Structure your response with clear sections and actionable insights.

Response:"""