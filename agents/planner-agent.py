"""
Planner Agent implementation.

This agent is responsible for strategic planning, task decomposition,
and overall coordination of the problem-solving process.
"""

import logging
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from reasoning.tree_of_thoughts import TreeOfThoughtsReasoner
from utils.prompt_templates import PLANNER_SYSTEM_MESSAGE, PLANNER_TASK_TEMPLATE

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for decomposing complex tasks into manageable steps.
    
    The Planner serves as the strategic core of the system, receiving high-level
    user queries and breaking them down into structured sequences of sub-tasks.
    It can use either Chain-of-Thought or Tree-of-Thoughts reasoning depending
    on the complexity of the task.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Planner Agent with specialized capabilities."""
        super().__init__(name="Planner Agent", **kwargs)
        
        # Use Tree-of-Thoughts for complex planning by default
        if "reasoner" not in kwargs:
            self.reasoner = TreeOfThoughtsReasoner()
            
        logger.info(f"Initialized Planner Agent (ID: {self.agent_id})")
        
    async def process(self, query: Dict) -> Dict:
        """
        Process a user query by creating a structured plan.
        
        Args:
            query: Dictionary containing the user query and context
                - query_text: The raw query from the user
                - context: Optional additional context
                - complexity: Optional assessment of query complexity
                
        Returns:
            A dictionary containing the decomposed plan:
                - plan_id: Unique identifier for this plan
                - steps: List of steps to execute
                - context: Context for execution
                - reasoning: Reasoning process used to create the plan
        """
        query_text = query.get("query_text", "")
        context = query.get("context", {})
        complexity = query.get("complexity", "medium")  # simple, medium, complex
        
        logger.info(f"Planning process started for query: {query_text[:50]}...")
        
        # Select reasoning approach based on complexity
        if complexity == "complex":
            # For complex queries, use Tree of Thoughts
            reasoning_result = await self.reasoner.reason(query_text, context)
            reasoning_approach = "tree_of_thoughts"
        else:
            # For simpler queries, use Chain of Thought
            # Temporarily switch reasoner if needed
            original_reasoner = self.reasoner
            self.reasoner = ChainOfThoughtReasoner()
            reasoning_result = await self.reasoner.reason(query_text, context)
            self.reasoner = original_reasoner
            reasoning_approach = "chain_of_thought"
        
        # Generate plan using the reasoning result
        prompt = PLANNER_TASK_TEMPLATE.format(
            query=query_text,
            context=self._format_context(context),
            reasoning=reasoning_result.get("reasoning", ""),
            complexity=complexity
        )
        
        plan_response = await self.generate_response(
            prompt=prompt,
            system_message=PLANNER_SYSTEM_MESSAGE
        )
        
        # Parse the plan response into structured steps
        try:
            structured_plan = self._parse_plan(plan_response)
        except Exception as e:
            logger.error(f"Error parsing plan: {str(e)}")
            # Fallback: Use a simpler prompt and try again
            fallback_prompt = f"Create a simple step-by-step plan for: {query_text}"
            plan_response = await self.generate_response(
                prompt=fallback_prompt,
                system_message=PLANNER_SYSTEM_MESSAGE
            )
            structured_plan = self._parse_plan_simple(plan_response)
        
        # Store plan in memory
        if self.memory:
            await self.memory.store_plan(structured_plan)
        
        # Return the full plan details
        return {
            "plan_id": structured_plan["plan_id"],
            "steps": structured_plan["steps"],
            "context": structured_plan["context"],
            "reasoning": {
                "approach": reasoning_approach,
                "process": reasoning_result.get("reasoning", ""),
                "conclusion": reasoning_result.get("conclusion", "")
            }
        }
    
    def _format_context(self, context: Dict) -> str:
        """Format context dictionary into a string for prompting."""
        context_str = ""
        for key, value in context.items():
            context_str += f"{key}: {value}\n"
        return context_str
    
    def _parse_plan(self, plan_text: str) -> Dict:
        """
        Parse the generated plan text into a structured format.
        
        This includes plan steps, dependencies, and execution context.
        
        Args:
            plan_text: The raw plan text from the LLM
            
        Returns:
            A structured plan dictionary
        """
        import uuid
        import re
        
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps = []
        
        # Extract steps using regex pattern matching
        step_pattern = r"(?:Step|STEP)\s+(\d+):\s+(.*?)(?=(?:Step|STEP)\s+\d+:|$)"
        matches = re.finditer(step_pattern, plan_text, re.DOTALL)
        
        for match in matches:
            step_num = int(match.group(1))
            step_content = match.group(2).strip()
            
            # Try to identify tools needed for this step
            tools_needed = []
            tools_pattern = r"(?:Tool|TOOL)(?:s)?(?:\s+needed)?:\s+(.*?)(?=\n|$)"
            tools_match = re.search(tools_pattern, step_content)
            if tools_match:
                tools_text = tools_match.group(1)
                tools_needed = [t.strip() for t in tools_text.split(',')]
            
            steps.append({
                "step_id": f"{plan_id}_step_{step_num}",
                "step_number": step_num,
                "description": step_content,
                "tools_needed": tools_needed,
                "status": "pending",
                "dependencies": [step_num - 1] if step_num > 1 else []
            })
        
        # If no steps were found with regex, try a simpler approach
        if not steps:
            return self._parse_plan_simple(plan_text)
        
        # Extract high-level context
        context = {
            "goal": plan_text.split('\n')[0] if '\n' in plan_text else plan_text[:100],
            "constraints": [],
            "resources_needed": []
        }
        
        return {
            "plan_id": plan_id,
            "steps": sorted(steps, key=lambda x: x["step_number"]),
            "context": context
        }
    
    def _parse_plan_simple(self, plan_text: str) -> Dict:
        """Simpler fallback plan parser that splits by newlines."""
        import uuid
        
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps = []
        
        lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
        
        step_num = 1
        for line in lines:
            if not line:
                continue
                
            steps.append({
                "step_id": f"{plan_id}_step_{step_num}",
                "step_number": step_num,
                "description": line,
                "tools_needed": [],
                "status": "pending",
                "dependencies": [step_num - 1] if step_num > 1 else []
            })
            step_num += 1
        
        return {
            "plan_id": plan_id,
            "steps": steps,
            "context": {"goal": lines[0] if lines else ""}
        }
        
    async def replan(self, original_plan: Dict, feedback: Dict) -> Dict:
        """
        Adjust an existing plan based on feedback.
        
        This is called when the Critic agent provides feedback indicating
        that replanning is necessary.
        
        Args:
            original_plan: The original plan that needs adjustment
            feedback: Feedback from the Critic agent
            
        Returns:
            A revised plan
        """
        # Construct a prompt for replanning
        replan_prompt = f"""
        I need to revise a plan based on the following feedback:
        
        ORIGINAL PLAN:
        {original_plan}
        
        FEEDBACK:
        {feedback.get('feedback_text', 'No specific feedback provided')}
        
        ISSUES TO ADDRESS:
        {feedback.get('issues', [])}
        
        Please create a revised plan that addresses these issues while maintaining the original goal.
        """
        
        # Generate a revised plan
        revised_plan_text = await self.generate_response(
            prompt=replan_prompt,
            system_message=PLANNER_SYSTEM_MESSAGE
        )
        
        # Parse the revised plan
        revised_plan = self._parse_plan(revised_plan_text)
        
        # Keep the original plan ID for continuity
        revised_plan["plan_id"] = original_plan["plan_id"]
        revised_plan["version"] = original_plan.get("version", 1) + 1
        revised_plan["previous_version"] = original_plan
        
        return revised_plan
