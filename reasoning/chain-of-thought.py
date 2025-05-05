"""
Chain of Thought (CoT) reasoning implementation.

This module provides a structured approach for guiding LLMs through step-by-step
reasoning processes, enhancing their ability to solve complex problems.
"""

import logging
from typing import Dict, Any, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class ChainOfThoughtReasoner:
    """
    Chain of Thought (CoT) reasoning implementation.
    
    Chain of Thought prompting guides LLMs to generate linear, step-by-step
    reasoning paths, suitable for tasks with clear sequential dependencies.
    This implementation structures and formalizes the CoT reasoning approach.
    """
    
    def __init__(self, 
                model: str = "gpt-4",
                temperature: float = 0.2,
                max_tokens: int = 1500,
                max_steps: int = 5):
        """
        Initialize the Chain of Thought reasoner.
        
        Args:
            model: The LLM model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            max_steps: Maximum number of reasoning steps
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        
        logger.info("Chain of Thought reasoner initialized")
    
    @retry(
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def reason(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Apply Chain of Thought reasoning to a query.
        
        Args:
            query: The question or problem to reason about
            context: Additional context information
            
        Returns:
            A dictionary containing:
            - reasoning: The step-by-step reasoning process
            - conclusion: The final answer or conclusion
        """
        context = context or {}
        
        # Create a prompt that encourages step-by-step reasoning
        prompt = self._create_cot_prompt(query, context)
        
        try:
            # Generate the reasoning
            messages = [
                {"role": "system", "content": self._get_system_message()},
                {"role": "user", "content": prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            reasoning_text = response.choices[0].message.content
            
            # Parse the reasoning and conclusion
            parsed_result = self._parse_reasoning(reasoning_text)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error during CoT reasoning: {str(e)}")
            return {
                "reasoning": f"Error occurred: {str(e)}",
                "conclusion": "Failed to generate a conclusion due to an error."
            }
    
    def _create_cot_prompt(self, query: str, context: Dict) -> str:
        """
        Create a prompt that encourages Chain of Thought reasoning.
        
        Args:
            query: The question or problem
            context: Additional context
            
        Returns:
            A formatted prompt string
        """
        # Format context if provided
        context_str = ""
        if context:
            context_str = "Context Information:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
            context_str += "\n"
        
        # Create the main prompt with step-by-step thinking instructions
        prompt = f"""
        {context_str}
        Question/Problem: {query}
        
        Please think through this step-by-step:
        
        1. First, let's understand what is being asked.
        2. Break down the problem into smaller parts.
        3. Address each part systematically.
        4. Consider any relevant assumptions or constraints.
        5. Identify and analyze possible approaches.
        
        Walk through your reasoning process explicitly, showing each step of your thinking.
        After working through the steps, provide your final conclusion.
        
        Use the following format:
        
        Step 1: [Your reasoning for step 1]
        Step 2: [Your reasoning for step 2]
        ...
        
        Conclusion: [Your final answer or conclusion]
        """
        
        return prompt
    
    def _get_system_message(self) -> str:
        """Get the system message that guides CoT reasoning."""
        return """You are an expert problem-solving assistant that uses Chain of Thought reasoning.
        
        Break down problems into clear, logical steps. Think carefully about each step and explain
        your reasoning thoroughly. Maintain a structured approach, considering different angles
        and possibilities. Focus on being methodical and precise in your reasoning process.
        
        Show your work clearly, labeling each step of your thinking process, and provide a
        well-reasoned conclusion at the end.
        """
    
    def _parse_reasoning(self, reasoning_text: str) -> Dict:
        """
        Parse the reasoning text to extract steps and conclusion.
        
        Args:
            reasoning_text: The raw reasoning text from the LLM
            
        Returns:
            Dictionary with reasoning steps and conclusion
        """
        import re
        
        # Extract reasoning steps
        steps = []
        step_pattern = r'Step\s+(\d+):\s+(.*?)(?=Step\s+\d+:|Conclusion:|$)'
        step_matches = re.finditer(step_pattern, reasoning_text, re.DOTALL)
        
        for match in step_matches:
            step_num = match.group(1)
            step_content = match.group(2).strip()
            steps.append(f"Step {step_num}: {step_content}")
        
        # If no steps were found with the pattern, split by newlines
        if not steps:
            lines = reasoning_text.split('\n')
            for line in lines:
                if line.strip().startswith("Step") or line.strip().startswith("STEP"):
                    steps.append(line.strip())
        
        # Extract conclusion
        conclusion = ""
        conclusion_pattern = r'Conclusion:\s+(.*?)(?=$)'
        conclusion_match = re.search(conclusion_pattern, reasoning_text, re.DOTALL)
        
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        else:
            # If no explicit conclusion section, use the last paragraph
            paragraphs = reasoning_text.split('\n\n')
            if paragraphs:
                conclusion = paragraphs[-1].strip()
        
        return {
            "reasoning": "\n\n".join(steps),
            "conclusion": conclusion
        }
