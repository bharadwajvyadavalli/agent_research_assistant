"""
Orchestrator for managing the multi-agent system.

The Orchestrator coordinates interactions between specialized agents,
manages the overall workflow, and ensures continuous improvement of the system.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional

from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.critic_agent import CriticAgent
from memory.memory_store import MemoryStore
from mcp.client import MCPClient
from improvement.feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for coordinating the multi-agent system workflow.
    
    The Orchestrator manages interactions between specialized agents (Planner,
    Executor, Critic), handles the overall task workflow, and ensures the system
    continuously improves based on feedback.
    """
    
    def __init__(self,
                 planner: Optional[PlannerAgent] = None,
                 executor: Optional[ExecutorAgent] = None,
                 critic: Optional[CriticAgent] = None,
                 memory_store: Optional[MemoryStore] = None,
                 mcp_client: Optional[MCPClient] = None,
                 max_iterations: int = 3,
                 improvement_interval: int = 5):
        """
        Initialize the Orchestrator with agents and configuration.
        
        Args:
            planner: The Planner agent
            executor: The Executor agent
            critic: The Critic agent
            memory_store: Shared memory system
            mcp_client: MCP client for tool access
            max_iterations: Maximum number of refinement iterations
            improvement_interval: Number of tasks before triggering improvement
        """
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.memory_store = memory_store
        self.mcp_client = mcp_client
        
        self.max_iterations = max_iterations
        self.improvement_interval = improvement_interval
        
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.completed_tasks = 0
        
        # Performance tracking
        self.performance_metrics = {
            "tasks_completed": 0,
            "average_iterations": 0,
            "total_iterations": 0,
            "critic_scores": []
        }
        
        # Improvement mechanism
        self.feedback_collector = FeedbackCollector()
        
        logger.info(f"Orchestrator initialized with session ID: {self.session_id}")
    
    async def setup(self):
        """Set up the orchestrator and initialize components if needed."""
        # Initialize memory store if not provided
        if not self.memory_store:
            from memory.memory_store import MemoryStore
            self.memory_store = MemoryStore()
            await self.memory_store.initialize()
        
        # Initialize MCP client if not provided
        if not self.mcp_client:
            from mcp.client import MCPClient
            self.mcp_client = MCPClient()
            await self.mcp_client.initialize()
        
        # Initialize agents if not provided
        if not self.planner:
            self.planner = PlannerAgent(
                memory_store=self.memory_store,
                mcp_client=self.mcp_client
            )
        
        if not self.executor:
            self.executor = ExecutorAgent(
                memory_store=self.memory_store,
                mcp_client=self.mcp_client
            )
        
        if not self.critic:
            self.critic = CriticAgent(
                memory_store=self.memory_store,
                mcp_client=self.mcp_client
            )
        
        logger.info("Orchestrator setup complete")
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a user query through the agent system.
        
        This is the main entry point for processing a query, which coordinates
        the multi-agent workflow of planning, execution, and evaluation.
        
        Args:
            query: The user's query or request
            context: Optional additional context
            
        Returns:
            Dictionary containing the final response and execution details
        """
        query_id = f"query_{uuid.uuid4().hex[:8]}"
        context = context or {}
        
        logger.info(f"Processing query {query_id}: {query[:50]}...")
        
        # Store the query in memory
        if self.memory_store:
            await self.memory_store.store_query(query, query_id, context)
        
        # Determine query complexity for appropriate reasoning approach
        complexity = await self._assess_complexity(query)
        
        # 1. Planning Phase
        plan_request = {
            "query_text": query,
            "context": context,
            "complexity": complexity
        }
        
        plan = await self.planner.process(plan_request)
        
        # 2. Evaluation of the Plan
        plan_evaluation = await self.critic.process({
            "content": str(plan["steps"]),
            "original_request": query,
            "content_type": "plan",
            "context": context
        })
        
        # 3. Refinement Loop
        iterations = 0
        while (plan_evaluation["action_needed"] == "refine" and 
               iterations < self.max_iterations):
            
            logger.info(f"Refining plan (iteration {iterations+1})")
            
            # Refine the plan based on feedback
            plan = await self.planner.replan(plan, plan_evaluation)
            
            # Re-evaluate the refined plan
            plan_evaluation = await self.critic.process({
                "content": str(plan["steps"]),
                "original_request": query,
                "content_type": "plan",
                "context": context
            })
            
            iterations += 1
        
        if plan_evaluation["action_needed"] == "reject":
            logger.warning("Plan rejected after refinement attempts")
            return {
                "query_id": query_id,
                "status": "failed",
                "reason": "Plan rejected by critic",
                "feedback": plan_evaluation["feedback"]
            }
        
        # 4. Execution Phase
        results = []
        
        for step in plan["steps"]:
            step_context = {**context, "plan": plan}
            
            # Execute the step
            step_result = await self.executor.process(step)
            
            # Evaluate the execution result
            execution_evaluation = await self.critic.process({
                "content": step_result["result"],
                "original_request": query,
                "content_type": "execution_result",
                "context": {
                    "step_info": step["description"],
                    "tools_used": step_result.get("tool_executions", []),
                    **context
                }
            })
            
            # Retry execution if needed (and within limits)
            retry_count = 0
            while (execution_evaluation["action_needed"] == "retry" and 
                   retry_count < 2):
                
                logger.info(f"Retrying step {step['step_id']} (attempt {retry_count+1})")
                
                # Update step with feedback
                step["feedback"] = execution_evaluation["feedback"]
                
                # Retry execution
                step_result = await self.executor.process(step)
                
                # Re-evaluate
                execution_evaluation = await self.critic.process({
                    "content": step_result["result"],
                    "original_request": query,
                    "content_type": "execution_result",
                    "context": {
                        "step_info": step["description"],
                        "tools_used": step_result.get("tool_executions", []),
                        **context
                    }
                })
                
                retry_count += 1
            
            # Store results regardless of evaluation outcome
            results.append({
                "step": step,
                "result": step_result["result"],
                "evaluation": execution_evaluation,
                "tool_executions": step_result.get("tool_executions", [])
            })
            
            # Stop execution if a critical step fails
            if execution_evaluation["action_needed"] == "fail":
                logger.warning(f"Step {step['step_id']} failed, stopping execution")
                break
        
        # 5. Synthesize Final Answer
        final_answer = await self._synthesize_results(query, plan, results)
        
        # 6. Evaluate Final Answer
        final_evaluation = await self.critic.process({
            "content": final_answer,
            "original_request": query,
            "content_type": "final_answer",
            "context": {
                "plan": plan,
                "results": results,
                **context
            }
        })
        
        # Refine answer if needed
        if final_evaluation["action_needed"] == "refine":
            logger.info("Refining final answer based on feedback")
            refined_answer = await self._refine_answer(
                final_answer, 
                final_evaluation, 
                query, 
                results
            )
            
            # Re-evaluate
            final_evaluation = await self.critic.process({
                "content": refined_answer,
                "original_request": query,
                "content_type": "final_answer",
                "context": {
                    "plan": plan,
                    "results": results,
                    **context
                }
            })
            
            final_answer = refined_answer
        
        # 7. Update Metrics
        self._update_metrics(iterations, final_evaluation)
        
        # 8. Store Results in Memory
        if self.memory_store:
            await self.memory_store.store_results(
                query_id=query_id,
                query=query,
                plan=plan,
                results=results,
                final_answer=final_answer,
                evaluation=final_evaluation
            )
        
        # 9. Consider Self-Improvement
        self.completed_tasks += 1
        if self.completed_tasks % self.improvement_interval == 0:
            # Schedule improvement in the background
            asyncio.create_task(self._trigger_improvement())
        
        # 10. Return Final Response
        return {
            "query_id": query_id,
            "status": "success" if final_evaluation["action_needed"] == "accept" else "partial_success",
            "answer": final_answer,
            "confidence": final_evaluation["overall_score"],
            "execution_details": {
                "plan": plan,
                "results": results,
                "iterations": iterations,
                "evaluation": final_evaluation
            }
        }
    
    async def _assess_complexity(self, query: str) -> str:
        """
        Assess the complexity of a query to determine the appropriate reasoning approach.
        
        Args:
            query: The user query
            
        Returns:
            Complexity level: "simple", "medium", or "complex"
        """
        # For simplicity, use the query length as a basic heuristic
        if len(query) < 50:
            return "simple"
        elif len(query) < 200:
            return "medium"
        else:
            return "complex"
        
        # A more sophisticated implementation would analyze the query content
        # to identify multi-step reasoning, research needs, etc.
    
    async def _synthesize_results(self, query: str, plan: Dict, results: List[Dict]) -> str:
        """
        Synthesize execution results into a coherent final answer.
        
        Args:
            query: Original user query
            plan: The execution plan
            results: Results from each execution step
            
        Returns:
            Synthesized final answer
        """
        # Prepare a prompt for the planner to synthesize results
        synthesis_prompt = f"""
        Please synthesize the following results into a coherent final answer for the user's query.
        
        USER QUERY:
        {query}
        
        EXECUTION RESULTS:
        """
        
        for i, result_item in enumerate(results):
            synthesis_prompt += f"\nStep {i+1}: {result_item['step']['description']}\n"
            synthesis_prompt += f"Result: {result_item['result']}\n"
        
        synthesis_prompt += "\nSynthesize a comprehensive, well-structured answer that directly addresses the user's query."
        
        # Use the planner agent for synthesis (with a different system message)
        system_message = """You are a helpful assistant that synthesizes information into clear, comprehensive answers.
        Focus on directly addressing the user's query with the information provided.
        Be concise yet thorough, and organize the information logically."""
        
        final_answer = await self.planner.generate_response(
            prompt=synthesis_prompt,
            system_message=system_message
        )
        
        return final_answer
    
    async def _refine_answer(self, answer: str, evaluation: Dict, query: str, results: List[Dict]) -> str:
        """
        Refine the final answer based on critic feedback.
        
        Args:
            answer: The original answer
            evaluation: Evaluation from the critic
            query: Original user query
            results: Execution results
            
        Returns:
            Refined answer
        """
        # Create a refinement prompt
        refinement_prompt = f"""
        Please refine this answer based on the following feedback:
        
        ORIGINAL QUERY:
        {query}
        
        CURRENT ANSWER:
        {answer}
        
        FEEDBACK:
        {evaluation['feedback']}
        
        IMPROVEMENT SUGGESTIONS:
        """
        
        for suggestion in evaluation.get('suggestions', []):
            refinement_prompt += f"- {suggestion}\n"
        
        refinement_prompt += "\nPlease provide an improved answer that addresses the feedback."
        
        # Use the planner for refinement
        refined_answer = await self.planner.generate_response(
            prompt=refinement_prompt,
            system_message="You are a helpful assistant that refines answers based on feedback. Incorporate the suggestions to improve the answer while maintaining accuracy and comprehensiveness."
        )
        
        return refined_answer
    
    def _update_metrics(self, iterations: int, evaluation: Dict) -> None:
        """Update performance metrics after a task."""
        self.performance_metrics["tasks_completed"] += 1
        self.performance_metrics["total_iterations"] += iterations
        self.performance_metrics["average_iterations"] = (
            self.performance_metrics["total_iterations"] / 
            self.performance_metrics["tasks_completed"]
        )
        
        if "overall_score" in evaluation:
            self.performance_metrics["critic_scores"].append(evaluation["overall_score"])
    
    async def _trigger_improvement(self) -> None:
        """Trigger the self-improvement process."""
        logger.info("Triggering self-improvement process")
        
        # Collect feedback and improvement suggestions
        feedback = await self.feedback_collector.collect_feedback(
            memory_store=self.memory_store,
            performance_metrics=self.performance_metrics
        )
        
        # Implement improvements
        await self._implement_improvements(feedback)
        
        logger.info("Self-improvement process completed")
    
    async def _implement_improvements(self, feedback: Dict) -> None:
        """
        Implement improvements based on feedback.
        
        Args:
            feedback: Collected feedback and improvement suggestions
        """
        # This is where you could implement more sophisticated
        # improvement mechanisms like fine-tuning, prompt optimization,
        # or dynamic adjustment of agent parameters
        
        logger.info(f"Improvement feedback: {feedback['summary']}")
        
        # Example: Adjust planner's temperature based on feedback
        if "planner_suggestions" in feedback:
            planner_feedback = feedback["planner_suggestions"]
            
            # Example adjustment: If the plans are too verbose, reduce temperature
            if "verbose" in planner_feedback.lower():
                new_temp = max(0.1, self.planner.temperature - 0.1)
                self.planner.temperature = new_temp
                logger.info(f"Adjusted planner temperature to {new_temp}")
        
        # Similar adjustments could be made for other agents
