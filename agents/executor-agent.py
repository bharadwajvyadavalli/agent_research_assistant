"""
Executor Agent implementation.

This agent is responsible for carrying out specific tasks by interacting
with external tools and APIs through the Model Context Protocol (MCP).
"""

import logging
import json
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from utils.prompt_templates import EXECUTOR_SYSTEM_MESSAGE, EXECUTOR_TASK_TEMPLATE

logger = logging.getLogger(__name__)

class ExecutorAgent(BaseAgent):
    """
    Executor Agent responsible for executing plan steps using appropriate tools.
    
    The Executor translates abstract plan steps into concrete actions by interacting
    with external tools and APIs through the Model Context Protocol (MCP). It serves
    as the operational core of the system, carrying out tasks like searching for
    information, reading PDFs, and performing calculations.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Executor Agent with tool access capabilities."""
        super().__init__(name="Executor Agent", **kwargs)
        
        # Available tools cache
        self.available_tools = {}
        
        # Tool execution history
        self.tool_execution_history = []
        
        logger.info(f"Initialized Executor Agent (ID: {self.agent_id})")
    
    async def process(self, step: Dict) -> Dict:
        """
        Execute a specific step from the plan.
        
        Args:
            step: Dictionary containing details about the step to execute
                - step_id: Unique identifier for this step
                - description: Description of what needs to be done
                - tools_needed: List of tools that might be needed
                - context: Additional context for execution
                
        Returns:
            A dictionary containing the execution results:
                - result: The output of the execution
                - tool_executions: Records of tools that were used
                - status: Success or failure indication
                - error: Error message if applicable
        """
        step_id = step.get("step_id", "unknown_step")
        description = step.get("description", "")
        tools_needed = step.get("tools_needed", [])
        context = step.get("context", {})
        
        logger.info(f"Executing step {step_id}: {description[:50]}...")
        
        # First, determine what tools are actually needed
        if not tools_needed:
            # If no tools were specified in the plan, determine them dynamically
            tools_needed = await self._determine_required_tools(description, context)
        
        # Ensure all needed tools are available
        available_tools = await self._get_available_tools()
        missing_tools = [tool for tool in tools_needed if tool not in available_tools]
        
        if missing_tools:
            logger.warning(f"Missing required tools: {missing_tools}")
            # We'll proceed anyway and let the agent figure out alternatives
        
        # Generate a task-specific prompt
        prompt = EXECUTOR_TASK_TEMPLATE.format(
            step_description=description,
            available_tools=self._format_available_tools(available_tools),
            context=self._format_context(context)
        )
        
        # Generate execution plan with reasoning
        execution_plan = await self.generate_response(
            prompt=prompt,
            system_message=EXECUTOR_SYSTEM_MESSAGE
        )
        
        # Execute the tools based on the execution plan
        tool_results = await self._execute_tools_from_plan(execution_plan)
        
        # Synthesize the results
        synthesis_prompt = f"""
        I've executed the following step: {description}
        
        The tools returned these results:
        {json.dumps(tool_results, indent=2)}
        
        Please synthesize these results into a coherent response that accomplishes the step.
        Focus on extracting the key information and presenting it clearly.
        """
        
        result = await self.generate_response(
            prompt=synthesis_prompt,
            system_message=EXECUTOR_SYSTEM_MESSAGE
        )
        
        # Record the execution in history
        execution_record = {
            "step_id": step_id,
            "description": description,
            "tools_used": tool_results,
            "result": result,
            "status": "success" if not any(tr.get("error") for tr in tool_results) else "partial_success"
        }
        
        self.tool_execution_history.append(execution_record)
        
        # Store in memory if available
        if self.memory:
            await self.memory.store_execution(execution_record)
        
        return {
            "step_id": step_id,
            "result": result,
            "tool_executions": tool_results,
            "status": execution_record["status"],
            "error": None
        }
    
    async def _determine_required_tools(self, description: str, context: Dict) -> List[str]:
        """
        Determine which tools are needed for a specific step.
        
        Args:
            description: Description of the step
            context: Additional context
            
        Returns:
            List of tool names that might be useful
        """
        available_tools = await self._get_available_tools()
        
        tool_selection_prompt = f"""
        For the following task, which tools would be most appropriate to use?
        
        TASK: {description}
        
        AVAILABLE TOOLS:
        {self._format_available_tools(available_tools)}
        
        Please list only the names of the tools that would be most useful for this task.
        """
        
        tool_selection_response = await self.generate_response(
            prompt=tool_selection_prompt,
            system_message="You are a helpful assistant that helps select the most appropriate tools for tasks."
        )
        
        # Extract tool names from the response
        import re
        
        tool_names = []
        # Look for tool names in the response
        for tool_name in available_tools:
            if re.search(r'\b' + re.escape(tool_name) + r'\b', tool_selection_response, re.IGNORECASE):
                tool_names.append(tool_name)
        
        return tool_names
    
    async def _get_available_tools(self) -> Dict:
        """
        Get the list of available tools from the MCP client.
        
        Returns:
            Dictionary of available tools with descriptions
        """
        if not self.mcp_client:
            logger.warning("MCP client not initialized, no tools available")
            return {}
        
        if not self.available_tools:
            try:
                # Call MCP client to list tools
                tools_list = await self.mcp_client.list_tools()
                
                # Format into a more usable structure
                for tool in tools_list:
                    self.available_tools[tool["name"]] = {
                        "description": tool.get("description", "No description available"),
                        "parameters": tool.get("parameters", {})
                    }
                    
            except Exception as e:
                logger.error(f"Error retrieving available tools: {str(e)}")
                return {}
        
        return self.available_tools
    
    def _format_available_tools(self, tools: Dict) -> str:
        """Format available tools into a string for prompting."""
        formatted = "Available Tools:\n"
        
        for name, info in tools.items():
            formatted += f"- {name}: {info['description']}\n"
            if info.get("parameters"):
                formatted += f"  Parameters: {json.dumps(info['parameters'], indent=2)}\n"
        
        return formatted
    
    def _format_context(self, context: Dict) -> str:
        """Format context dictionary into a string for prompting."""
        context_str = "Context:\n"
        for key, value in context.items():
            context_str += f"- {key}: {value}\n"
        return context_str
    
    async def _execute_tools_from_plan(self, execution_plan: str) -> List[Dict]:
        """
        Execute tools based on the execution plan.
        
        Args:
            execution_plan: The execution plan text from the LLM
            
        Returns:
            List of tool execution results
        """
        import re
        
        tool_results = []
        
        # Extract tool calls using regex
        tool_pattern = r"TOOL:\s+(\w+)\s+PARAMETERS:\s+(\{.*?\})"
        tool_matches = re.finditer(tool_pattern, execution_plan, re.DOTALL)
        
        for match in tool_matches:
            tool_name = match.group(1)
            parameters_str = match.group(2)
            
            try:
                # Parse parameters
                parameters = json.loads(parameters_str)
                
                # Execute the tool
                if not self.mcp_client:
                    result = {"error": "MCP client not initialized"}
                else:
                    result = await self.mcp_client.call_tool(tool_name, parameters)
                
                tool_results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "error": None
                })
                
            except json.JSONDecodeError:
                logger.error(f"Invalid parameters format for tool {tool_name}: {parameters_str}")
                tool_results.append({
                    "tool": tool_name,
                    "parameters": parameters_str,
                    "result": None,
                    "error": "Invalid parameters format"
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                tool_results.append({
                    "tool": tool_name,
                    "parameters": parameters_str if 'parameters_str' in locals() else {},
                    "result": None,
                    "error": str(e)
                })
        
        # If no tool calls were found, try a simpler pattern
        if not tool_results:
            simple_tool_pattern = r"Use\s+(?:the\s+)?(\w+)(?:\s+tool)?\s+with\s+(?:parameters\s+)?(\{.*?\})"
            simple_matches = re.finditer(simple_tool_pattern, execution_plan, re.DOTALL | re.IGNORECASE)
            
            for match in simple_matches:
                tool_name = match.group(1)
                parameters_str = match.group(2)
                
                try:
                    parameters = json.loads(parameters_str)
                    
                    if not self.mcp_client:
                        result = {"error": "MCP client not initialized"}
                    else:
                        result = await self.mcp_client.call_tool(tool_name, parameters)
                    
                    tool_results.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "error": None
                    })
                    
                except Exception as e:
                    logger.error(f"Error with simple pattern tool execution {tool_name}: {str(e)}")
                    tool_results.append({
                        "tool": tool_name,
                        "parameters": parameters_str,
                        "result": None,
                        "error": str(e)
                    })
        
        # If still no tool calls found, try to infer tool calls from context
        if not tool_results and self.mcp_client:
            logger.info("No explicit tool calls found, inferring from execution plan")
            
            # Get available tools for reference
            available_tools = await self._get_available_tools()
            
            # For each available tool, check if it's mentioned in the execution plan
            for tool_name in available_tools:
                if re.search(r'\b' + re.escape(tool_name) + r'\b', execution_plan, re.IGNORECASE):
                    # Found a mention of this tool, try to infer parameters
                    infer_prompt = f"""
                    Based on this execution plan:
                    
                    {execution_plan}
                    
                    What parameters should I use for the {tool_name} tool?
                    Please respond with a valid JSON object containing the parameters.
                    """
                    
                    parameters_response = await self.generate_response(
                        prompt=infer_prompt,
                        system_message="You are a helpful assistant that extracts tool parameters from text."
                    )
                    
                    # Try to extract JSON from the response
                    json_pattern = r'```json\s*(.*?)\s*```|(\{.*\})'
                    json_match = re.search(json_pattern, parameters_response, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1) or json_match.group(2)
                        try:
                            parameters = json.loads(json_str)
                            
                            result = await self.mcp_client.call_tool(tool_name, parameters)
                            
                            tool_results.append({
                                "tool": tool_name,
                                "parameters": parameters,
                                "result": result,
                                "error": None,
                                "note": "Parameters inferred from context"
                            })
                            
                        except Exception as e:
                            logger.error(f"Error with inferred tool execution {tool_name}: {str(e)}")
                            tool_results.append({
                                "tool": tool_name,
                                "parameters": json_str,
                                "result": None,
                                "error": str(e),
                                "note": "Parameters inferred from context"
                            })
        
        return tool_results
