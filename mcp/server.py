"""
MCP Server implementation for exposing tools, resources, and prompts.

This module provides a base server implementation that can be extended
to create standalone tool servers that conform to the Model Context Protocol.
"""

import logging
import json
import sys
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Type

from mcp.protocol import (
    MCPMessage, MCPMessageType, MCPMethod, 
    MCPTool, MCPResource, MCPPrompt
)

logger = logging.getLogger(__name__)

T = TypeVar('T', MCPTool, MCPResource, MCPPrompt)

class MCPServer:
    """
    Base server for exposing tools, resources, and prompts via MCP.
    
    This class provides the core functionality for an MCP server, including
    request handling, message processing, and registration of capabilities.
    It can be extended to create specialized servers for different domains.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the MCP server.
        
        Args:
            name: Name of the server
            description: Description of the server's purpose
        """
        self.name = name
        self.description = description
        
        # Server capabilities
        self.capabilities = {
            "tools": True,
            "resources": True,
            "prompts": True
        }
        
        # Registered items
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        
        # Message handlers
        self.handlers = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.RESOURCES_LIST.value: self._handle_resources_list,
            MCPMethod.RESOURCES_GET.value: self._handle_resources_get,
            MCPMethod.PROMPTS_LIST.value: self._handle_prompts_list,
            MCPMethod.PROMPTS_GET.value: self._handle_prompts_get
        }
        
        # Server state
        self.initialized = False
        
        logger.info(f"MCP server '{name}' initialized")
    
    def register_tool(self, tool: MCPTool) -> None:
        """
        Register a tool with the server.
        
        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource) -> None:
        """
        Register a resource with the server.
        
        Args:
            resource: The resource to register
        """
        self.resources[resource.name] = resource
        logger.info(f"Registered resource: {resource.name}")
    
    def register_prompt(self, prompt: MCPPrompt) -> None:
        """
        Register a prompt template with the server.
        
        Args:
            prompt: The prompt to register
        """
        self.prompts[prompt.name] = prompt
        logger.info(f"Registered prompt: {prompt.name}")
    
    def register_items(self, items: List[T]) -> None:
        """
        Register multiple items with the server.
        
        Args:
            items: List of tools, resources, or prompts to register
        """
        for item in items:
            if isinstance(item, MCPTool):
                self.register_tool(item)
            elif isinstance(item, MCPResource):
                self.register_resource(item)
            elif isinstance(item, MCPPrompt):
                self.register_prompt(item)
            else:
                logger.warning(f"Unknown item type: {type(item)}")
    
    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """
        Handle initialization requests.
        
        Args:
            message: The initialization request message
            
        Returns:
            Response message with server capabilities
        """
        client_capabilities = message.params.get("capabilities", {})
        
        # Adjust server capabilities based on client requirements
        adjusted_capabilities = {
            key: value for key, value in self.capabilities.items()
            if client_capabilities.get(key, True)
        }
        
        self.initialized = True
        
        return MCPMessage.create_response(
            id=message.id,
            result={
                "name": self.name,
                "description": self.description,
                "capabilities": adjusted_capabilities,
                "version": "1.0.0"
            }
        )
    
    async def _handle_tools_list(self, message: MCPMessage) -> MCPMessage:
        """
        Handle requests to list available tools.
        
        Args:
            message: The tools list request message
            
        Returns:
            Response message with tool definitions
        """
        if not self.initialized:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Server not initialized"
            )
        
        tools = [tool.to_dict() for tool in self.tools.values()]
        
        return MCPMessage.create_response(
            id=message.id,
            result={"tools": tools}
        )
    
    async def _handle_tools_call(self, message: MCPMessage) -> MCPMessage:
        """
        Handle requests to call a tool.
        
        Args:
            message: The tool call request message
            
        Returns:
            Response message with tool execution result
        """
        if not self.initialized:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Server not initialized"
            )
        
        tool_name = message.params.get("name")
        parameters = message.params.get("parameters", {})
        
        if not tool_name:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Missing tool name"
            )
        
        tool = self.tools.get(tool_name)
        
        if not tool:
            return MCPMessage.create_error_response(
                id=message.id,
                code=404,
                message=f"Tool '{tool_name}' not found"
            )
        
        if not tool.function:
            return MCPMessage.create_error_response(
                id=message.id,
                code=501,
                message=f"Tool '{tool_name}' has no implementation"
            )
        
        try:
            # Execute the tool function
            if tool.is_async:
                result = await tool.function(**parameters)
            else:
                # Run synchronous function in a thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool.function(**parameters)
                )
            
            return MCPMessage.create_response(
                id=message.id,
                result=result
            )
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            
            return MCPMessage.create_error_response(
                id=message.id,
                code=500,
                message=f"Error executing tool: {str(e)}"
            )
    
    async def _handle_resources_list(self, message: MCPMessage) -> MCPMessage:
        """
        Handle requests to list available resources.
        
        Args:
            message: The resources list request message
            
        Returns:
            Response message with resource definitions
        """
        if not self.initialized:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Server not initialized"
            )
        
        resources = [resource.to_dict() for resource in self.resources.values()]
        
        return MCPMessage.create_response(
            id=message.id,
            result={"resources": resources}
        )
    
    async def _handle_resources_get(self, message: MCPMessage) -> MCPMessage:
        """
        Handle requests to get a resource.
        
        Args:
            message: The resource get request message
            
        Returns:
            Response message with resource content
        """
        if not self.initialized:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Server not initialized"
            )
        
        resource_name = message.params.get("name")
        parameters = message.params.get("parameters", {})
        
        if not resource_name:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Missing resource name"
            )
        
        resource = self.resources.get(resource_name)
        
        if not resource:
            return MCPMessage.create_error_response(
                id=message.id,
                code=404,
                message=f"Resource '{resource_name}' not found"
            )
        
        if not resource.provider:
            return MCPMessage.create_error_response(
                id=message.id,
                code=501,
                message=f"Resource '{resource_name}' has no provider"
            )
        
        try:
            # Call the resource provider
            result = await resource.provider(**parameters)
            
            return MCPMessage.create_response(
                id=message.id,
                result={
                    "content": result,
                    "content_type": resource.content_type
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting resource {resource_name}: {str(e)}")
            
            return MCPMessage.create_error_response(
                id=message.id,
                code=500,
                message=f"Error getting resource: {str(e)}"
            )
    
    async def _handle_prompts_list(self, message: MCPMessage) -> MCPMessage:
        """
        Handle requests to list available prompt templates.
        
        Args:
            message: The prompts list request message
            
        Returns:
            Response message with prompt definitions
        """
        if not self.initialized:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Server not initialized"
            )
        
        prompts = [prompt.to_dict() for prompt in self.prompts.values()]
        
        return MCPMessage.create_response(
            id=message.id,
            result={"prompts": prompts}
        )
    
    async def _handle_prompts_get(self, message: MCPMessage) -> MCPMessage:
        """
        Handle requests to get a filled prompt template.
        
        Args:
            message: The prompt get request message
            
        Returns:
            Response message with filled prompt text
        """
        if not self.initialized:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Server not initialized"
            )
        
        prompt_name = message.params.get("name")
        parameters = message.params.get("parameters", {})
        
        if not prompt_name:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Missing prompt name"
            )
        
        prompt = self.prompts.get(prompt_name)
        
        if not prompt:
            return MCPMessage.create_error_response(
                id=message.id,
                code=404,
                message=f"Prompt '{prompt_name}' not found"
            )
        
        try:
            # Fill the prompt template
            filled_prompt = prompt.fill(parameters)
            
            return MCPMessage.create_response(
                id=message.id,
                result={
                    "text": filled_prompt
                }
            )
            
        except Exception as e:
            logger.error(f"Error filling prompt {prompt_name}: {str(e)}")
            
            return MCPMessage.create_error_response(
                id=message.id,
                code=500,
                message=f"Error filling prompt: {str(e)}"
            )
    
    async def process_message(self, message: MCPMessage) -> MCPMessage:
        """
        Process an incoming message and generate a response.
        
        Args:
            message: The incoming message
            
        Returns:
            Response message
        """
        if message.message_type != MCPMessageType.REQUEST:
            return MCPMessage.create_error_response(
                id=message.id,
                code=400,
                message="Only request messages are supported"
            )
        
        handler = self.handlers.get(message.method)
        
        if not handler:
            return MCPMessage.create_error_response(
                id=message.id,
                code=404,
                message=f"Unknown method: {message.method}"
            )
        
        try:
            return await handler(message)
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            
            return MCPMessage.create_error_response(
                id=message.id,
                code=500,
                message=f"Internal server error: {str(e)}"
            )
    
    async def run_stdio_server(self):
        """Run the server using stdio for communication."""
        logger.info(f"Starting MCP server '{self.name}' using stdio transport")
        
        while True:
            try:
                # Read a line from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    # End of input, exit
                    break
                
                # Parse the message
                try:
                    message = MCPMessage.from_json(line.strip())
                except Exception as e:
                    logger.error(f"Error parsing message: {str(e)}")
                    
                    response = MCPMessage.create_error_response(
                        id="unknown",
                        code=400,
                        message=f"Invalid message format: {str(e)}"
                    )
                    
                    print(response.to_json(), flush=True)
                    continue
                
                # Process the message
                response = await self.process_message(message)
                
                # Send the response
                print(response.to_json(), flush=True)
                
            except Exception as e:
                logger.error(f"Error in message loop: {str(e)}")
                
                # Try to send an error response
                try:
                    error_response = MCPMessage.create_error_response(
                        id="unknown",
                        code=500,
                        message=f"Internal server error: {str(e)}"
                    )
                    
                    print(error_response.to_json(), flush=True)
                except:
                    # Last resort: print to stderr
                    print(f"FATAL ERROR: {str(e)}", file=sys.stderr, flush=True)
        
        logger.info(f"MCP server '{self.name}' stopped")

class StandaloneToolServer(MCPServer):
    """
    Standalone tool server for convenient implementation.
    
    This class extends MCPServer with a simple entry point for running
    a standalone tool server, making it easy to create and deploy MCP-compatible
    tool servers.
    """
    
    @classmethod
    def create_and_run(cls, name: str, description: str, tools: List[MCPTool] = None,
                   resources: List[MCPResource] = None, prompts: List[MCPPrompt] = None):
        """
        Create and run a standalone tool server.
        
        Args:
            name: Name of the server
            description: Description of the server's purpose
            tools: List of tools to register
            resources: List of resources to register
            prompts: List of prompt templates to register
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)]
        )
        
        # Create server instance
        server = cls(name, description)
        
        # Register items
        if tools:
            for tool in tools:
                server.register_tool(tool)
        
        if resources:
            for resource in resources:
                server.register_resource(resource)
        
        if prompts:
            for prompt in prompts:
                server.register_prompt(prompt)
        
        # Run the server
        try:
            asyncio.run(server.run_stdio_server())
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
