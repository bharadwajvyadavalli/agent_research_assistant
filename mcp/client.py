"""
MCP Client implementation for connecting to Model Context Protocol servers.

This module provides the client interface for AI agents to discover and
interact with external tools, resources, and prompts through the MCP.
"""

import logging
import json
import asyncio
import uuid
import subprocess
import shlex
import os
from typing import Dict, List, Any, Optional, Union

import aiohttp

from mcp.protocol import MCPMessage, MCPMessageType, MCPMethod, MCPTool, MCPResource, MCPPrompt
from config import MCP_CONFIG

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for interacting with Model Context Protocol servers.
    
    The MCP client is responsible for communicating with MCP servers to
    discover and invoke tools, access resources, and retrieve prompts.
    It handles the intricacies of the protocol, allowing agents to focus
    on their core reasoning and decision-making.
    """
    
    def __init__(self, 
                 transport: str = None,
                 server_host: str = None,
                 server_port: int = None,
                 tools_directory: str = None):
        """
        Initialize the MCP client.
        
        Args:
            transport: Transport mechanism ('stdio' or 'http')
            server_host: Host address for HTTP transport
            server_port: Port number for HTTP transport
            tools_directory: Directory to look for local tool servers
        """
        self.transport = transport or MCP_CONFIG.get("transport", "stdio")
        self.server_host = server_host or MCP_CONFIG.get("server_host", "localhost")
        self.server_port = server_port or MCP_CONFIG.get("server_port", 8000)
        self.tools_directory = tools_directory or MCP_CONFIG.get("tools_directory", "mcp/tools")
        
        # Connection state
        self.initialized = False
        self.server_capabilities = {}
        self.server_processes = {}
        self.http_session = None
        
        # Cache for discovered resources
        self.available_tools = {}
        self.available_resources = {}
        self.available_prompts = {}
        
        logger.info(f"MCP client initialized with {self.transport} transport")
    
    async def initialize(self):
        """Initialize the client and establish connection with servers."""
        if self.transport == "http":
            self.http_session = aiohttp.ClientSession()
        
        # Start local tool servers if needed
        await self._start_local_servers()
        
        # Initialize connection with each server
        await self._initialize_servers()
        
        self.initialized = True
        logger.info("MCP client initialization complete")
    
    async def _start_local_servers(self):
        """Start local tool servers from the tools directory."""
        if not os.path.exists(self.tools_directory):
            logger.warning(f"Tools directory {self.tools_directory} does not exist")
            return
        
        for filename in os.listdir(self.tools_directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                server_name = filename[:-3]  # Remove .py extension
                server_path = os.path.join(self.tools_directory, filename)
                
                try:
                    if self.transport == "stdio":
                        # Start the server as a subprocess with stdio communication
                        process = subprocess.Popen(
                            ["python", server_path],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1  # Line buffered
                        )
                        
                        self.server_processes[server_name] = process
                        logger.info(f"Started local server {server_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to start local server {server_name}: {str(e)}")
    
    async def _initialize_servers(self):
        """Initialize connections with all servers."""
        # For stdio servers, send initialization messages
        for server_name, process in self.server_processes.items():
            try:
                init_message = MCPMessage.create_request(
                    method=MCPMethod.INITIALIZE.value,
                    params={"capabilities": {"tools": True, "resources": True, "prompts": True}}
                )
                
                # Send the message
                if self.transport == "stdio":
                    await self._send_stdio_message(process, init_message)
                    
                    # Read the response
                    response = await self._read_stdio_message(process)
                    
                    if response and response.message_type == MCPMessageType.RESPONSE:
                        if response.error:
                            logger.error(f"Server {server_name} initialization failed: {response.error}")
                        else:
                            self.server_capabilities[server_name] = response.result
                            logger.info(f"Server {server_name} initialized with capabilities: {response.result}")
                
            except Exception as e:
                logger.error(f"Error initializing server {server_name}: {str(e)}")
    
    async def list_tools(self) -> List[Dict]:
        """
        List all available tools from connected servers.
        
        Returns:
            List of tool definitions
        """
        if not self.initialized:
            await self.initialize()
        
        # Clear cache
        self.available_tools = {}
        
        tools = []
        
        # Query each server for tools
        for server_name, capabilities in self.server_capabilities.items():
            if capabilities.get("tools", False):
                process = self.server_processes.get(server_name)
                
                if process:
                    try:
                        list_message = MCPMessage.create_request(
                            method=MCPMethod.TOOLS_LIST.value
                        )
                        
                        # Send the message
                        if self.transport == "stdio":
                            await self._send_stdio_message(process, list_message)
                            
                            # Read the response
                            response = await self._read_stdio_message(process)
                            
                            if response and response.message_type == MCPMessageType.RESPONSE:
                                if response.error:
                                    logger.error(f"Error listing tools from {server_name}: {response.error}")
                                else:
                                    server_tools = response.result.get("tools", [])
                                    
                                    # Add server info to each tool
                                    for tool in server_tools:
                                        tool["server"] = server_name
                                        tools.append(tool)
                                        self.available_tools[tool["name"]] = tool
                        
                    except Exception as e:
                        logger.error(f"Error listing tools from {server_name}: {str(e)}")
        
        return tools
    
    async def call_tool(self, tool_name: str, parameters: Dict) -> Any:
        """
        Call a tool with the specified parameters.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        if not self.initialized:
            await self.initialize()
        
        # If tool cache is empty, populate it
        if not self.available_tools:
            await self.list_tools()
        
        # Find the tool
        tool_info = self.available_tools.get(tool_name)
        
        if not tool_info:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        server_name = tool_info.get("server")
        process = self.server_processes.get(server_name)
        
        if not process:
            raise ValueError(f"Server for tool '{tool_name}' not available")
        
        try:
            call_message = MCPMessage.create_request(
                method=MCPMethod.TOOLS_CALL.value,
                params={
                    "name": tool_name,
                    "parameters": parameters
                }
            )
            
            # Send the message
            if self.transport == "stdio":
                await self._send_stdio_message(process, call_message)
                
                # Read the response
                response = await self._read_stdio_message(process)
                
                if response and response.message_type == MCPMessageType.RESPONSE:
                    if response.error:
                        logger.error(f"Error calling tool {tool_name}: {response.error}")
                        return {"error": response.error}
                    else:
                        return response.result
            
            elif self.transport == "http":
                url = f"http://{self.server_host}:{self.server_port}/{server_name}/tools/call"
                
                async with self.http_session.post(url, json=call_message.to_dict()) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        response = MCPMessage.from_dict(response_data)
                        
                        if response.error:
                            logger.error(f"Error calling tool {tool_name}: {response.error}")
                            return {"error": response.error}
                        else:
                            return response.result
                    else:
                        error_text = await resp.text()
                        logger.error(f"HTTP error calling tool {tool_name}: {resp.status} - {error_text}")
                        return {"error": f"HTTP {resp.status}: {error_text}"}
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return {"error": str(e)}
        
        return {"error": "Unknown transport or communication error"}
    
    async def list_resources(self) -> List[Dict]:
        """
        List all available resources from connected servers.
        
        Returns:
            List of resource definitions
        """
        if not self.initialized:
            await self.initialize()
        
        # Clear cache
        self.available_resources = {}
        
        resources = []
        
        # Query each server for resources
        for server_name, capabilities in self.server_capabilities.items():
            if capabilities.get("resources", False):
                process = self.server_processes.get(server_name)
                
                if process:
                    try:
                        list_message = MCPMessage.create_request(
                            method=MCPMethod.RESOURCES_LIST.value
                        )
                        
                        # Send the message
                        if self.transport == "stdio":
                            await self._send_stdio_message(process, list_message)
                            
                            # Read the response
                            response = await self._read_stdio_message(process)
                            
                            if response and response.message_type == MCPMessageType.RESPONSE:
                                if response.error:
                                    logger.error(f"Error listing resources from {server_name}: {response.error}")
                                else:
                                    server_resources = response.result.get("resources", [])
                                    
                                    # Add server info to each resource
                                    for resource in server_resources:
                                        resource["server"] = server_name
                                        resources.append(resource)
                                        self.available_resources[resource["name"]] = resource
                        
                    except Exception as e:
                        logger.error(f"Error listing resources from {server_name}: {str(e)}")
        
        return resources
    
    async def get_resource(self, resource_name: str, params: Optional[Dict] = None) -> Any:
        """
        Get a resource with the specified parameters.
        
        Args:
            resource_name: Name of the resource to get
            params: Optional parameters to access the resource
            
        Returns:
            The resource content
        """
        if not self.initialized:
            await self.initialize()
        
        # If resource cache is empty, populate it
        if not self.available_resources:
            await self.list_resources()
        
        # Find the resource
        resource_info = self.available_resources.get(resource_name)
        
        if not resource_info:
            raise ValueError(f"Resource '{resource_name}' not found")
        
        server_name = resource_info.get("server")
        process = self.server_processes.get(server_name)
        
        if not process:
            raise ValueError(f"Server for resource '{resource_name}' not available")
        
        try:
            get_message = MCPMessage.create_request(
                method=MCPMethod.RESOURCES_GET.value,
                params={
                    "name": resource_name,
                    "parameters": params or {}
                }
            )
            
            # Send the message
            if self.transport == "stdio":
                await self._send_stdio_message(process, get_message)
                
                # Read the response
                response = await self._read_stdio_message(process)
                
                if response and response.message_type == MCPMessageType.RESPONSE:
                    if response.error:
                        logger.error(f"Error getting resource {resource_name}: {response.error}")
                        return {"error": response.error}
                    else:
                        return response.result
            
        except Exception as e:
            logger.error(f"Error getting resource {resource_name}: {str(e)}")
            return {"error": str(e)}
        
        return {"error": "Unknown transport or communication error"}
    
    async def list_prompts(self) -> List[Dict]:
        """
        List all available prompt templates from connected servers.
        
        Returns:
            List of prompt template definitions
        """
        if not self.initialized:
            await self.initialize()
        
        # Clear cache
        self.available_prompts = {}
        
        prompts = []
        
        # Query each server for prompts
        for server_name, capabilities in self.server_capabilities.items():
            if capabilities.get("prompts", False):
                process = self.server_processes.get(server_name)
                
                if process:
                    try:
                        list_message = MCPMessage.create_request(
                            method=MCPMethod.PROMPTS_LIST.value
                        )
                        
                        # Send the message
                        if self.transport == "stdio":
                            await self._send_stdio_message(process, list_message)
                            
                            # Read the response
                            response = await self._read_stdio_message(process)
                            
                            if response and response.message_type == MCPMessageType.RESPONSE:
                                if response.error:
                                    logger.error(f"Error listing prompts from {server_name}: {response.error}")
                                else:
                                    server_prompts = response.result.get("prompts", [])
                                    
                                    # Add server info to each prompt
                                    for prompt in server_prompts:
                                        prompt["server"] = server_name
                                        prompts.append(prompt)
                                        self.available_prompts[prompt["name"]] = prompt
                        
                    except Exception as e:
                        logger.error(f"Error listing prompts from {server_name}: {str(e)}")
        
        return prompts
    
    async def get_prompt(self, prompt_name: str, params: Optional[Dict] = None) -> str:
        """
        Get a filled prompt template.
        
        Args:
            prompt_name: Name of the prompt template
            params: Parameters to fill the template
            
        Returns:
            The filled prompt text
        """
        if not self.initialized:
            await self.initialize()
        
        # If prompt cache is empty, populate it
        if not self.available_prompts:
            await self.list_prompts()
        
        # Find the prompt
        prompt_info = self.available_prompts.get(prompt_name)
        
        if not prompt_info:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        server_name = prompt_info.get("server")
        process = self.server_processes.get(server_name)
        
        if not process:
            raise ValueError(f"Server for prompt '{prompt_name}' not available")
        
        try:
            get_message = MCPMessage.create_request(
                method=MCPMethod.PROMPTS_GET.value,
                params={
                    "name": prompt_name,
                    "parameters": params or {}
                }
            )
            
            # Send the message
            if self.transport == "stdio":
                await self._send_stdio_message(process, get_message)
                
                # Read the response
                response = await self._read_stdio_message(process)
                
                if response and response.message_type == MCPMessageType.RESPONSE:
                    if response.error:
                        logger.error(f"Error getting prompt {prompt_name}: {response.error}")
                        return {"error": response.error}
                    else:
                        return response.result.get("text", "")
            
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name}: {str(e)}")
            return {"error": str(e)}
        
        return {"error": "Unknown transport or communication error"}
    
    async def _send_stdio_message(self, process, message: MCPMessage):
        """Send a message to a stdio-based server."""
        try:
            message_json = message.to_json() + "\n"
            process.stdin.write(message_json)
            process.stdin.flush()
        except Exception as e:
            logger.error(f"Error sending stdio message: {str(e)}")
            raise
    
    async def _read_stdio_message(self, process) -> Optional[MCPMessage]:
        """Read a message from a stdio-based server."""
        try:
            # Use asyncio to read from stdout without blocking
            line = await asyncio.get_event_loop().run_in_executor(
                None, process.stdout.readline
            )
            
            if not line:
                return None
            
            return MCPMessage.from_json(line.strip())
        except Exception as e:
            logger.error(f"Error reading stdio message: {str(e)}")
            return None
    
    async def close(self):
        """Close all connections and clean up resources."""
        # Close HTTP session if it exists
        if self.http_session:
            await self.http_session.close()
        
        # Terminate all server processes
        for server_name, process in self.server_processes.items():
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception as e:
                logger.warning(f"Error terminating server {server_name}: {str(e)}")
                try:
                    process.kill()
                except:
                    pass
        
        self.server_processes.clear()
        self.initialized = False
        logger.info("MCP client closed")
