"""
Model Context Protocol (MCP) implementation.

This module provides the core protocol classes for standardizing how external
context and tools are provided to LLMs, facilitating interoperability and
abstraction of external capabilities.
"""

import logging
import json
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

class MCPMessageType(Enum):
    """Message types for MCP protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class MCPMethod(Enum):
    """Standard MCP methods."""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_GET = "resources/get"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"

class MCPMessage:
    """
    Represents a message in the Model Context Protocol.
    
    MCP uses JSON-RPC 2.0 format for communication between clients and servers.
    """
    
    def __init__(self, 
                 message_type: MCPMessageType,
                 method: Optional[str] = None,
                 params: Optional[Dict] = None,
                 id: Optional[str] = None,
                 result: Optional[Any] = None,
                 error: Optional[Dict] = None):
        """
        Initialize an MCP message.
        
        Args:
            message_type: Type of message (request, response, notification)
            method: Method name for requests/notifications
            params: Parameters for the method
            id: Message ID for requests/responses
            result: Result data for responses
            error: Error information for failed responses
        """
        self.message_type = message_type
        self.method = method
        self.params = params or {}
        self.id = id or str(uuid.uuid4())
        self.result = result
        self.error = error
    
    def to_dict(self) -> Dict:
        """Convert the message to a dictionary for serialization."""
        message = {
            "jsonrpc": "2.0"
        }
        
        if self.message_type == MCPMessageType.REQUEST:
            message["method"] = self.method
            message["params"] = self.params
            message["id"] = self.id
        elif self.message_type == MCPMessageType.RESPONSE:
            message["id"] = self.id
            if self.error:
                message["error"] = self.error
            else:
                message["result"] = self.result
        elif self.message_type == MCPMessageType.NOTIFICATION:
            message["method"] = self.method
            message["params"] = self.params
        
        return message
    
    def to_json(self) -> str:
        """Convert the message to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPMessage':
        """
        Create an MCP message from a dictionary.
        
        Args:
            data: Dictionary representing the message
            
        Returns:
            An MCPMessage instance
        """
        if "method" in data and "id" in data:
            return cls(
                message_type=MCPMessageType.REQUEST,
                method=data["method"],
                params=data.get("params", {}),
                id=data["id"]
            )
        elif "method" in data and "id" not in data:
            return cls(
                message_type=MCPMessageType.NOTIFICATION,
                method=data["method"],
                params=data.get("params", {})
            )
        elif "id" in data:
            return cls(
                message_type=MCPMessageType.RESPONSE,
                id=data["id"],
                result=data.get("result"),
                error=data.get("error")
            )
        else:
            raise ValueError("Invalid message format")
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """
        Create an MCP message from a JSON string.
        
        Args:
            json_str: JSON string representing the message
            
        Returns:
            An MCPMessage instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def create_request(cls, method: str, params: Dict = None, id: str = None) -> 'MCPMessage':
        """Create a request message."""
        return cls(
            message_type=MCPMessageType.REQUEST,
            method=method,
            params=params,
            id=id
        )
    
    @classmethod
    def create_response(cls, id: str, result: Any = None, error: Dict = None) -> 'MCPMessage':
        """Create a response message."""
        return cls(
            message_type=MCPMessageType.RESPONSE,
            id=id,
            result=result,
            error=error
        )
    
    @classmethod
    def create_notification(cls, method: str, params: Dict = None) -> 'MCPMessage':
        """Create a notification message."""
        return cls(
            message_type=MCPMessageType.NOTIFICATION,
            method=method,
            params=params
        )
    
    @classmethod
    def create_error_response(cls, id: str, code: int, message: str, data: Any = None) -> 'MCPMessage':
        """Create an error response message."""
        error = {
            "code": code,
            "message": message
        }
        
        if data:
            error["data"] = data
            
        return cls(
            message_type=MCPMessageType.RESPONSE,
            id=id,
            error=error
        )

class MCPTool:
    """
    Represents a tool definition in the Model Context Protocol.
    
    Tools are executable functions that the LLM can invoke to interact with
    external systems, APIs, or perform operations.
    """
    
    def __init__(self,
                 name: str,
                 description: str,
                 parameters: Dict,
                 function: Optional[Callable] = None,
                 version: str = "1.0.0",
                 is_async: bool = True):
        """
        Initialize a tool definition.
        
        Args:
            name: Unique name of the tool
            description: Human-readable description
            parameters: JSON Schema for input parameters
            function: The executable function for this tool
            version: Tool version
            is_async: Whether the tool function is asynchronous
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.version = version
        self.is_async = is_async
    
    def to_dict(self) -> Dict:
        """Convert the tool definition to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict, function: Optional[Callable] = None) -> 'MCPTool':
        """
        Create a tool from a dictionary definition.
        
        Args:
            data: Dictionary with tool properties
            function: Optional executable function
            
        Returns:
            An MCPTool instance
        """
        return cls(
            name=data["name"],
            description=data.get("description", "No description provided"),
            parameters=data.get("parameters", {}),
            function=function,
            version=data.get("version", "1.0.0"),
            is_async=data.get("is_async", True)
        )

class MCPResource:
    """
    Represents a resource definition in the Model Context Protocol.
    
    Resources are data or content exposed by the server that the LLM can access,
    such as file contents, database records, or configuration settings.
    """
    
    def __init__(self,
                 name: str,
                 description: str,
                 content_type: str,
                 access_params: Optional[Dict] = None,
                 provider: Optional[Callable] = None,
                 version: str = "1.0.0"):
        """
        Initialize a resource definition.
        
        Args:
            name: Unique name of the resource
            description: Human-readable description
            content_type: MIME type of the resource
            access_params: Parameters needed to access the resource
            provider: Function that provides the resource content
            version: Resource version
        """
        self.name = name
        self.description = description
        self.content_type = content_type
        self.access_params = access_params or {}
        self.provider = provider
        self.version = version
    
    def to_dict(self) -> Dict:
        """Convert the resource definition to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "content_type": self.content_type,
            "access_params": self.access_params,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict, provider: Optional[Callable] = None) -> 'MCPResource':
        """
        Create a resource from a dictionary definition.
        
        Args:
            data: Dictionary with resource properties
            provider: Optional provider function
            
        Returns:
            An MCPResource instance
        """
        return cls(
            name=data["name"],
            description=data.get("description", "No description provided"),
            content_type=data.get("content_type", "application/json"),
            access_params=data.get("access_params", {}),
            provider=provider,
            version=data.get("version", "1.0.0")
        )

class MCPPrompt:
    """
    Represents a prompt template definition in the Model Context Protocol.
    
    Prompts are reusable templates that can integrate tools and resources,
    potentially incorporating predefined interaction workflows.
    """
    
    def __init__(self,
                 name: str,
                 description: str,
                 template: str,
                 parameters: Dict,
                 version: str = "1.0.0"):
        """
        Initialize a prompt template.
        
        Args:
            name: Unique name of the prompt
            description: Human-readable description
            template: The prompt template string
            parameters: JSON Schema for template parameters
            version: Prompt version
        """
        self.name = name
        self.description = description
        self.template = template
        self.parameters = parameters
        self.version = version
    
    def to_dict(self) -> Dict:
        """Convert the prompt definition to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "parameters": self.parameters,
            "version": self.version
        }
    
    def fill(self, params: Dict) -> str:
        """
        Fill the template with the provided parameters.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            The filled prompt string
        """
        try:
            return self.template.format(**params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPPrompt':
        """
        Create a prompt from a dictionary definition.
        
        Args:
            data: Dictionary with prompt properties
            
        Returns:
            An MCPPrompt instance
        """
        return cls(
            name=data["name"],
            description=data.get("description", "No description provided"),
            template=data["template"],
            parameters=data.get("parameters", {}),
            version=data.get("version", "1.0.0")
        )
