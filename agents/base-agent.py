"""
Base agent class that defines common functionality for all agent types.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from memory.memory_store import MemoryStore
from mcp.client import MCPClient
from reasoning.chain_of_thought import ChainOfThoughtReasoner

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agent types.
    
    Defines common functionality for interacting with LLMs, storing memory,
    and leveraging reasoning techniques.
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 name: str = "Base Agent",
                 description: str = "Generic AI Agent",
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 max_tokens: int = 1000,
                 memory_store: Optional[MemoryStore] = None,
                 mcp_client: Optional[MCPClient] = None,
                 reasoner: Optional[Any] = None):
        """
        Initialize the base agent with common properties.
        
        Args:
            agent_id: Unique identifier for the agent, generated if not provided
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            model: The LLM model to use for this agent
            temperature: Sampling temperature for the model
            max_tokens: Maximum tokens to generate in responses
            memory_store: Memory system for the agent
            mcp_client: MCP client for tool access
            reasoner: Reasoning module for the agent
        """
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Components
        self.memory = memory_store
        self.mcp_client = mcp_client
        self.reasoner = reasoner or ChainOfThoughtReasoner()
        
        # State
        self.conversation_history = []
        self.current_task = None
        self.performance_metrics = {
            "successes": 0,
            "failures": 0,
            "feedback_scores": []
        }
        
        logger.info(f"Initialized {self.name} (ID: {self.agent_id})")
    
    @retry(
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_response(self, 
                               prompt: str, 
                               system_message: Optional[str] = None,
                               temperature: Optional[float] = None) -> str:
        """
        Generate a response from the LLM with retry logic for API errors.
        
        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message to set context
            temperature: Optional temperature override
            
        Returns:
            The generated text response
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add conversation history if available
            for msg in self.conversation_history[-5:]:  # Last 5 messages for context
                messages.append(msg)
                
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Store in episodic memory if available
            if self.memory:
                await self.memory.store_interaction(prompt, response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def reason(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Apply reasoning techniques to a query.
        
        Args:
            query: The question or problem to reason about
            context: Additional context information
            
        Returns:
            A dictionary containing reasoning steps and conclusion
        """
        return await self.reasoner.reason(query, context)
    
    async def use_tool(self, tool_name: str, parameters: Dict) -> Any:
        """
        Use an MCP tool through the client.
        
        Args:
            tool_name: Name of the tool to use
            parameters: Parameters to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        if not self.mcp_client:
            raise ValueError("MCP client not initialized")
        
        return await self.mcp_client.call_tool(tool_name, parameters)
    
    async def reflect(self) -> Dict:
        """
        Self-reflect on recent performance and adapt behavior.
        
        Returns:
            Insights from reflection process
        """
        # Implement base reflection logic
        return {"reflection": "Base reflection performed"}
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Process input according to the agent's specific role.
        
        This method must be implemented by all subclasses.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed result
        """
        pass
