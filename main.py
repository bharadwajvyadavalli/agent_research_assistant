"""
Main application for the Agent Research Assistant.

This module serves as the entry point for the application, initializing
and coordinating all components of the agent system.
"""

import os
import sys
import logging
import asyncio
import argparse
import json
from typing import Dict, List, Any, Optional

from agents.orchestrator import Orchestrator
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.critic_agent import CriticAgent
from memory.memory_store import MemoryStore
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from mcp.client import MCPClient
from reasoning.chain_of_thought import ChainOfThoughtReasoner
from reasoning.tree_of_thoughts import TreeOfThoughtsReasoner
from utils.prompt_templates import SYSTEM_PROMPT
from utils.logging_utils import setup_logging
from config import API_KEY, API_MODEL

# Check for API key
if not API_KEY:
    print("Error: API key not found. Please set the OPENAI_API_KEY environment variable.")
    print("You can create a .env file with the following content:")
    print("OPENAI_API_KEY=your_api_key_here")
    sys.exit(1)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class AgentResearchAssistant:
    """
    Main application class for the Agent Research Assistant.
    
    This class initializes and coordinates all components of the agent
    system, providing a high-level interface for interacting with it.
    """
    
    def __init__(self):
        """Initialize the Agent Research Assistant."""
        self.orchestrator = None
        self.memory_store = None
        self.mcp_client = None
        
        # Initialize auxiliary memory systems
        self.episodic_memory = None
        self.semantic_memory = None
        
        # Initialize components
        self.initialized = False
        
        logger.info("Agent Research Assistant initialized")
    
    async def initialize(self):
        """Initialize all components of the system."""
        if self.initialized:
            return
        
        logger.info("Initializing Agent Research Assistant components...")
        
        # Initialize memory store
        self.memory_store = MemoryStore()
        await self.memory_store.initialize()
        
        # Initialize MCP client
        self.mcp_client = MCPClient()
        await self.mcp_client.initialize()
        
        # Initialize auxiliary memory systems
        self.episodic_memory = EpisodicMemory(self.memory_store)
        self.semantic_memory = SemanticMemory(self.memory_store)
        
        # Initialize specialized agents
        planner = PlannerAgent(
            memory_store=self.memory_store,
            mcp_client=self.mcp_client,
            reasoner=TreeOfThoughtsReasoner()
        )
        
        executor = ExecutorAgent(
            memory_store=self.memory_store,
            mcp_client=self.mcp_client
        )
        
        critic = CriticAgent(
            memory_store=self.memory_store,
            mcp_client=self.mcp_client
        )
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            planner=planner,
            executor=executor,
            critic=critic,
            memory_store=self.memory_store,
            mcp_client=self.mcp_client
        )
        
        await self.orchestrator.setup()
        
        self.initialized = True
        logger.info("Agent Research Assistant initialization complete")
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a user query through the agent system.
        
        Args:
            query: The user's query
            context: Optional additional context
            
        Returns:
            The result of processing the query
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"Processing query: {query[:50]}...")
        
        context = context or {}
        
        # Process the query through the orchestrator
        result = await self.orchestrator.process_query(query, context)
        
        return result
    
    async def process_pdf(self, pdf_path: str, query: Optional[str] = None) -> Dict:
        """
        Process a PDF document and answer a query about it.
        
        Args:
            pdf_path: Path to the PDF file
            query: Optional query about the PDF content
            
        Returns:
            The result of processing the PDF and query
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Call the PDF tool to extract metadata
        try:
            metadata = await self.mcp_client.call_tool("pdf_get_metadata", {"file_path": pdf_path})
            
            if not metadata.get("success", False):
                return {
                    "status": "error",
                    "reason": metadata.get("error", "Failed to extract PDF metadata"),
                    "pdf_path": pdf_path
                }
            
            # Create context with PDF information
            context = {
                "pdf_path": pdf_path,
                "pdf_metadata": metadata.get("metadata", {}),
                "num_pages": metadata.get("num_pages", 0)
            }
            
            # Extract text from the first 10 pages or all if less
            num_pages = min(10, metadata.get("num_pages", 0))
            pages_to_extract = list(range(1, num_pages + 1))
            
            text_result = await self.mcp_client.call_tool("pdf_extract_text", {
                "file_path": pdf_path,
                "page_numbers": pages_to_extract
            })
            
            if text_result.get("success", False):
                # Add text content to context
                context["pdf_content"] = text_result.get("text", {})
            
            # If a query is provided, process it with the PDF context
            if query:
                return await self.process_query(query, context)
            else:
                # Create a default query to summarize the PDF
                default_query = f"Please analyze this PDF document and provide a comprehensive summary of its content."
                return await self.process_query(default_query, context)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "status": "error",
                "reason": f"Error processing PDF: {str(e)}",
                "pdf_path": pdf_path
            }
    
    async def close(self):
        """Close all connections and clean up resources."""
        if self.mcp_client:
            await self.mcp_client.close()
        
        logger.info("Agent Research Assistant closed")

async def run_interactive_mode():
    """Run the assistant in interactive mode."""
    assistant = AgentResearchAssistant()
    await assistant.initialize()
    
    print("\nAgent Research Assistant - Interactive Mode")
    print("===========================================")
    print("Enter your query or use one of the following commands:")
    print("  !pdf <path>: Process a PDF document")
    print("  !exit: Exit the application")
    print()
    
    while True:
        try:
            user_input = input("> ")
            
            if user_input.lower() == "!exit":
                break
            
            elif user_input.lower().startswith("!pdf "):
                # Extract PDF path
                pdf_path = user_input[5:].strip()
                
                if not os.path.exists(pdf_path):
                    print(f"Error: PDF file not found: {pdf_path}")
                    continue
                
                # Ask for a query about the PDF
                query = input("Query about the PDF (or press Enter to summarize): ")
                
                result = await assistant.process_pdf(pdf_path, query if query else None)
                
                if result.get("status") == "error":
                    print(f"Error: {result.get('reason')}")
                else:
                    print("\nResult:")
                    print("-------")
                    print(result.get("answer", "No answer generated"))
                    print()
            
            elif user_input:
                # Process regular query
                result = await assistant.process_query(user_input)
                
                if result.get("status") == "error":
                    print(f"Error: {result.get('reason')}")
                else:
                    print("\nResult:")
                    print("-------")
                    print(result.get("answer", "No answer generated"))
                    print()
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
    await assistant.close()

async def run_api_mode(host: str = "localhost", port: int = 8000):
    """
    Run the assistant as an API server.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    from aiohttp import web
    
    assistant = AgentResearchAssistant()
    await assistant.initialize()
    
    async def handle_query(request):
        try:
            data = await request.json()
            query = data.get("query")
            context = data.get("context")
            
            if not query:
                return web.json_response({"error": "Missing query parameter"}, status=400)
            
            result = await assistant.process_query(query, context)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_pdf(request):
        try:
            data = await request.json()
            pdf_path = data.get("pdf_path")
            query = data.get("query")
            
            if not pdf_path:
                return web.json_response({"error": "Missing pdf_path parameter"}, status=400)
            
            if not os.path.exists(pdf_path):
                return web.json_response({"error": f"PDF file not found: {pdf_path}"}, status=404)
            
            result = await assistant.process_pdf(pdf_path, query)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    app = web.Application()
    app.router.add_post("/query", handle_query)
    app.router.add_post("/pdf", handle_pdf)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    logger.info(f"API server running at http://{host}:{port}")
    await site.start()
    
    try:
        # Keep the server running until interrupted
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
        await assistant.close()

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Agent Research Assistant")
    parser.add_argument("--api", action="store_true", help="Run in API mode")
    parser.add_argument("--host", default="localhost", help="Host address for API mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for API mode")
    parser.add_argument("--query", help="Process a single query and exit")
    parser.add_argument("--pdf", help="Process a PDF file")
    
    args = parser.parse_args()
    
    if args.query:
        # Process a single query and exit
        async def run_single_query():
            assistant = AgentResearchAssistant()
            await assistant.initialize()
            
            result = await assistant.process_query(args.query)
            print(json.dumps(result, indent=2))
            
            await assistant.close()
        
        asyncio.run(run_single_query())
        
    elif args.pdf:
        # Process a PDF file and exit
        async def run_pdf_processing():
            assistant = AgentResearchAssistant()
            await assistant.initialize()
            
            # Check if the PDF file exists
            if not os.path.exists(args.pdf):
                print(f"Error: PDF file not found: {args.pdf}")
                await assistant.close()
                return
            
            # Use query if provided, otherwise summarize
            result = await assistant.process_pdf(args.pdf, args.query)
            print(json.dumps(result, indent=2))
            
            await assistant.close()
        
        asyncio.run(run_pdf_processing())
        
    elif args.api:
        # Run in API mode
        asyncio.run(run_api_mode(args.host, args.port))
        
    else:
        # Run in interactive mode
        asyncio.run(run_interactive_mode())

if __name__ == "__main__":
    main()
