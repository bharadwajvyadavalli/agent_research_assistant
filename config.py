"""
Configuration settings for the Agent Research Assistant application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv("OPENAI_API_KEY")  # or ANTHROPIC_API_KEY
API_BASE = os.getenv("API_BASE", "https://api.openai.com/v1")
API_MODEL = os.getenv("API_MODEL", "gpt-4")

# Agent configuration
AGENT_CONFIGS = {
    "planner": {
        "name": "Planner Agent",
        "description": "Strategic agent responsible for decomposing complex tasks into manageable steps",
        "temperature": 0.2,
        "max_tokens": 1000
    },
    "executor": {
        "name": "Executor Agent",
        "description": "Operational agent responsible for executing specific tasks using available tools",
        "temperature": 0.3,
        "max_tokens": 1000
    },
    "critic": {
        "name": "Critic Agent",
        "description": "Evaluative agent responsible for verifying outputs and providing feedback",
        "temperature": 0.1,
        "max_tokens": 1000
    }
}

# Memory configuration
MEMORY_CONFIG = {
    "vector_db_path": "data/vector_store",
    "max_episodic_memory_items": 100,
    "semantic_memory_refresh_interval": 24  # hours
}

# MCP configuration
MCP_CONFIG = {
    "server_host": "localhost",
    "server_port": 8000,
    "transport": "stdio",  # "stdio" or "http"
    "tools_directory": "mcp/tools"
}

# Reasoning configuration
REASONING_CONFIG = {
    "default_reasoning": "chain_of_thought",  # "chain_of_thought" or "tree_of_thoughts"
    "max_reasoning_steps": 5,
    "cot_prompt_template": "Let's think about {query} step by step."
}

# Improvement configuration
IMPROVEMENT_CONFIG = {
    "feedback_collection_enabled": True,
    "self_improvement_interval": 10,  # Number of interactions before attempting self-improvement
    "improvement_strategies": ["critic_feedback", "user_feedback", "performance_metrics"]
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": "logs/agent_assistant.log",
    "console_output": True
}
