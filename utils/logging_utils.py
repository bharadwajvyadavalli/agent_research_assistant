"""
Logging utilities for the agent research assistant.

This module provides standardized logging functionality across the project.
Features include:
- Configurable log levels
- Colorized console output
- File logging with rotation
- Structured logging for agent activities
- Context-aware logging for tracking agent reasoning chains
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, Optional, Union, List


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors to logs in console output."""
    
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'       # Reset to default
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message


class AgentContextFilter(logging.Filter):
    """Filter that adds agent context information to log records."""
    
    def __init__(self, agent_id=None, session_id=None):
        super().__init__()
        self.agent_id = agent_id
        self.session_id = session_id
    
    def filter(self, record):
        record.agent_id = self.agent_id
        record.session_id = self.session_id
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON strings."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
        }
        
        # Add agent context if available
        if hasattr(record, 'agent_id') and record.agent_id:
            log_data['agent_id'] = record.agent_id
        if hasattr(record, 'session_id') and record.session_id:
            log_data['session_id'] = record.session_id
            
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 
                          'filename', 'funcName', 'id', 'levelname', 'levelno',
                          'lineno', 'module', 'msecs', 'message', 'msg', 'name', 
                          'pathname', 'process', 'processName', 'relativeCreated', 
                          'stack_info', 'thread', 'threadName']:
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_data[key] = value
                else:
                    try:
                        log_data[key] = str(value)
                    except:
                        log_data[key] = "UNPRINTABLE"
        
        return json.dumps(log_data)


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    json_output: bool = False,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    console_output: bool = True,
    colored_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, no file logging)
        max_file_size: Maximum size of log file before rotation in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        format_string: Format string for the log messages
        json_output: Whether to output logs as JSON (default: False)
        agent_id: ID of the agent (for context filtering)
        session_id: ID of the session (for context filtering)
        console_output: Whether to output logs to console (default: True)
        colored_console: Whether to colorize console output (default: True)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add context filter if agent_id or session_id is provided
    if agent_id or session_id:
        context_filter = AgentContextFilter(agent_id, session_id)
        logger.addFilter(context_filter)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if json_output:
            console_formatter = JSONFormatter()
        elif colored_console:
            console_formatter = ColorFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)
            
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        
        if json_output:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(format_string)
            
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class AgentLogger:
    """
    Logger class specifically designed for agents with additional context.
    Provides methods for logging agent activities, thoughts, and communication.
    """
    
    def __init__(
        self, 
        agent_id: str,
        session_id: Optional[str] = None,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        json_output: bool = False
    ):
        self.agent_id = agent_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log filename
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{agent_id}_{self.session_id}.log") if file_output else None
        
        # Set up logger
        self.logger = setup_logger(
            name=f"agent.{agent_id}",
            log_level=log_level,
            log_file=log_file,
            json_output=json_output,
            agent_id=agent_id,
            session_id=self.session_id,
            console_output=console_output
        )
    
    def debug(self, message: str, **kwargs):
        """Log a debug message with additional context."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message with additional context."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message with additional context."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message with additional context."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message with additional context."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal method to log messages with extra context."""
        extra = {**kwargs}
        self.logger.log(level, message, extra=extra)
    
    # Specialized logging methods for agent operations
    
    def log_thought(self, thought: str, thought_type: str = "chain_of_thought", **kwargs):
        """Log an agent's thought process."""
        self.debug(f"THOUGHT ({thought_type}): {thought}", 
                   thought_type=thought_type, **kwargs)
    
    def log_action(self, action: str, action_data: Any = None, **kwargs):
        """Log an action taken by the agent."""
        action_str = action
        if action_data:
            try:
                if isinstance(action_data, dict):
                    action_str += f" - {json.dumps(action_data)}"
                else:
                    action_str += f" - {str(action_data)}"
            except:
                action_str += " - [Unprintable action data]"
                
        self.info(f"ACTION: {action_str}", action=action, **kwargs)
    
    def log_observation(self, observation: Any, source: Optional[str] = None, **kwargs):
        """Log an observation made by the agent."""
        if isinstance(observation, dict):
            try:
                obs_str = json.dumps(observation)
            except:
                obs_str = str(observation)
        else:
            obs_str = str(observation)
            
        source_str = f" from {source}" if source else ""
        self.info(f"OBSERVATION{source_str}: {obs_str}", 
                  observation=obs_str, source=source, **kwargs)
    
    def log_communication(self, message: str, target: Optional[str] = None, 
                         direction: str = "outgoing", **kwargs):
        """Log communication between agents."""
        target_str = f" to {target}" if target else ""
        self.info(f"{direction.upper()} MESSAGE{target_str}: {message}",
                 comm_direction=direction, comm_target=target, **kwargs)
    
    def log_memory_access(self, memory_type: str, operation: str, 
                         key: Optional[str] = None, success: bool = True, **kwargs):
        """Log memory operations."""
        key_str = f" - {key}" if key else ""
        status = "SUCCESS" if success else "FAILED"
        self.debug(f"MEMORY {operation.upper()} {status} ({memory_type}){key_str}",
                  memory_type=memory_type, memory_operation=operation,
                  memory_key=key, operation_success=success, **kwargs)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log an error with additional context information."""
        self.error(f"ERROR: {str(error)}", 
                  error_type=type(error).__name__,
                  error_context=str(context))


class OrchestrationLogger:
    """
    Logger for the orchestrator component that coordinates multiple agents.
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        json_output: bool = False
    ):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log filename
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"orchestrator_{self.session_id}.log") if file_output else None
        
        # Set up logger
        self.logger = setup_logger(
            name=f"orchestrator.{self.session_id}",
            log_level=log_level,
            log_file=log_file,
            json_output=json_output,
            agent_id="orchestrator",
            session_id=self.session_id,
            console_output=console_output
        )
        
        # Track agent loggers
        self.agent_loggers = {}
    
    def get_agent_logger(self, agent_id: str) -> AgentLogger:
        """Get or create an agent logger for the specified agent."""
        if agent_id not in self.agent_loggers:
            self.agent_loggers[agent_id] = AgentLogger(
                agent_id=agent_id,
                session_id=self.session_id,
                console_output=False  # Let orchestrator handle console output
            )
        return self.agent_loggers[agent_id]
    
    def log_task_start(self, task_id: str, task_description: str, 
                      assigned_agents: List[str]):
        """Log the start of a new task."""
        self.logger.info(f"TASK START: {task_id} - {task_description}",
                        extra={
                            "task_id": task_id,
                            "task_description": task_description,
                            "assigned_agents": assigned_agents
                        })
    
    def log_task_complete(self, task_id: str, success: bool, 
                         result: Optional[Any] = None):
        """Log the completion of a task."""
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(f"TASK COMPLETE: {task_id} - {status}",
                        extra={
                            "task_id": task_id,
                            "task_success": success,
                            "task_result": str(result) if result else None
                        })
    
    def log_agent_assignment(self, agent_id: str, role: str, task_id: str):
        """Log the assignment of an agent to a role in a task."""
        self.logger.info(f"AGENT ASSIGNMENT: {agent_id} as {role} for task {task_id}",
                        extra={
                            "agent_id": agent_id,
                            "agent_role": role,
                            "task_id": task_id
                        })
    
    def log_interaction(self, from_agent: str, to_agent: str, 
                       interaction_type: str, content: str):
        """Log an interaction between agents."""
        self.logger.info(f"INTERACTION: {from_agent} -> {to_agent} ({interaction_type})",
                        extra={
                            "from_agent": from_agent,
                            "to_agent": to_agent,
                            "interaction_type": interaction_type,
                            "content": content
                        })


# Global default logger
def get_default_logger():
    """Get the default application logger."""
    return setup_logger(
        name="agent_research_assistant",
        log_level=logging.INFO,
        log_file="logs/application.log"
    )


default_logger = get_default_logger()


# Convenience functions that use the default logger
def debug(message: str, **kwargs):
    """Log a debug message with the default logger."""
    default_logger.debug(message, extra=kwargs)

def info(message: str, **kwargs):
    """Log an info message with the default logger."""
    default_logger.info(message, extra=kwargs)

def warning(message: str, **kwargs):
    """Log a warning message with the default logger."""
    default_logger.warning(message, extra=kwargs)

def error(message: str, **kwargs):
    """Log an error message with the default logger."""
    default_logger.error(message, extra=kwargs)

def critical(message: str, **kwargs):
    """Log a critical message with the default logger."""
    default_logger.critical(message, extra=kwargs)
