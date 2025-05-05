"""
Episodic Memory implementation for storing and retrieving specific past events.

This module provides specialized functionality for managing episodic memories,
which are records of specific past events, interactions, and experiences.
"""

import logging
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)

class EpisodicMemory:
    """
    Specialized memory system for storing and retrieving episodic memories.
    
    Episodic memory stores records of specific past events, preserving their
    temporal context (what happened when). It is crucial for learning from
    particular past trials, enabling self-reflection on failed action sequences,
    and understanding user-specific history.
    """
    
    def __init__(self, memory_store: MemoryStore):
        """
        Initialize the episodic memory system.
        
        Args:
            memory_store: The main memory store instance
        """
        self.memory_store = memory_store
        logger.info("Episodic memory system initialized")
    
    async def store_event(self, event_type: str, event_data: Dict, context: Optional[Dict] = None) -> str:
        """
        Store an event in episodic memory.
        
        Args:
            event_type: Type of event (e.g., 'user_input', 'system_output', 'action')
            event_data: Data associated with the event
            context: Additional context information
            
        Returns:
            Memory ID of the stored event
        """
        # Create a standardized event structure
        event = {
            "input": f"EVENT: {event_type}",
            "output": json.dumps(event_data),
            "metadata": {
                "event_type": event_type,
                "context": context or {}
            }
        }
        
        # Store in the main memory store as an interaction
        memory_id = await self.memory_store.store_interaction(
            input_text=event["input"],
            output_text=event["output"],
            metadata=event["metadata"]
        )
        
        return memory_id
    
    async def store_conversation_turn(self, user_input: str, system_output: str, 
                                   context: Optional[Dict] = None) -> str:
        """
        Store a conversation turn in episodic memory.
        
        Args:
            user_input: User's input message
            system_output: System's response message
            context: Additional context information
            
        Returns:
            Memory ID of the stored conversation turn
        """
        # Store directly in the main memory store
        memory_id = await self.memory_store.store_interaction(
            input_text=user_input,
            output_text=system_output,
            metadata={"event_type": "conversation", "context": context or {}}
        )
        
        return memory_id
    
    async def store_action_sequence(self, goal: str, actions: List[Dict], 
                                 outcome: Dict, success: bool,
                                 context: Optional[Dict] = None) -> str:
        """
        Store an action sequence (plan execution) in episodic memory.
        
        Args:
            goal: The goal of the action sequence
            actions: List of actions taken
            outcome: The outcome of the action sequence
            success: Whether the sequence was successful
            context: Additional context information
            
        Returns:
            Memory ID of the stored action sequence
        """
        # Create a standardized action sequence structure
        action_sequence = {
            "goal": goal,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "context": context or {}
        }
        
        # Store as a special event type
        memory_id = await self.store_event(
            event_type="action_sequence",
            event_data=action_sequence
        )
        
        return memory_id
    
    async def store_reflection(self, topic: str, thoughts: str, 
                           insights: List[str], context: Optional[Dict] = None) -> str:
        """
        Store a reflection or self-evaluation in episodic memory.
        
        Args:
            topic: The topic of reflection
            thoughts: The reflection content
            insights: Key insights from the reflection
            context: Additional context information
            
        Returns:
            Memory ID of the stored reflection
        """
        # Create a standardized reflection structure
        reflection = {
            "topic": topic,
            "thoughts": thoughts,
            "insights": insights,
            "context": context or {}
        }
        
        # Store as a special event type
        memory_id = await self.store_event(
            event_type="reflection",
            event_data=reflection
        )
        
        return memory_id
    
    async def retrieve_recent_events(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve the most recent events from episodic memory.
        
        Args:
            limit: Maximum number of events to retrieve
            
        Returns:
            List of recent events
        """
        # Use the memory store to retrieve recent memories
        # For this basic implementation, we'll use a simple search
        results = await self.memory_store.retrieve_by_keyword("EVENT:", limit)
        
        # Process the results to extract event data
        events = []
        for result in results:
            memory = result["memory"]
            
            try:
                event_data = json.loads(memory["output"])
                event_type = memory["metadata"].get("event_type", "unknown")
                
                events.append({
                    "id": memory["id"],
                    "timestamp": memory["timestamp"],
                    "type": event_type,
                    "data": event_data
                })
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse event data for memory {memory['id']}")
        
        return events
    
    async def retrieve_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve the most recent conversation turns from episodic memory.
        
        Args:
            limit: Maximum number of conversation turns to retrieve
            
        Returns:
            List of recent conversation turns
        """
        # Use the memory store to retrieve conversations
        conversations = []
        
        for memory in self.memory_store.episodic_memory:
            if memory["type"] == "interaction" and memory["metadata"].get("event_type") == "conversation":
                conversations.append({
                    "id": memory["id"],
                    "timestamp": memory["timestamp"],
                    "user_input": memory["input"],
                    "system_output": memory["output"],
                    "context": memory["metadata"].get("context", {})
                })
                
                if len(conversations) >= limit:
                    break
        
        # Sort by timestamp (newest first)
        conversations.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return conversations[:limit]
    
    async def retrieve_action_sequences(self, goal_keyword: Optional[str] = None, 
                                     success: Optional[bool] = None,
                                     limit: int = 5) -> List[Dict]:
        """
        Retrieve action sequences matching specified criteria.
        
        Args:
            goal_keyword: Optional keyword to search in goals
            success: Optional filter for successful/unsuccessful sequences
            limit: Maximum number of sequences to retrieve
            
        Returns:
            List of matching action sequences
        """
        # Use the memory store to retrieve action sequences
        results = await self.memory_store.retrieve_by_keyword("action_sequence", 100)  # Get more to filter
        
        # Filter the results
        filtered_sequences = []
        for result in results:
            memory = result["memory"]
            
            try:
                event_data = json.loads(memory["output"])
                
                # Apply filters
                if goal_keyword and goal_keyword.lower() not in event_data.get("goal", "").lower():
                    continue
                
                if success is not None and event_data.get("success") != success:
                    continue
                
                filtered_sequences.append({
                    "id": memory["id"],
                    "timestamp": memory["timestamp"],
                    "goal": event_data.get("goal", ""),
                    "actions": event_data.get("actions", []),
                    "outcome": event_data.get("outcome", {}),
                    "success": event_data.get("success", False)
                })
                
                if len(filtered_sequences) >= limit:
                    break
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse event data for memory {memory['id']}")
        
        return filtered_sequences
    
    async def retrieve_reflections(self, topic_keyword: Optional[str] = None, 
                               limit: int = 5) -> List[Dict]:
        """
        Retrieve reflections matching specified criteria.
        
        Args:
            topic_keyword: Optional keyword to search in topics
            limit: Maximum number of reflections to retrieve
            
        Returns:
            List of matching reflections
        """
        # Use the memory store to retrieve reflections
        results = await self.memory_store.retrieve_by_keyword("reflection", 100)  # Get more to filter
        
        # Filter the results
        filtered_reflections = []
        for result in results:
            memory = result["memory"]
            
            try:
                event_data = json.loads(memory["output"])
                
                # Apply filters
                if topic_keyword and topic_keyword.lower() not in event_data.get("topic", "").lower():
                    continue
                
                filtered_reflections.append({
                    "id": memory["id"],
                    "timestamp": memory["timestamp"],
                    "topic": event_data.get("topic", ""),
                    "thoughts": event_data.get("thoughts", ""),
                    "insights": event_data.get("insights", [])
                })
                
                if len(filtered_reflections) >= limit:
                    break
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse event data for memory {memory['id']}")
        
        return filtered_reflections
    
    async def retrieve_event_by_id(self, event_id: str) -> Optional[Dict]:
        """
        Retrieve a specific event by its ID.
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            The event data if found, None otherwise
        """
        # Find the memory in the memory store
        for memory in self.memory_store.episodic_memory:
            if memory["id"] == event_id and memory["type"] == "interaction":
                try:
                    if "EVENT:" in memory["input"]:
                        event_data = json.loads(memory["output"])
                        event_type = memory["metadata"].get("event_type", "unknown")
                        
                        return {
                            "id": memory["id"],
                            "timestamp": memory["timestamp"],
                            "type": event_type,
                            "data": event_data
                        }
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse event data for memory {memory['id']}")
        
        return None
    
    async def retrieve_similar_events(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve events similar to the query.
        
        Args:
            query: The search query
            limit: Maximum number of events to retrieve
            
        Returns:
            List of similar events
        """
        # Use the memory store's similarity search
        results = await self.memory_store.retrieve_by_similarity(query, limit * 2)  # Get more to filter
        
        # Filter for events only
        events = []
        for result in results:
            memory = result["memory"]
            
            if memory["type"] == "interaction" and "EVENT:" in memory["input"]:
                try:
                    event_data = json.loads(memory["output"])
                    event_type = memory["metadata"].get("event_type", "unknown")
                    
                    events.append({
                        "id": memory["id"],
                        "timestamp": memory["timestamp"],
                        "type": event_type,
                        "data": event_data,
                        "similarity": result.get("similarity", 0)
                    })
                    
                    if len(events) >= limit:
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse event data for memory {memory['id']}")
        
        return events
