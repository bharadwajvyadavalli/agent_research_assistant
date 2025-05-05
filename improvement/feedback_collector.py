"""
Feedback collection module for the agent research assistant.

This module provides functionality to collect, store, and analyze feedback
on agent performance from various sources including:
- Human evaluators
- Self-critique
- Automated evaluation metrics
- Peer agent reviews
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from utils.logging_utils import AgentLogger

class FeedbackItem:
    """Class representing a single piece of feedback."""
    
    def __init__(
        self,
        source: str,
        feedback_type: str,
        content: str,
        target_agent_id: Optional[str] = None,
        target_task_id: Optional[str] = None,
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new feedback item.
        
        Args:
            source: Source of the feedback (human, agent ID, or system)
            feedback_type: Type of feedback (critique, suggestion, rating, etc.)
            content: Textual content of the feedback
            target_agent_id: ID of the agent receiving feedback (if applicable)
            target_task_id: ID of the task the feedback is about (if applicable)
            rating: Numerical rating if applicable (e.g., 1-5 scale)
            metadata: Additional metadata related to this feedback
        """
        self.id = f"feedback_{int(time.time() * 1000)}_{hash(content) % 10000}"
        self.timestamp = datetime.now().isoformat()
        self.source = source
        self.feedback_type = feedback_type
        self.content = content
        self.target_agent_id = target_agent_id
        self.target_task_id = target_task_id
        self.rating = rating
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the feedback item to a dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "feedback_type": self.feedback_type,
            "content": self.content,
            "target_agent_id": self.target_agent_id,
            "target_task_id": self.target_task_id,
            "rating": self.rating,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create a feedback item from a dictionary."""
        feedback = cls(
            source=data["source"],
            feedback_type=data["feedback_type"],
            content=data["content"],
            target_agent_id=data.get("target_agent_id"),
            target_task_id=data.get("target_task_id"),
            rating=data.get("rating"),
            metadata=data.get("metadata", {})
        )
        feedback.id = data["id"]
        feedback.timestamp = data["timestamp"]
        return feedback


class FeedbackCollector:
    """
    Class for collecting and storing feedback from various sources.
    """
    
    def __init__(
        self,
        storage_dir: str = "data/feedback",
        session_id: Optional[str] = None
    ):
        """
        Initialize the feedback collector.
        
        Args:
            storage_dir: Directory for storing feedback data
            session_id: Current session ID (generated if not provided)
        """
        self.storage_dir = storage_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.feedback_items: List[FeedbackItem] = []
        self.logger = AgentLogger(
            agent_id="feedback_collector",
            session_id=self.session_id
        )
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing feedback if available
        self._load_feedback()
        
    def _get_feedback_file_path(self) -> str:
        """Get the path to the feedback storage file."""
        return os.path.join(self.storage_dir, f"feedback_{self.session_id}.json")
    
    def _load_feedback(self) -> None:
        """Load existing feedback from storage."""
        file_path = self._get_feedback_file_path()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.feedback_items = [FeedbackItem.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.feedback_items)} feedback items from storage")
            except Exception as e:
                self.logger.error(f"Failed to load feedback: {str(e)}")
    
    def _save_feedback(self) -> None:
        """Save feedback to storage."""
        file_path = self._get_feedback_file_path()
        try:
            with open(file_path, 'w') as f:
                json.dump([item.to_dict() for item in self.feedback_items], f, indent=2)
            self.logger.debug(f"Saved {len(self.feedback_items)} feedback items to storage")
        except Exception as e:
            self.logger.error(f"Failed to save feedback: {str(e)}")
    
    def add_feedback(self, feedback_item: FeedbackItem) -> str:
        """
        Add a new feedback item to the collection.
        
        Args:
            feedback_item: The feedback item to add
            
        Returns:
            The ID of the added feedback item
        """
        self.feedback_items.append(feedback_item)
        self.logger.info(
            f"Added feedback from {feedback_item.source} on " +
            f"{feedback_item.target_agent_id or 'system'}"
        )
        self._save_feedback()
        return feedback_item.id
    
    def create_feedback(
        self,
        source: str,
        feedback_type: str,
        content: str,
        target_agent_id: Optional[str] = None,
        target_task_id: Optional[str] = None,
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create and add a new feedback item.
        
        Args:
            source: Source of the feedback (human, agent ID, or system)
            feedback_type: Type of feedback (critique, suggestion, rating, etc.)
            content: Textual content of the feedback
            target_agent_id: ID of the agent receiving feedback (if applicable)
            target_task_id: ID of the task the feedback is about (if applicable)
            rating: Numerical rating if applicable (e.g., 1-5 scale)
            metadata: Additional metadata related to this feedback
            
        Returns:
            The ID of the created feedback item
        """
        feedback_item = FeedbackItem(
            source=source,
            feedback_type=feedback_type,
            content=content,
            target_agent_id=target_agent_id,
            target_task_id=target_task_id,
            rating=rating,
            metadata=metadata
        )
        return self.add_feedback(feedback_item)
    
    def get_feedback(self, feedback_id: str) -> Optional[FeedbackItem]:
        """
        Get a specific feedback item by ID.
        
        Args:
            feedback_id: ID of the feedback to retrieve
            
        Returns:
            The feedback item if found, None otherwise
        """
        for item in self.feedback_items:
            if item.id == feedback_id:
                return item
        return None
    
    def get_feedback_for_agent(self, agent_id: str) -> List[FeedbackItem]:
        """
        Get all feedback for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of feedback items for the agent
        """
        return [item for item in self.feedback_items if item.target_agent_id == agent_id]
    
    def get_feedback_for_task(self, task_id: str) -> List[FeedbackItem]:
        """
        Get all feedback for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of feedback items for the task
        """
        return [item for item in self.feedback_items if item.target_task_id == task_id]
    
    def get_all_feedback(self) -> List[FeedbackItem]:
        """Get all feedback items."""
        return self.feedback_items
    
    def get_avg_rating_for_agent(self, agent_id: str) -> Optional[float]:
        """
        Calculate the average rating for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Average rating or None if no ratings available
        """
        ratings = [item.rating for item in self.get_feedback_for_agent(agent_id) 
                  if item.rating is not None]
        if not ratings:
            return None
        return sum(ratings) / len(ratings)
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all feedback.
        
        Returns:
            Dictionary with feedback summary statistics
        """
        agents = {}
        tasks = {}
        types = {}
        sources = {}
        
        for item in self.feedback_items:
            # Count by agent
            if item.target_agent_id:
                agents[item.target_agent_id] = agents.get(item.target_agent_id, 0) + 1
            
            # Count by task
            if item.target_task_id:
                tasks[item.target_task_id] = tasks.get(item.target_task_id, 0) + 1
            
            # Count by feedback type
            types[item.feedback_type] = types.get(item.feedback_type, 0) + 1
            
            # Count by source
            sources[item.source] = sources.get(item.source, 0) + 1
        
        return {
            "total_feedback": len(self.feedback_items),
            "feedback_by_agent": agents,
            "feedback_by_task": tasks,
            "feedback_by_type": types,
            "feedback_by_source": sources
        }
    
    def process_human_feedback(
        self, 
        content: str, 
        target_agent_id: Optional[str] = None,
        target_task_id: Optional[str] = None,
        rating: Optional[float] = None
    ) -> str:
        """
        Process feedback from a human user.
        
        Args:
            content: Textual content of the feedback
            target_agent_id: ID of the agent receiving feedback (if applicable)
            target_task_id: ID of the task the feedback is about (if applicable)
            rating: Numerical rating if applicable (e.g., 1-5 scale)
            
        Returns:
            The ID of the created feedback item
        """
        return self.create_feedback(
            source="human",
            feedback_type="user_feedback",
            content=content,
            target_agent_id=target_agent_id,
            target_task_id=target_task_id,
            rating=rating,
            metadata={"timestamp": datetime.now().isoformat()}
        )
    
    def process_agent_feedback(
        self,
        source_agent_id: str,
        content: str,
        target_agent_id: str,
        feedback_type: str = "peer_review",
        target_task_id: Optional[str] = None,
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process feedback from one agent about another.
        
        Args:
            source_agent_id: ID of the agent providing feedback
            content: Textual content of the feedback
            target_agent_id: ID of the agent receiving feedback
            feedback_type: Type of feedback (default: peer_review)
            target_task_id: ID of the task the feedback is about (if applicable)
            rating: Numerical rating if applicable (e.g., 1-5 scale)
            metadata: Additional metadata related to this feedback
            
        Returns:
            The ID of the created feedback item
        """
        return self.create_feedback(
            source=source_agent_id,
            feedback_type=feedback_type,
            content=content,
            target_agent_id=target_agent_id,
            target_task_id=target_task_id,
            rating=rating,
            metadata=metadata or {}
        )
    
    def process_self_critique(
        self,
        agent_id: str,
        content: str,
        target_task_id: Optional[str] = None,
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process self-critique from an agent.
        
        Args:
            agent_id: ID of the agent providing self-critique
            content: Textual content of the self-critique
            target_task_id: ID of the task the feedback is about (if applicable)
            rating: Self-rating if applicable (e.g., 1-5 scale)
            metadata: Additional metadata related to this feedback
            
        Returns:
            The ID of the created feedback item
        """
        return self.create_feedback(
            source=agent_id,
            feedback_type="self_critique",
            content=content,
            target_agent_id=agent_id,  # Self-critique targets the same agent
            target_task_id=target_task_id,
            rating=rating,
            metadata=metadata or {}
        )
    
    def process_system_feedback(
        self,
        content: str,
        target_agent_id: Optional[str] = None,
        target_task_id: Optional[str] = None,
        feedback_type: str = "system_evaluation",
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process automated system feedback.
        
        Args:
            content: Textual content of the feedback
            target_agent_id: ID of the agent receiving feedback (if applicable)
            target_task_id: ID of the task the feedback is about (if applicable)
            feedback_type: Type of feedback (default: system_evaluation)
            rating: Numerical rating if applicable (e.g., 1-5 scale)
            metadata: Additional metadata related to this feedback
            
        Returns:
            The ID of the created feedback item
        """
        return self.create_feedback(
            source="system",
            feedback_type=feedback_type,
            content=content,
            target_agent_id=target_agent_id,
            target_task_id=target_task_id,
            rating=rating,
            metadata=metadata or {}
        )
    
    def extract_improvement_suggestions(self, agent_id: str) -> List[str]:
        """
        Extract actionable improvement suggestions for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        for item in self.get_feedback_for_agent(agent_id):
            # Simple heuristic: look for keywords in feedback that indicate suggestions
            content = item.content.lower()
            keywords = ["suggest", "improve", "better if", "try to", "should", "could"]
            
            if any(keyword in content for keyword in keywords):
                suggestions.append(item.content)
        
        return suggestions
    
    def analyze_feedback_trends(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trends in feedback over time.
        
        Args:
            agent_id: Optional agent ID to filter feedback
            
        Returns:
            Dictionary with trend analysis results
        """
        items = self.get_feedback_for_agent(agent_id) if agent_id else self.feedback_items
        
        # Sort by timestamp
        items.sort(key=lambda x: x.timestamp)
        
        # Group by day
        feedback_by_day = {}
        ratings_by_day = {}
        
        for item in items:
            day = item.timestamp.split("T")[0]
            feedback_by_day[day] = feedback_by_day.get(day, 0) + 1
            
            if item.rating is not None:
                if day not in ratings_by_day:
                    ratings_by_day[day] = []
                ratings_by_day[day].append(item.rating)
        
        # Calculate average ratings by day
        avg_ratings_by_day = {}
        for day, ratings in ratings_by_day.items():
            avg_ratings_by_day[day] = sum(ratings) / len(ratings)
        
        return {
            "total_feedback": len(items),
            "feedback_by_day": feedback_by_day,
            "avg_ratings_by_day": avg_ratings_by_day
        }
