"""
Self-improvement module for the agent research assistant.

This module provides functionality for agents to improve themselves based on
feedback, past performance, and learning from experience.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict

from utils.logging_utils import AgentLogger
from .feedback_collector import FeedbackCollector, FeedbackItem

class ImprovementArea:
    """Class representing an area for agent improvement."""
    
    def __init__(
        self,
        name: str,
        description: str,
        priority: float = 0.0,
        supporting_feedback: Optional[List[str]] = None,
        improvement_strategies: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a new improvement area.
        
        Args:
            name: Short name for the improvement area
            description: Detailed description of what needs improvement
            priority: Priority score (0.0-1.0, higher is more important)
            supporting_feedback: List of feedback IDs supporting this improvement area
            improvement_strategies: List of strategies to address this area
        """
        self.id = f"area_{int(time.time())}_{hash(name) % 10000}"
        self.name = name
        self.description = description
        self.priority = priority
        self.supporting_feedback = supporting_feedback or []
        self.improvement_strategies = improvement_strategies or []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = "identified"  # identified, in_progress, implemented, verified
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the improvement area to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "supporting_feedback": self.supporting_feedback,
            "improvement_strategies": self.improvement_strategies,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementArea':
        """Create an improvement area from a dictionary."""
        area = cls(
            name=data["name"],
            description=data["description"],
            priority=data["priority"],
            supporting_feedback=data["supporting_feedback"],
            improvement_strategies=data["improvement_strategies"]
        )
        area.id = data["id"]
        area.created_at = data["created_at"]
        area.updated_at = data["updated_at"]
        area.status = data["status"]
        return area
    
    def add_strategy(self, strategy: Dict[str, Any]) -> None:
        """
        Add an improvement strategy to this area.
        
        Args:
            strategy: Dictionary describing the improvement strategy
                     Should contain at least 'description' and 'type' keys
        """
        if "created_at" not in strategy:
            strategy["created_at"] = datetime.now().isoformat()
        if "status" not in strategy:
            strategy["status"] = "proposed"
        
        self.improvement_strategies.append(strategy)
        self.updated_at = datetime.now().isoformat()


class ImprovementPlan:
    """Class representing a plan for improving an agent."""
    
    def __init__(
        self,
        agent_id: str,
        areas: Optional[List[ImprovementArea]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize a new improvement plan.
        
        Args:
            agent_id: ID of the agent this plan is for
            areas: List of improvement areas
            name: Name of the improvement plan
            description: Description of the improvement plan
        """
        self.id = f"plan_{agent_id}_{int(time.time())}"
        self.agent_id = agent_id
        self.areas = areas or []
        self.name = name or f"Improvement Plan for {agent_id}"
        self.description = description or f"Automated improvement plan for agent {agent_id}"
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = "draft"  # draft, active, completed, archived
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the improvement plan to a dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "areas": [area.to_dict() for area in self.areas],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementPlan':
        """Create an improvement plan from a dictionary."""
        plan = cls(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"]
        )
        plan.id = data["id"]
        plan.areas = [ImprovementArea.from_dict(area) for area in data["areas"]]
        plan.created_at = data["created_at"]
        plan.updated_at = data["updated_at"]
        plan.status = data["status"]
        return plan
    
    def add_area(self, area: ImprovementArea) -> None:
        """
        Add an improvement area to this plan.
        
        Args:
            area: The improvement area to add
        """
        self.areas.append(area)
        self.updated_at = datetime.now().isoformat()
    
    def get_area_by_id(self, area_id: str) -> Optional[ImprovementArea]:
        """
        Get an improvement area by ID.
        
        Args:
            area_id: ID of the improvement area
            
        Returns:
            The improvement area if found, None otherwise
        """
        for area in self.areas:
            if area.id == area_id:
                return area
        return None
    
    def get_prioritized_areas(self) -> List[ImprovementArea]:
        """
        Get improvement areas sorted by priority.
        
        Returns:
            List of improvement areas sorted by priority (highest first)
        """
        return sorted(self.areas, key=lambda x: x.priority, reverse=True)
    
    def get_implementation_status(self) -> Dict[str, int]:
        """
        Get the implementation status of improvement strategies.
        
        Returns:
            Dictionary with counts of strategies by status
        """
        status_counts = defaultdict(int)
        
        for area in self.areas:
            for strategy in area.improvement_strategies:
                status = strategy.get("status", "unknown")
                status_counts[status] += 1
        
        return dict(status_counts)
    
    def update_status(self) -> None:
        """Update the overall status of the improvement plan based on areas."""
        if not self.areas:
            self.status = "draft"
            return
        
        # Check if all areas are verified
        if all(area.status == "verified" for area in self.areas):
            self.status = "completed"
        # Check if all areas are at least implemented
        elif all(area.status in ["implemented", "verified"] for area in self.areas):
            self.status = "active"
        else:
            self.status = "active"
        
        self.updated_at = datetime.now().isoformat()


class SelfImprovement:
    """
    Class for managing agent self-improvement based on feedback and experience.
    """
    
    def __init__(
        self,
        agent_id: str,
        feedback_collector: Optional[FeedbackCollector] = None,
        storage_dir: str = "data/improvement",
        session_id: Optional[str] = None
    ):
        """
        Initialize the self-improvement manager.
        
        Args:
            agent_id: ID of the agent this manager is for
            feedback_collector: FeedbackCollector instance or None to create new
            storage_dir: Directory for storing improvement data
            session_id: Current session ID (generated if not provided)
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create or use feedback collector
        if feedback_collector:
            self.feedback_collector = feedback_collector
        else:
            self.feedback_collector = FeedbackCollector(
                session_id=self.session_id
            )
        
        self.logger = AgentLogger(
            agent_id=f"{agent_id}_improvement",
            session_id=self.session_id
        )
        
        # Current improvement plan
        self.current_plan: Optional[ImprovementPlan] = None
        
        # Historical improvement plans
        self.historical_plans: List[ImprovementPlan] = []
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing improvement data
        self._load_improvement_data()
    
    def _get_improvement_file_path(self) -> str:
        """Get the path to the improvement data storage file."""
        return os.path.join(self.storage_dir, f"improvement_{self.agent_id}.json")
    
    def _load_improvement_data(self) -> None:
        """Load existing improvement data from storage."""
        file_path = self._get_improvement_file_path()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load current plan
                    if "current_plan" in data and data["current_plan"]:
                        self.current_plan = ImprovementPlan.from_dict(data["current_plan"])
                    
                    # Load historical plans
                    if "historical_plans" in data:
                        self.historical_plans = [
                            ImprovementPlan.from_dict(plan) for plan in data["historical_plans"]
                        ]
                
                self.logger.info(f"Loaded improvement data with {len(self.historical_plans)} historical plans")
            
            except Exception as e:
                self.logger.error(f"Failed to load improvement data: {str(e)}")
    
    def _save_improvement_data(self) -> None:
        """Save improvement data to storage."""
        file_path = self._get_improvement_file_path()
        try:
            data = {
                "current_plan": self.current_plan.to_dict() if self.current_plan else None,
                "historical_plans": [plan.to_dict() for plan in self.historical_plans]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved improvement data to storage")
        
        except Exception as e:
            self.logger.error(f"Failed to save improvement data: {str(e)}")
    
    def create_new_plan(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ImprovementPlan:
        """
        Create a new improvement plan for the agent.
        If there's an active plan, it will be archived.
        
        Args:
            name: Name for the new plan
            description: Description for the new plan
            
        Returns:
            The newly created improvement plan
        """
        # Archive current plan if it exists
        if self.current_plan:
            self.current_plan.status = "archived"
            self.historical_plans.append(self.current_plan)
        
        # Create new plan
        self.current_plan = ImprovementPlan(
            agent_id=self.agent_id,
            name=name,
            description=description
        )
        
        self.logger.info(f"Created new improvement plan: {self.current_plan.name}")
        self._save_improvement_data()
        return self.current_plan
    
    def add_improvement_area(
        self,
        name: str,
        description: str,
        priority: float = 0.5,
        supporting_feedback: Optional[List[str]] = None
    ) -> ImprovementArea:
        """
        Add a new improvement area to the current plan.
        
        Args:
            name: Short name for the improvement area
            description: Detailed description of what needs improvement
            priority: Priority score (0.0-1.0, higher is more important)
            supporting_feedback: List of feedback IDs supporting this area
            
        Returns:
            The newly created improvement area
        """
        # Create plan if needed
        if not self.current_plan:
            self.create_new_plan()
        
        # Create and add the improvement area
        area = ImprovementArea(
            name=name,
            description=description,
            priority=priority,
            supporting_feedback=supporting_feedback
        )
        
        self.current_plan.add_area(area)
        self.logger.info(f"Added improvement area: {name}")
        self._save_improvement_data()
        return area
    
    def add_improvement_strategy(
        self,
        area_id: str,
        description: str,
        strategy_type: str = "behavioral",
        implementation_steps: Optional[List[str]] = None,
        estimated_difficulty: float = 0.5,
        estimated_impact: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Add an improvement strategy to an existing area.
        
        Args:
            area_id: ID of the improvement area
            description: Description of the improvement strategy
            strategy_type: Type of strategy (behavioral, architectural, learning, etc.)
            implementation_steps: List of steps to implement this strategy
            estimated_difficulty: Estimated difficulty (0.0-1.0)
            estimated_impact: Estimated impact (0.0-1.0)
            
        Returns:
            The strategy dict if successful, None if area not found
        """
        if not self.current_plan:
            self.logger.warning("No current improvement plan exists")
            return None
        
        area = self.current_plan.get_area_by_id(area_id)
        if not area:
            self.logger.warning(f"Improvement area not found: {area_id}")
            return None
        
        # Create the strategy
        strategy = {
            "id": f"strategy_{int(time.time())}_{hash(description) % 10000}",
            "description": description,
            "type": strategy_type,
            "implementation_steps": implementation_steps or [],
            "estimated_difficulty": estimated_difficulty,
            "estimated_impact": estimated_impact,
            "created_at": datetime.now().isoformat(),
            "status": "proposed"
        }
        
        # Add to the area
        area.add_strategy(strategy)
        self.logger.info(f"Added improvement strategy to area {area.name}")
        self._save_improvement_data()
        return strategy
    
    def update_strategy_status(
        self,
        area_id: str,
        strategy_id: str,
        new_status: str
    ) -> bool:
        """
        Update the status of an improvement strategy.
        
        Args:
            area_id: ID of the improvement area
            strategy_id: ID of the strategy
            new_status: New status (proposed, in_progress, implemented, verified)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_plan:
            self.logger.warning("No current improvement plan exists")
            return False
        
        area = self.current_plan.get_area_by_id(area_id)
        if not area:
            self.logger.warning(f"Improvement area not found: {area_id}")
            return False
        
        # Find and update the strategy
        for strategy in area.improvement_strategies:
            if strategy.get("id") == strategy_id:
                strategy["status"] = new_status
                strategy["updated_at"] = datetime.now().isoformat()
                
                self.logger.info(f"Updated strategy status to {new_status}")
                self._save_improvement_data()
                
                # Check if all strategies in this area are implemented/verified
                if all(s.get("status") in ["implemented", "verified"] 
                       for s in area.improvement_strategies):
                    area.status = "implemented"
                    # If all are verified, mark area as verified
                    if all(s.get("status") == "verified" for s in area.improvement_strategies):
                        area.status = "verified"
                
                # Update the plan status
                self.current_plan.update_status()
                self._save_improvement_data()
                return True
        
        self.logger.warning(f"Strategy not found: {strategy_id}")
        return False
    
    def analyze_feedback_for_improvements(self) -> List[Dict[str, Any]]:
        """
        Analyze feedback to identify potential improvement areas.
        Uses basic heuristics to look for patterns in feedback.
        
        Returns:
            List of potential improvement areas
        """
        # Get feedback for this agent
        feedback_items = self.feedback_collector.get_feedback_for_agent(self.agent_id)
        if not feedback_items:
            self.logger.info("No feedback available for analysis")
            return []
        
        # Extract common themes from feedback
        themes = self._extract_feedback_themes(feedback_items)
        
        # Convert themes to potential improvement areas
        potential_areas = []
        for theme, data in themes.items():
            if len(data["feedback_ids"]) >= 2:  # Consider themes mentioned in at least 2 feedback items
                potential_areas.append({
                    "name": theme,
                    "description": f"Improvement area identified from feedback: {theme}",
                    "priority": min(1.0, 0.3 + (len(data['feedback_ids']) * 0.1)),  # Higher priority for more mentions
                    "supporting_feedback": data["feedback_ids"],
                    "feedback_snippets": data["snippets"][:5]  # Include up to 5 example snippets
                })
        
        # Sort by priority (descending)
        potential_areas.sort(key=lambda x: x["priority"], reverse=True)
        
        self.logger.info(f"Identified {len(potential_areas)} potential improvement areas from feedback")
        return potential_areas
    
    def _extract_feedback_themes(self, feedback_items: List[FeedbackItem]) -> Dict[str, Dict[str, Any]]:
        """
        Extract common themes from feedback using simple keyword analysis.
        
        Args:
            feedback_items: List of feedback items to analyze
            
        Returns:
            Dictionary of themes with supporting feedback
        """
        # Simple keyword sets for different aspects of agent behavior
        keywords = {
            "Information Accuracy": ["accuracy", "incorrect", "wrong", "accurate", "fact", "error"],
            "Response Time": ["slow", "fast", "time", "quick", "delay", "speed"],
            "Reasoning": ["reasoning", "logic", "rational", "thinking", "conclusion", "inference"],
            "Search Strategy": ["search", "query", "lookup", "find", "information retrieval"],
            "Planning": ["plan", "strategy", "approach", "organize", "structure"],
            "Execution": ["execute", "implement", "perform", "carry out", "action"],
            "Collaboration": ["collaborate", "teamwork", "coordinate", "together", "cooperate"],
            "Learning": ["learn", "improve", "adapt", "adjust", "growth"],
            "Communication": ["communicate", "explain", "clear", "understandable", "articulate"],
            "Creativity": ["creative", "novel", "innovative", "unique", "original"],
            "Critical Thinking": ["critical", "evaluate", "assess", "analyze", "critique"]
        }
        
        # Initialize themes dictionary
        themes = {theme: {"feedback_ids": [], "snippets": []} for theme in keywords}
        
        # Analyze each feedback item
        for item in feedback_items:
            content = item.content.lower()
            
            # Check for each theme's keywords
            for theme, theme_keywords in keywords.items():
                for keyword in theme_keywords:
                    if keyword.lower() in content:
                        if item.id not in themes[theme]["feedback_ids"]:
                            themes[theme]["feedback_ids"].append(item.id)
                            
                            # Find a relevant snippet (simple approach - just the sentence with the keyword)
                            sentences = content.split('.')
                            for sentence in sentences:
                                if keyword.lower() in sentence.lower():
                                    snippet = sentence.strip() + '.'
                                    if snippet not in themes[theme]["snippets"]:
                                        themes[theme]["snippets"].append(snippet)
                                    break
                        
                        break  # Only count each theme once per feedback item
        
        # Remove themes with no mentions
        themes = {theme: data for theme, data in themes.items() if data["feedback_ids"]}
        
        return themes
    
    def generate_improvement_plan_from_feedback(self) -> Optional[ImprovementPlan]:
        """
        Generate a new improvement plan based on feedback analysis.
        
        Returns:
            The newly created improvement plan or None if not enough feedback
        """
        # Analyze feedback for potential improvement areas
        potential_areas = self.analyze_feedback_for_improvements()
        if not potential_areas:
            self.logger.info("Not enough feedback to generate an improvement plan")
            return None
        
        # Create a new plan
        new_plan = self.create_new_plan(
            name=f"Feedback-based Improvement Plan for {self.agent_id}",
            description=f"Improvement plan generated from feedback analysis on {datetime.now().isoformat()}"
        )
        
        # Add the top improvement areas (max 5)
        for area_data in potential_areas[:5]:
            area = self.add_improvement_area(
                name=area_data["name"],
                description=area_data["description"],
                priority=area_data["priority"],
                supporting_feedback=area_data["supporting_feedback"]
            )
            
            # For each area, add a generic strategy
            self.add_improvement_strategy(
                area_id=area.id,
                description=f"Improve {area.name.lower()} based on feedback",
                strategy_type="behavioral",
                implementation_steps=[
                    f"Review feedback related to {area.name.lower()}",
                    "Identify specific behaviors to change",
                    "Implement changes in agent behavior",
                    "Test the effectiveness of changes",
                    "Gather new feedback to verify improvement"
                ]
            )
        
        self.logger.info(f"Generated new improvement plan with {len(new_plan.areas)} areas")
        return new_plan
    
    def record_self_assessment(
        self,
        task_id: str,
        assessment: str,
        strengths: List[str],
        weaknesses: List[str],
        rating: float,
        improvement_ideas: Optional[List[str]] = None
    ) -> str:
        """
        Record a self-assessment by the agent.
        
        Args:
            task_id: ID of the task being assessed
            assessment: Overall self-assessment text
            strengths: List of identified strengths
            weaknesses: List of identified weaknesses
            rating: Self-rating (0.0-1.0)
            improvement_ideas: Optional list of improvement ideas
            
        Returns:
            The ID of the created feedback item
        """
        # Create a structured self-assessment
        assessment_data = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_ideas": improvement_ideas or [],
            "rating": rating
        }
        
        # Record as self-critique feedback
        feedback_id = self.feedback_collector.process_self_critique(
            agent_id=self.agent_id,
            content=assessment,
            target_task_id=task_id,
            rating=rating,
            metadata=assessment_data
        )
        
        self.logger.info(f"Recorded self-assessment for task {task_id}")
        
        # If there are weaknesses, automatically add them as improvement areas
        if self.current_plan and weaknesses:
            for weakness in weaknesses:
                # Check if a similar area already exists
                if not any(area.name.lower() == weakness.lower() for area in self.current_plan.areas):
                    area = self.add_improvement_area(
                        name=weakness,
                        description=f"Self-identified weakness: {weakness}",
                        priority=0.7,  # Higher priority for self-identified issues
                        supporting_feedback=[feedback_id]
                    )
                    
                    # Add a strategy if improvement ideas were provided
                    if improvement_ideas:
                        relevant_ideas = [idea for idea in improvement_ideas 
                                         if any(word in idea.lower() for word in weakness.lower().split())]
                        
                        if relevant_ideas:
                            self.add_improvement_strategy(
                                area_id=area.id,
                                description=relevant_ideas[0],
                                strategy_type="self-improvement",
                                implementation_steps=[
                                    f"Focus on improving {weakness}",
                                    "Practice in controlled test scenarios",
                                    "Apply improvements in real tasks",
                                    "Reflect on effectiveness"
                                ]
                            )
        
        return feedback_id
    
    def evaluate_improvement_progress(self) -> Dict[str, Any]:
        """
        Evaluate progress on the current improvement plan.
        
        Returns:
            Dictionary with progress metrics
        """
        if not self.current_plan:
            return {"status": "no_plan", "message": "No active improvement plan"}
        
        # Get all strategies across all areas
        all_strategies = []
        for area in self.current_plan.areas:
            all_strategies.extend(area.improvement_strategies)
        
        # Count by status
        status_counts = defaultdict(int)
        for strategy in all_strategies:
            status = strategy.get("status", "unknown")
            status_counts[status] += 1
        
        # Calculate completion percentage
        total_strategies = len(all_strategies)
        completed = status_counts.get("implemented", 0) + status_counts.get("verified", 0)
        completion_percentage = (completed / total_strategies) * 100 if total_strategies > 0 else 0
        
        # Get prioritized area statuses
        area_statuses = [
            {
                "id": area.id,
                "name": area.name,
                "priority": area.priority,
                "status": area.status,
                "strategies_count": len(area.improvement_strategies),
                "implemented_count": sum(1 for s in area.improvement_strategies 
                                        if s.get("status") in ["implemented", "verified"])
            }
            for area in self.current_plan.get_prioritized_areas()
        ]
        
        return {
            "plan_id": self.current_plan.id,
            "plan_name": self.current_plan.name,
            "plan_status": self.current_plan.status,
            "total_areas": len(self.current_plan.areas),
            "total_strategies": total_strategies,
            "status_counts": dict(status_counts),
            "completion_percentage": completion_percentage,
            "area_statuses": area_statuses,
            "last_updated": self.current_plan.updated_at
        }
    
    def identify_improvement_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in improvements over time across historical plans.
        
        Returns:
            Dictionary with identified patterns
        """
        if not self.historical_plans:
            return {"status": "no_history", "message": "No historical improvement plans"}
        
        # Combine current plan (if exists) with historical plans for analysis
        all_plans = self.historical_plans.copy()
        if self.current_plan:
            all_plans.append(self.current_plan)
        
        # Track recurring themes
        theme_counts = defaultdict(int)
        theme_priorities = defaultdict(list)
        theme_first_appearance = {}
        theme_last_appearance = {}
        
        # Analyze improvement areas across all plans
        for plan in all_plans:
            plan_date = datetime.fromisoformat(plan.created_at.split("+")[0])
            
            for area in plan.areas:
                # Normalize area name to identify similar themes
                normalized_name = area.name.lower()
                
                # Update theme tracking
                theme_counts[normalized_name] += 1
                theme_priorities[normalized_name].append(area.priority)
                
                # Track first and last appearance
                if normalized_name not in theme_first_appearance:
                    theme_first_appearance[normalized_name] = plan_date
                theme_last_appearance[normalized_name] = plan_date
        
        # Identify recurring themes (mentioned in multiple plans)
        recurring_themes = []
        for theme, count in theme_counts.items():
            if count > 1:
                avg_priority = sum(theme_priorities[theme]) / len(theme_priorities[theme])
                time_span = (theme_last_appearance[theme] - theme_first_appearance[theme]).days
                
                recurring_themes.append({
                    "theme": theme,
                    "occurrence_count": count,
                    "average_priority": avg_priority,
                    "first_appearance": theme_first_appearance[theme].isoformat(),
                    "last_appearance": theme_last_appearance[theme].isoformat(),
                    "time_span_days": time_span
                })
        
        # Sort by occurrence count (descending)
        recurring_themes.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        # Analyze improvement over time
        improvement_trajectory = {}
        if len(all_plans) >= 2:
            # Sort plans by date
            sorted_plans = sorted(all_plans, key=lambda x: x.created_at)
            
            # Look at when areas were marked as completed
            for i, plan in enumerate(sorted_plans):
                # Skip the last plan as it's likely the current one
                if i == len(sorted_plans) - 1:
                    continue
                
                for area in plan.areas:
                    if area.status in ["implemented", "verified"]:
                        normalized_name = area.name.lower()
                        
                        # Check if this theme reappears in later plans
                        reappears = False
                        for later_plan in sorted_plans[i+1:]:
                            if any(a.name.lower() == normalized_name for a in later_plan.areas):
                                reappears = True
                                break
                        
                        improvement_trajectory[normalized_name] = not reappears
        
        return {
            "total_plans_analyzed": len(all_plans),
            "time_span": (
                datetime.fromisoformat(all_plans[-1].created_at.split("+")[0]) - 
                datetime.fromisoformat(all_plans[0].created_at.split("+")[0])
            ).days,
            "recurring_themes": recurring_themes,
            "persistent_issues": [theme for theme, improved in improvement_trajectory.items() if not improved],
            "successfully_improved": [theme for theme, improved in improvement_trajectory.items() if improved]
        }
    
    def apply_learning_from_historical_plans(self) -> bool:
        """
        Apply learnings from historical plans to the current plan.
        
        Returns:
            True if successful, False if not enough history or no current plan
        """
        if not self.historical_plans or not self.current_plan:
            self.logger.info("Not enough history or no current plan to apply learnings")
            return False
        
        # Get patterns from historical plans
        patterns = self.identify_improvement_patterns()
        
        # Check for persistent issues that should be prioritized
        if "persistent_issues" in patterns:
            for issue in patterns["persistent_issues"]:
                # Check if this issue is already in the current plan
                if not any(area.name.lower() == issue.lower() for area in self.current_plan.areas):
                    # Add as a high-priority area
                    self.add_improvement_area(
                        name=issue.title(),  # Capitalize the first letter
                        description=f"Persistent issue identified across multiple improvement plans: {issue}",
                        priority=0.9,  # High priority for persistent issues
                        supporting_feedback=[]
                    )
                    self.logger.info(f"Added persistent issue as improvement area: {issue}")
        
        # Learn from successful improvements
        if "successfully_improved" in patterns and "recurring_themes" in patterns:
            # Find effective strategies for successful improvements
            effective_strategies = []
            
            # Look at successful themes
            for success_theme in patterns.get("successfully_improved", []):
                # Find the historical plan where this was successfully addressed
                for plan in self.historical_plans:
                    for area in plan.areas:
                        if area.name.lower() == success_theme.lower() and area.status in ["implemented", "verified"]:
                            # Extract successful strategies
                            for strategy in area.improvement_strategies:
                                if strategy.get("status") in ["implemented", "verified"]:
                                    effective_strategies.append({
                                        "theme": success_theme,
                                        "strategy_type": strategy.get("type"),
                                        "description": strategy.get("description"),
                                        "implementation_steps": strategy.get("implementation_steps", [])
                                    })
            
            # Apply successful strategies to similar areas in current plan
            if effective_strategies:
                for area in self.current_plan.areas:
                    # Find similar themes
                    similar_themes = [s for s in effective_strategies 
                                     if any(word in area.name.lower() for word in s["theme"].split())]
                    
                    if similar_themes:
                        # Apply the strategy
                        strategy = similar_themes[0]
                        self.add_improvement_strategy(
                            area_id=area.id,
                            description=f"Based on past success: {strategy['description']}",
                            strategy_type=strategy.get("strategy_type", "learned"),
                            implementation_steps=strategy.get("implementation_steps", []),
                            estimated_impact=0.8  # High estimated impact based on past success
                        )
                        self.logger.info(f"Applied learned strategy to area: {area.name}")
        
        self.logger.info("Applied learnings from historical plans")
        return True
    
    def complete_plan(self) -> bool:
        """
        Mark the current plan as completed and move it to historical plans.
        
        Returns:
            True if successful, False if no current plan
        """
        if not self.current_plan:
            self.logger.warning("No current improvement plan to complete")
            return False
        
        self.current_plan.status = "completed"
        self.current_plan.updated_at = datetime.now().isoformat()
        
        # Move to historical plans
        self.historical_plans.append(self.current_plan)
        self.current_plan = None
        
        self.logger.info("Completed current improvement plan and moved to history")
        self._save_improvement_data()
        return True
