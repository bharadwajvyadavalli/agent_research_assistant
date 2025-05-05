"""
Tree of Thoughts (ToT) reasoning implementation.

This module provides a more advanced reasoning approach that allows exploring
multiple potential reasoning paths simultaneously for complex problem solving.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class TreeOfThoughtsReasoner:
    """
    Tree of Thoughts (ToT) reasoning implementation.
    
    ToT extends Chain of Thought by exploring multiple potential reasoning paths
    simultaneously. It enables backtracking if a path proves unpromising and is
    especially useful for tasks with uncertainty or where multiple approaches
    might be viable.
    """
    
    def __init__(self, 
                model: str = "gpt-4",
                temperature: float = 0.4,  # Higher temperature for more diverse paths
                max_tokens: int = 2000,
                max_depth: int = 3,
                max_branches: int = 3,
                evaluation_temperature: float = 0.1):  # Lower temp for evaluation
        """
        Initialize the Tree of Thoughts reasoner.
        
        Args:
            model: The LLM model to use
            temperature: Temperature for generating diverse thought branches
            max_tokens: Maximum tokens for generation
            max_depth: Maximum depth of the reasoning tree
            max_branches: Maximum branching factor at each node
            evaluation_temperature: Temperature for path evaluation (lower for consistency)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.evaluation_temperature = evaluation_temperature
        
        logger.info("Tree of Thoughts reasoner initialized")
    
    @retry(
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def reason(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Apply Tree of Thoughts reasoning to a query.
        
        Args:
            query: The question or problem to reason about
            context: Additional context information
            
        Returns:
            A dictionary containing:
            - reasoning: The multi-path reasoning process
            - conclusion: The final answer or conclusion
            - best_path: The most promising reasoning path
        """
        context = context or {}
        
        # 1. Generate initial thought branches
        initial_branches = await self._generate_thought_branches(query, context)
        
        # 2. Explore and evaluate the reasoning tree
        best_path, all_paths = await self._explore_reasoning_tree(query, initial_branches, context)
        
        # 3. Format the result
        reasoning_text = self._format_reasoning_tree(all_paths)
        conclusion = best_path[-1]["content"] if best_path else "Failed to reach a conclusion."
        
        return {
            "reasoning": reasoning_text,
            "conclusion": conclusion,
            "best_path": [node["content"] for node in best_path],
            "all_paths": all_paths
        }
    
    async def _generate_thought_branches(self, query: str, context: Dict) -> List[Dict]:
        """
        Generate initial thought branches to explore different approaches.
        
        Args:
            query: The problem or question
            context: Additional context
            
        Returns:
            List of initial thought branches to explore
        """
        prompt = self._create_branching_prompt(query, context)
        
        try:
            messages = [
                {"role": "system", "content": self._get_branching_system_message()},
                {"role": "user", "content": prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            branching_text = response.choices[0].message.content
            
            # Parse branches
            branches = self._parse_branches(branching_text)
            
            # If parsing fails or returns too few branches, try again with a simpler prompt
            if len(branches) < 2:
                logger.warning("Failed to parse sufficient branches, trying simpler prompt")
                simple_prompt = f"""
                Please provide {self.max_branches} different approaches to solve this problem:
                
                Problem: {query}
                
                Format your response as:
                Approach 1: [First approach]
                Approach 2: [Second approach]
                ...
                """
                
                simple_response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": simple_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                branching_text = simple_response.choices[0].message.content
                branches = self._parse_branches(branching_text)
            
            return [{"id": i, "content": branch, "depth": 0, "parent": None} 
                   for i, branch in enumerate(branches[:self.max_branches])]
            
        except Exception as e:
            logger.error(f"Error generating thought branches: {str(e)}")
            # Return a single default branch
            return [{"id": 0, "content": f"Let's tackle this problem methodically. {query}", "depth": 0, "parent": None}]
    
    async def _explore_reasoning_tree(
        self, 
        query: str, 
        initial_branches: List[Dict], 
        context: Dict
    ) -> Tuple[List[Dict], Dict]:
        """
        Explore the reasoning tree by evaluating and expanding promising paths.
        
        Args:
            query: The original query
            initial_branches: Initial thought branches
            context: Additional context
            
        Returns:
            Tuple of (best path nodes, all paths in the tree)
        """
        # Track all paths in the tree
        all_paths = {
            "root": {"content": query, "children": [b["id"] for b in initial_branches]},
        }
        for branch in initial_branches:
            all_paths[branch["id"]] = branch
        
        # Active paths to explore (we'll use a breadth-first approach)
        active_paths = initial_branches.copy()
        
        # Node ID counter
        next_node_id = len(initial_branches)
        
        # Evaluate initial branches
        evaluations = await self._evaluate_branches(query, initial_branches, context)
        
        for i, branch in enumerate(initial_branches):
            branch["evaluation"] = evaluations[i]
            
        # Sort branches by evaluation
        active_paths.sort(key=lambda x: x["evaluation"], reverse=True)
        
        # Keep only the top branches to explore further
        active_paths = active_paths[:min(len(active_paths), 2)]
        
        # Explore the tree up to max_depth
        current_depth = 1
        while current_depth < self.max_depth and active_paths:
            next_active_paths = []
            
            for path in active_paths:
                # Skip paths with low evaluation scores
                if path.get("evaluation", 0) < 0.5 and len(next_active_paths) > 0:
                    continue
                
                # Expand this path
                try:
                    expanded_nodes = await self._expand_reasoning_path(
                        query, path, context, current_depth
                    )
                    
                    if expanded_nodes:
                        # Update node IDs
                        for node in expanded_nodes:
                            node["id"] = next_node_id
                            node["parent"] = path["id"]
                            node["depth"] = current_depth
                            all_paths[next_node_id] = node
                            next_node_id += 1
                        
                        # Update parent's children
                        path["children"] = [node["id"] for node in expanded_nodes]
                        
                        # Evaluate expanded nodes
                        evaluations = await self._evaluate_branches(
                            query, expanded_nodes, context, path["content"]
                        )
                        
                        for i, node in enumerate(expanded_nodes):
                            node["evaluation"] = evaluations[i]
                        
                        # Add the most promising expanded nodes to the next active set
                        expanded_nodes.sort(key=lambda x: x["evaluation"], reverse=True)
                        next_active_paths.extend(expanded_nodes[:1])  # Only follow the best path from each branch
                        
                except Exception as e:
                    logger.error(f"Error expanding path: {str(e)}")
                    path["error"] = str(e)
            
            active_paths = next_active_paths
            current_depth += 1
        
        # Find the best complete path
        best_path = self._find_best_path(all_paths)
        
        return best_path, all_paths
    
    async def _expand_reasoning_path(
        self, 
        query: str, 
        current_node: Dict, 
        context: Dict, 
        depth: int
    ) -> List[Dict]:
        """
        Expand a reasoning path by generating next steps.
        
        Args:
            query: The original query
            current_node: The current reasoning node to expand
            context: Additional context
            depth: Current depth in the reasoning tree
            
        Returns:
            List of expanded reasoning nodes
        """
        # Create a prompt for continuing the reasoning
        continuation_prompt = f"""
        I'm solving this problem: {query}
        
        So far, I've thought:
        {current_node["content"]}
        
        Now I'll continue from here and explore the next step in this reasoning path.
        Let me think about {self.max_branches} possible ways to proceed from here:
        """
        
        try:
            messages = [
                {"role": "system", "content": self._get_continuation_system_message()},
                {"role": "user", "content": continuation_prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            continuation_text = response.choices[0].message.content
            
            # Parse continuation options
            continuations = self._parse_continuations(continuation_text)
            
            # If we're at max_depth-1, generate conclusions instead of more branches
            if depth == self.max_depth - 1:
                conclusion_nodes = await self._generate_conclusions(
                    query, current_node["content"], context
                )
                return conclusion_nodes
            
            return [{"content": cont} for cont in continuations[:self.max_branches]]
            
        except Exception as e:
            logger.error(f"Error expanding reasoning path: {str(e)}")
            return []
    
    async def _generate_conclusions(
        self, 
        query: str, 
        reasoning_so_far: str, 
        context: Dict
    ) -> List[Dict]:
        """
        Generate conclusion nodes for a reasoning path.
        
        Args:
            query: The original query
            reasoning_so_far: The reasoning path so far
            context: Additional context
            
        Returns:
            List of conclusion nodes
        """
        conclusion_prompt = f"""
        I've been solving this problem: {query}
        
        My reasoning so far:
        {reasoning_so_far}
        
        Based on this line of reasoning, I'll now formulate a final conclusion that directly answers the original question.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides clear conclusions based on reasoning."},
                {"role": "user", "content": conclusion_prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            conclusion_text = response.choices[0].message.content
            
            # Just return the conclusion as a single node
            return [{"content": f"Conclusion: {conclusion_text}"}]
            
        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            return [{"content": "Failed to generate a conclusion due to an error."}]
    
    async def _evaluate_branches(
        self, 
        query: str, 
        branches: List[Dict], 
        context: Dict,
        previous_reasoning: str = ""
    ) -> List[float]:
        """
        Evaluate the quality and promise of reasoning branches.
        
        Args:
            query: The original query
            branches: List of reasoning branches to evaluate
            context: Additional context
            previous_reasoning: Previous reasoning steps (if any)
            
        Returns:
            List of evaluation scores (0.0 to 1.0) for each branch
        """
        if not branches:
            return []
        
        evaluation_prompt = f"""
        I need to evaluate which of these reasoning paths is most promising for solving this problem:
        
        Problem: {query}
        
        {previous_reasoning if previous_reasoning else ""}
        
        Reasoning paths:
        """
        
        for i, branch in enumerate(branches):
            evaluation_prompt += f"\nPath {i+1}:\n{branch['content']}\n"
        
        evaluation_prompt += """
        For each path, evaluate:
        1. Relevance to the problem
        2. Logical coherence
        3. Potential to lead to a correct solution
        4. Insightfulness
        
        Assign a score from 0.0 to 1.0 for each path, where 1.0 means extremely promising.
        
        Return your evaluation as a JSON array of scores in the format:
        [score_for_path_1, score_for_path_2, ...]
        """
        
        try:
            messages = [
                {"role": "system", "content": self._get_evaluation_system_message()},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=self.evaluation_temperature,
                max_tokens=self.max_tokens
            )
            
            evaluation_text = response.choices[0].message.content
            
            # Parse JSON scores
            scores = self._parse_evaluation_scores(evaluation_text, len(branches))
            
            return scores
            
        except Exception as e:
            logger.error(f"Error evaluating branches: {str(e)}")
            # Return default scores
            return [0.7] * len(branches)
    
    def _parse_branches(self, branching_text: str) -> List[str]:
        """
        Parse the response text to extract distinct thought branches.
        
        Args:
            branching_text: Raw text containing the branches
            
        Returns:
            List of distinct thought branches
        """
        import re
        
        # Try to find branches labeled with "Approach", "Path", "Branch", etc.
        patterns = [
            r'(?:Approach|Path|Branch|Option)\s+\d+:?\s+(.*?)(?=(?:Approach|Path|Branch|Option)\s+\d+:|$)',
            r'(?:\d+\.\s+)([^.]+.*?)(?=\d+\.\s+|$)',
            r'(?:•\s+)([^•]+.*?)(?=•\s+|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, branching_text, re.DOTALL)
            branches = [match.group(1).strip() for match in matches]
            
            if len(branches) >= 2:
                return branches
        
        # If no patterns match, split by double newlines
        paragraphs = branching_text.split('\n\n')
        if len(paragraphs) >= 2:
            return [p.strip() for p in paragraphs if p.strip()]
        
        # Last resort: split by single newlines
        lines = branching_text.split('\n')
        return [line.strip() for line in lines if line.strip()]
    
    def _parse_continuations(self, continuation_text: str) -> List[str]:
        """Parse the continuation options from the response text."""
        # Same logic as _parse_branches
        return self._parse_branches(continuation_text)
    
    def _parse_evaluation_scores(self, evaluation_text: str, num_branches: int) -> List[float]:
        """
        Parse evaluation scores from the response text.
        
        Args:
            evaluation_text: Raw text containing the evaluation
            num_branches: Expected number of branches
            
        Returns:
            List of evaluation scores
        """
        import re
        import json
        
        # Try to find JSON array in the text
        json_pattern = r'\[\s*(?:[\d.]+\s*,\s*)*[\d.]+\s*\]'
        json_match = re.search(json_pattern, evaluation_text)
        
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
                if isinstance(scores, list) and len(scores) == num_branches:
                    return scores
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON scores")
        
        # Try to extract scores with regex
        score_pattern = r'(?:Path|Option|Approach)\s+(\d+).*?score.*?(\d+\.\d+|\d+)'
        score_matches = re.finditer(score_pattern, evaluation_text, re.IGNORECASE)
        
        scores_dict = {}
        for match in score_matches:
            path_num = int(match.group(1))
            score = float(match.group(2))
            scores_dict[path_num] = score
        
        if scores_dict:
            return [scores_dict.get(i+1, 0.5) for i in range(num_branches)]
        
        # Default: assign decreasing scores
        base_score = 0.7
        decrement = 0.05
        return [max(0.1, base_score - i * decrement) for i in range(num_branches)]
    
    def _find_best_path(self, all_paths: Dict) -> List[Dict]:
        """
        Find the best complete path through the reasoning tree.
        
        Args:
            all_paths: Dictionary of all nodes in the tree
            
        Returns:
            List of nodes representing the best path
        """
        # Find leaf nodes
        leaf_nodes = []
        for node_id, node in all_paths.items():
            if node_id != "root" and "children" not in node:
                leaf_nodes.append(node)
        
        # Sort leaf nodes by evaluation score
        leaf_nodes.sort(key=lambda x: x.get("evaluation", 0), reverse=True)
        
        if not leaf_nodes:
            return []
        
        # Trace back from the best leaf to the root
        best_path = []
        current = leaf_nodes[0]
        
        while current:
            best_path.append(current)
            if "parent" in current and current["parent"] is not None:
                current = all_paths.get(current["parent"])
            else:
                current = None
        
        # Reverse to get path from root to leaf
        best_path.reverse()
        
        return best_path
    
    def _format_reasoning_tree(self, all_paths: Dict) -> str:
        """
        Format the reasoning tree into a readable text format.
        
        Args:
            all_paths: Dictionary of all nodes in the tree
            
        Returns:
            Formatted reasoning text
        """
        # Find the best path
        best_path = self._find_best_path(all_paths)
        
        # Format as text
        reasoning_text = "Tree of Thoughts Reasoning:\n\n"
        
        # Add the selected path
        reasoning_text += "Selected Reasoning Path:\n"
        for i, node in enumerate(best_path):
            prefix = "  " * i
            reasoning_text += f"{prefix}Step {i+1}: {node['content']}\n\n"
        
        # Add the conclusion
        if best_path:
            reasoning_text += f"Conclusion: {best_path[-1]['content']}\n"
        else:
            reasoning_text += "Could not determine a clear conclusion.\n"
        
        return reasoning_text
    
    def _create_branching_prompt(self, query: str, context: Dict) -> str:
        """
        Create a prompt for generating initial thought branches.
        
        Args:
            query: The problem or question
            context: Additional context
            
        Returns:
            Formatted prompt string
        """
        # Format context if provided
        context_str = ""
        if context:
            context_str = "Context Information:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
            context_str += "\n"
        
        prompt = f"""
        {context_str}
        Problem: {query}
        
        I'll approach this problem using Tree of Thoughts reasoning, where I explore multiple different solution approaches.
        
        Let me think of {self.max_branches} distinct approaches to solving this problem:
        
        Approach 1: 
        """
        
        return prompt
    
    def _get_branching_system_message(self) -> str:
        """Get the system message for generating thought branches."""
        return """You are an expert problem-solving assistant that uses Tree of Thoughts reasoning.
        
        When approaching a problem, generate multiple distinct solution approaches that take different perspectives.
        Each approach should be well-reasoned, creative, and tackle the problem from a unique angle.
        
        Make sure your approaches are:
        1. Diverse - explore different methodologies
        2. Specific - not vague or generic
        3. Relevant - directly applicable to the problem
        4. Promising - have potential to lead to a solution
        """
    
    def _get_continuation_system_message(self) -> str:
        """Get the system message for continuing reasoning paths."""
        return """You are an expert problem-solving assistant that continues reasoning along a specific path.
        
        Given a problem and the reasoning so far, your task is to:
        1. Understand the direction of thought developed so far
        2. Generate multiple ways to continue this specific line of reasoning
        3. Ensure each continuation is coherent with the previous thinking
        4. Make meaningful progress toward solving the problem
        
        Maintain the same perspective and approach as the initial reasoning, but add new insights and developments.
        """
    
    def _get_evaluation_system_message(self) -> str:
        """Get the system message for evaluating reasoning paths."""
        return """You are an expert evaluator of reasoning approaches.
        
        Your task is to objectively assess different reasoning paths and determine which is most promising.
        Evaluate each path based on:
        
        1. Relevance - how directly it addresses the core problem
        2. Logical coherence - the soundness of the reasoning
        3. Progress - how much closer it gets to a solution
        4. Insightfulness - the depth and originality of thinking
        
        Assign numerical scores from 0.0 to 1.0 for each path, where higher scores indicate more promising approaches.
        Be fair and consistent in your evaluations.
        """
