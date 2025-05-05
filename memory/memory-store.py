"""
Memory Store implementation for agent memory management.

This module provides storage and retrieval mechanisms for different types of
memories including episodic, semantic, and procedural memory components.
"""

import logging
import os
import json
import uuid
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import aiofiles
from config import MEMORY_CONFIG

try:
    # Try to import vector store libraries
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logging.warning("Vector store dependencies not available. Install with: pip install numpy sentence-transformers")

logger = logging.getLogger(__name__)

class MemoryStore:
    """
    Memory system for storing and retrieving different types of agent memories.
    
    The MemoryStore implements a hybrid memory system combining vector-based 
    embedding storage for semantic similarity retrieval with structured storage
    for episodic memories and procedural knowledge.
    """
    
    def __init__(self, 
                vector_db_path: str = None,
                max_episodic_items: int = None,
                semantic_refresh_interval: int = None):
        """
        Initialize the memory store with configuration.
        
        Args:
            vector_db_path: Path to store vector embeddings
            max_episodic_items: Maximum number of episodic memories to retain
            semantic_refresh_interval: Hours between semantic memory consolidation
        """
        self.vector_db_path = vector_db_path or MEMORY_CONFIG.get("vector_db_path", "data/vector_store")
        self.max_episodic_items = max_episodic_items or MEMORY_CONFIG.get("max_episodic_memory_items", 100)
        self.semantic_refresh_interval = semantic_refresh_interval or MEMORY_CONFIG.get("semantic_memory_refresh_interval", 24)
        
        # Memory storage
        self.episodic_memory = []
        self.semantic_memory = {}
        self.procedural_memory = {}
        
        # Vector store components
        self.embedding_model = None
        self.vector_store = None
        self.vector_store_initialized = False
        
        # Meta information
        self.last_semantic_refresh = time.time()
        
        logger.info("Memory store initialized")
    
    async def initialize(self):
        """Initialize memory storage and load existing memories."""
        # Create directories if they don't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        os.makedirs(os.path.join(self.vector_db_path, "episodic"), exist_ok=True)
        os.makedirs(os.path.join(self.vector_db_path, "semantic"), exist_ok=True)
        os.makedirs(os.path.join(self.vector_db_path, "procedural"), exist_ok=True)
        
        # Initialize vector store if available
        if VECTOR_STORE_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Load existing vector store if it exists
                vector_store_path = os.path.join(self.vector_db_path, "vector_store.npz")
                if os.path.exists(vector_store_path):
                    data = np.load(vector_store_path, allow_pickle=True)
                    self.vector_store = {
                        "texts": data["texts"].tolist(),
                        "embeddings": data["embeddings"],
                        "metadata": data["metadata"].tolist()
                    }
                else:
                    self.vector_store = {
                        "texts": [],
                        "embeddings": np.array([]).reshape(0, 384),  # Embedding dimension
                        "metadata": []
                    }
                
                self.vector_store_initialized = True
                logger.info("Vector store initialized")
                
            except Exception as e:
                logger.error(f"Error initializing vector store: {str(e)}")
        
        # Load episodic memories
        try:
            episodic_path = os.path.join(self.vector_db_path, "episodic", "memories.json")
            if os.path.exists(episodic_path):
                async with aiofiles.open(episodic_path, 'r') as f:
                    content = await f.read()
                    self.episodic_memory = json.loads(content)
                    logger.info(f"Loaded {len(self.episodic_memory)} episodic memories")
        except Exception as e:
            logger.error(f"Error loading episodic memories: {str(e)}")
        
        # Load semantic memories
        try:
            semantic_path = os.path.join(self.vector_db_path, "semantic", "memories.json")
            if os.path.exists(semantic_path):
                async with aiofiles.open(semantic_path, 'r') as f:
                    content = await f.read()
                    self.semantic_memory = json.loads(content)
                    logger.info(f"Loaded {len(self.semantic_memory)} semantic memories")
        except Exception as e:
            logger.error(f"Error loading semantic memories: {str(e)}")
        
        # Load procedural memories
        try:
            procedural_path = os.path.join(self.vector_db_path, "procedural", "memories.json")
            if os.path.exists(procedural_path):
                async with aiofiles.open(procedural_path, 'r') as f:
                    content = await f.read()
                    self.procedural_memory = json.loads(content)
                    logger.info(f"Loaded {len(self.procedural_memory)} procedural memories")
        except Exception as e:
            logger.error(f"Error loading procedural memories: {str(e)}")
    
    async def store_interaction(self, input_text: str, output_text: str, metadata: Dict = None) -> str:
        """
        Store an interaction in episodic memory.
        
        Args:
            input_text: User input or query
            output_text: System output or response
            metadata: Additional metadata for context
            
        Returns:
            Memory ID of the stored interaction
        """
        memory_id = f"interaction_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        # Create memory entry
        memory = {
            "id": memory_id,
            "type": "interaction",
            "timestamp": timestamp,
            "input": input_text,
            "output": output_text,
            "metadata": metadata or {}
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Manage memory size
        if len(self.episodic_memory) > self.max_episodic_items:
            self.episodic_memory = self.episodic_memory[-self.max_episodic_items:]
        
        # Save to disk
        await self._save_episodic_memories()
        
        # Add to vector store for retrieval
        if self.vector_store_initialized:
            await self._add_to_vector_store(
                f"INPUT: {input_text}\nOUTPUT: {output_text}",
                {"id": memory_id, "type": "interaction", "timestamp": timestamp}
            )
        
        # Consider semantic memory consolidation
        elapsed_hours = (time.time() - self.last_semantic_refresh) / 3600
        if elapsed_hours >= self.semantic_refresh_interval:
            asyncio.create_task(self._consolidate_semantic_memory())
        
        return memory_id
    
    async def store_query(self, query: str, query_id: str, context: Dict = None) -> str:
        """
        Store a user query in episodic memory.
        
        Args:
            query: The user query
            query_id: Unique identifier for the query
            context: Additional context information
            
        Returns:
            Memory ID of the stored query
        """
        timestamp = datetime.now().isoformat()
        
        # Create memory entry
        memory = {
            "id": query_id,
            "type": "query",
            "timestamp": timestamp,
            "query": query,
            "context": context or {}
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Manage memory size
        if len(self.episodic_memory) > self.max_episodic_items:
            self.episodic_memory = self.episodic_memory[-self.max_episodic_items:]
        
        # Save to disk
        await self._save_episodic_memories()
        
        # Add to vector store for retrieval
        if self.vector_store_initialized:
            await self._add_to_vector_store(
                f"QUERY: {query}",
                {"id": query_id, "type": "query", "timestamp": timestamp}
            )
        
        return query_id
    
    async def store_plan(self, plan: Dict) -> str:
        """
        Store a plan in episodic memory.
        
        Args:
            plan: The plan to store
            
        Returns:
            Memory ID of the stored plan
        """
        # Extract plan_id or generate a new one
        plan_id = plan.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}")
        timestamp = datetime.now().isoformat()
        
        # Create memory entry
        memory = {
            "id": plan_id,
            "type": "plan",
            "timestamp": timestamp,
            "plan": plan
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Manage memory size
        if len(self.episodic_memory) > self.max_episodic_items:
            self.episodic_memory = self.episodic_memory[-self.max_episodic_items:]
        
        # Save to disk
        await self._save_episodic_memories()
        
        # Add to vector store for retrieval
        if self.vector_store_initialized:
            plan_text = json.dumps(plan.get("steps", []))
            await self._add_to_vector_store(
                f"PLAN: {plan_text}",
                {"id": plan_id, "type": "plan", "timestamp": timestamp}
            )
        
        return plan_id
    
    async def store_execution(self, execution_record: Dict) -> str:
        """
        Store an execution record in episodic memory.
        
        Args:
            execution_record: Record of tool execution
            
        Returns:
            Memory ID of the stored execution
        """
        step_id = execution_record.get("step_id", f"execution_{uuid.uuid4().hex[:8]}")
        timestamp = datetime.now().isoformat()
        
        # Create memory entry
        memory = {
            "id": step_id,
            "type": "execution",
            "timestamp": timestamp,
            "execution": execution_record
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Manage memory size
        if len(self.episodic_memory) > self.max_episodic_items:
            self.episodic_memory = self.episodic_memory[-self.max_episodic_items:]
        
        # Save to disk
        await self._save_episodic_memories()
        
        # Add to vector store for retrieval
        if self.vector_store_initialized:
            desc = execution_record.get("description", "")
            result = execution_record.get("result", "")
            await self._add_to_vector_store(
                f"EXECUTION: {desc}\nRESULT: {result}",
                {"id": step_id, "type": "execution", "timestamp": timestamp}
            )
        
        # Also store tool usage in procedural memory
        tools_used = []
        for tool_execution in execution_record.get("tool_executions", []):
            tool_name = tool_execution.get("tool", "unknown_tool")
            
            # Update procedural memory
            if tool_name not in self.procedural_memory:
                self.procedural_memory[tool_name] = {
                    "usage_count": 1,
                    "last_used": timestamp,
                    "examples": [{"parameters": tool_execution.get("parameters", {})}]
                }
            else:
                self.procedural_memory[tool_name]["usage_count"] += 1
                self.procedural_memory[tool_name]["last_used"] = timestamp
                
                # Add a new example if it's different
                examples = self.procedural_memory[tool_name]["examples"]
                current_params = tool_execution.get("parameters", {})
                
                # Only store up to 5 examples
                if len(examples) < 5:
                    # Check if this is a unique example
                    is_unique = True
                    for example in examples:
                        if example["parameters"] == current_params:
                            is_unique = False
                            break
                    
                    if is_unique:
                        examples.append({"parameters": current_params})
            
            tools_used.append(tool_name)
        
        # Save procedural memory
        if tools_used:
            await self._save_procedural_memories()
        
        return step_id
    
    async def store_evaluation(self, evaluation: Dict) -> str:
        """
        Store an evaluation record in episodic memory.
        
        Args:
            evaluation: Evaluation results
            
        Returns:
            Memory ID of the stored evaluation
        """
        eval_id = f"eval_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        # Create memory entry
        memory = {
            "id": eval_id,
            "type": "evaluation",
            "timestamp": timestamp,
            "evaluation": evaluation
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Manage memory size
        if len(self.episodic_memory) > self.max_episodic_items:
            self.episodic_memory = self.episodic_memory[-self.max_episodic_items:]
        
        # Save to disk
        await self._save_episodic_memories()
        
        # Add to vector store for retrieval
        if self.vector_store_initialized:
            feedback = evaluation.get("feedback", "")
            suggestions = "\n".join(evaluation.get("suggestions", []))
            await self._add_to_vector_store(
                f"EVALUATION FEEDBACK: {feedback}\nSUGGESTIONS: {suggestions}",
                {"id": eval_id, "type": "evaluation", "timestamp": timestamp}
            )
        
        return eval_id
    
    async def store_results(self, query_id: str, query: str, plan: Dict, 
                         results: List[Dict], final_answer: str, evaluation: Dict) -> str:
        """
        Store complete results from a query processing session.
        
        Args:
            query_id: ID of the original query
            query: The user query
            plan: The execution plan
            results: Results from step executions
            final_answer: The final answer
            evaluation: Evaluation of the final answer
            
        Returns:
            Memory ID of the stored results
        """
        result_id = f"result_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        # Create memory entry
        memory = {
            "id": result_id,
            "type": "result",
            "timestamp": timestamp,
            "query_id": query_id,
            "query": query,
            "final_answer": final_answer,
            "evaluation": evaluation,
            "plan_id": plan.get("plan_id", ""),
            "step_results": [result.get("step_id", "") for result in results]
        }
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Manage memory size
        if len(self.episodic_memory) > self.max_episodic_items:
            self.episodic_memory = self.episodic_memory[-self.max_episodic_items:]
        
        # Save to disk
        await self._save_episodic_memories()
        
        # Add to vector store for retrieval
        if self.vector_store_initialized:
            await self._add_to_vector_store(
                f"QUERY: {query}\nANSWER: {final_answer}",
                {"id": result_id, "type": "result", "timestamp": timestamp, 
                 "query_id": query_id, "plan_id": plan.get("plan_id", "")}
            )
            
            # Add or update semantic memory if answer was well-evaluated
            if evaluation.get("overall_score", 0) >= 0.8:
                # Extract key concepts
                await self._add_semantic_memory(query, final_answer)
        
        return result_id
    
    async def retrieve_by_similarity(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve memories based on semantic similarity to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of similar memories
        """
        if not self.vector_store_initialized:
            logger.warning("Vector store not initialized, falling back to keyword search")
            return await self.retrieve_by_keyword(query, limit)
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate similarity scores
            similarities = np.dot(self.vector_store["embeddings"], query_embedding)
            
            # Get top results
            top_indices = np.argsort(similarities)[-limit:][::-1]
            
            results = []
            for idx in top_indices:
                # Get the corresponding memory
                metadata = self.vector_store["metadata"][idx]
                memory_id = metadata.get("id")
                
                # Find the full memory in episodic memory
                memory = None
                for m in self.episodic_memory:
                    if m["id"] == memory_id:
                        memory = m
                        break
                
                if memory:
                    results.append({
                        "memory": memory,
                        "similarity": float(similarities[idx]),
                        "text": self.vector_store["texts"][idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving by similarity: {str(e)}")
            return []
    
    async def retrieve_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve memories based on keyword matching.
        
        Args:
            keyword: The search keyword
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        results = []
        
        # Convert keyword to lowercase for case-insensitive matching
        keyword_lower = keyword.lower()
        
        for memory in self.episodic_memory:
            memory_text = ""
            
            # Extract text based on memory type
            if memory["type"] == "interaction":
                memory_text = f"{memory['input']} {memory['output']}"
            elif memory["type"] == "query":
                memory_text = memory["query"]
            elif memory["type"] == "plan":
                memory_text = str(memory["plan"])
            elif memory["type"] == "execution":
                memory_text = f"{memory['execution'].get('description', '')} {memory['execution'].get('result', '')}"
            elif memory["type"] == "evaluation":
                memory_text = f"{memory['evaluation'].get('feedback', '')}"
            elif memory["type"] == "result":
                memory_text = f"{memory['query']} {memory['final_answer']}"
            
            # Check if keyword is in memory text
            if keyword_lower in memory_text.lower():
                results.append({
                    "memory": memory,
                    "text": memory_text[:200] + "..." if len(memory_text) > 200 else memory_text
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def retrieve_semantic_concept(self, concept: str) -> Optional[Dict]:
        """
        Retrieve a semantic memory concept.
        
        Args:
            concept: The concept to retrieve
            
        Returns:
            The semantic memory for the concept, if it exists
        """
        # Try exact match
        if concept in self.semantic_memory:
            return self.semantic_memory[concept]
        
        # Try case-insensitive match
        concept_lower = concept.lower()
        for key, value in self.semantic_memory.items():
            if key.lower() == concept_lower:
                return value
        
        # If vector store is available, try semantic search
        if self.vector_store_initialized:
            try:
                # Check if any concepts are semantically similar
                concept_embedding = self.embedding_model.encode([concept])[0]
                
                # Create embeddings for all concepts
                concept_texts = list(self.semantic_memory.keys())
                concept_embeddings = self.embedding_model.encode(concept_texts)
                
                # Calculate similarities
                similarities = np.dot(concept_embeddings, concept_embedding)
                
                # If any similarity is above threshold, return the concept
                threshold = 0.8
                max_idx = np.argmax(similarities)
                
                if similarities[max_idx] >= threshold:
                    matched_concept = concept_texts[max_idx]
                    return self.semantic_memory[matched_concept]
                
            except Exception as e:
                logger.error(f"Error in semantic concept retrieval: {str(e)}")
        
        return None
    
    async def retrieve_procedural_knowledge(self, tool_name: str) -> Optional[Dict]:
        """
        Retrieve procedural knowledge for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Procedural knowledge for the tool, if available
        """
        if tool_name in self.procedural_memory:
            return self.procedural_memory[tool_name]
        
        # Try case-insensitive match
        tool_name_lower = tool_name.lower()
        for key, value in self.procedural_memory.items():
            if key.lower() == tool_name_lower:
                return value
        
        return None
    
    async def retrieve_related_memories(self, memory_id: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve memories related to a specific memory.
        
        Args:
            memory_id: ID of the memory to find related memories for
            limit: Maximum number of results to return
            
        Returns:
            List of related memories
        """
        # Find the target memory
        target_memory = None
        for memory in self.episodic_memory:
            if memory["id"] == memory_id:
                target_memory = memory
                break
        
        if not target_memory:
            return []
        
        # Extract search text based on memory type
        search_text = ""
        if target_memory["type"] == "interaction":
            search_text = target_memory["input"]
        elif target_memory["type"] == "query":
            search_text = target_memory["query"]
        elif target_memory["type"] == "plan":
            # Extract goal from plan
            plan_context = target_memory["plan"].get("context", {})
            search_text = plan_context.get("goal", "")
        elif target_memory["type"] == "result":
            search_text = target_memory["query"]
        
        # If we have meaningful search text, retrieve by similarity
        if search_text:
            results = await self.retrieve_by_similarity(search_text, limit)
            
            # Filter out the target memory itself
            results = [r for r in results if r["memory"]["id"] != memory_id]
            
            return results[:limit]
        
        # Fallback to recent memories of the same type
        same_type_memories = [m for m in self.episodic_memory if m["type"] == target_memory["type"]]
        same_type_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        
        results = []
        for memory in same_type_memories:
            if memory["id"] != memory_id:
                results.append({"memory": memory})
                if len(results) >= limit:
                    break
        
        return results
    
    async def _add_to_vector_store(self, text: str, metadata: Dict):
        """
        Add a text entry to the vector store.
        
        Args:
            text: The text to embed
            metadata: Metadata for the entry
        """
        if not self.vector_store_initialized:
            return
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([text])[0]
            
            # Add to vector store
            self.vector_store["texts"].append(text)
            self.vector_store["metadata"].append(metadata)
            
            # Append to embeddings array
            if len(self.vector_store["embeddings"]) == 0:
                self.vector_store["embeddings"] = np.array([embedding])
            else:
                self.vector_store["embeddings"] = np.vstack([self.vector_store["embeddings"], embedding])
            
            # Save vector store periodically
            if len(self.vector_store["texts"]) % 10 == 0:
                await self._save_vector_store()
                
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
    
    async def _add_semantic_memory(self, query: str, answer: str) -> bool:
        """
        Add or update a semantic memory from a query-answer pair.
        
        Args:
            query: The user query
            answer: The system answer
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Extract key concepts from the query
            keywords = self._extract_keywords(query)
            
            timestamp = datetime.now().isoformat()
            
            # Add or update semantic memory for each keyword
            for keyword in keywords:
                if keyword in self.semantic_memory:
                    # Update existing concept
                    self.semantic_memory[keyword]["occurrences"] += 1
                    self.semantic_memory[keyword]["last_updated"] = timestamp
                    
                    # If this query-answer adds new information, append it
                    current_info = self.semantic_memory[keyword]["information"]
                    if len(current_info) < 1000 and answer not in current_info:
                        # Append new information
                        self.semantic_memory[keyword]["information"] += f" {answer}"
                        # Trim if too long
                        if len(self.semantic_memory[keyword]["information"]) > 1000:
                            self.semantic_memory[keyword]["information"] = \
                                self.semantic_memory[keyword]["information"][:1000] + "..."
                else:
                    # Create new concept
                    self.semantic_memory[keyword] = {
                        "id": f"semantic_{uuid.uuid4().hex[:8]}",
                        "concept": keyword,
                        "information": answer[:1000] if len(answer) > 1000 else answer,
                        "occurrences": 1,
                        "created": timestamp,
                        "last_updated": timestamp,
                        "related_concepts": []
                    }
            
            # Save semantic memories
            await self._save_semantic_memories()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding semantic memory: {str(e)}")
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract key concepts from text.
        
        This is a simple keyword extraction that could be replaced
        with more sophisticated NLP approaches.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        # Simple approach: split by common separators and filter
        words = text.replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split()
        
        # Filter out common stop words and short words
        stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
                      "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                      "do", "does", "did", "can", "could", "will", "would", "should", "may",
                      "might", "must", "of", "by", "this", "that", "these", "those", "it", "its",
                      "from", "as", "not", "what", "when", "where", "who", "why", "how"}
        
        keywords = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 3]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:5]  # Return top 5 keywords
    
    async def _consolidate_semantic_memory(self):
        """
        Consolidate semantic memories to identify relationships and patterns.
        
        This is periodically triggered to update semantic memory connections.
        """
        logger.info("Starting semantic memory consolidation")
        self.last_semantic_refresh = time.time()
        
        if not self.vector_store_initialized or len(self.semantic_memory) < 2:
            logger.info("Not enough semantic memories to consolidate")
            return
        
        try:
            # Get all concepts
            concepts = list(self.semantic_memory.keys())
            
            # Generate embeddings for all concepts
            concept_embeddings = self.embedding_model.encode(concepts)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(concept_embeddings, concept_embeddings.T)
            
            # Update related concepts for each concept
            for i, concept in enumerate(concepts):
                # Get top 3 related concepts (excluding self)
                similarities = similarity_matrix[i]
                
                # Set self-similarity to 0
                similarities[i] = 0
                
                # Get indices of top 3 similar concepts
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                # Filter by minimum similarity threshold
                threshold = 0.6
                related = []
                for idx in top_indices:
                    if similarities[idx] >= threshold:
                        related.append({
                            "concept": concepts[idx],
                            "similarity": float(similarities[idx])
                        })
                
                # Update related concepts
                self.semantic_memory[concept]["related_concepts"] = related
            
            # Save semantic memories
            await self._save_semantic_memories()
            
            logger.info("Semantic memory consolidation complete")
            
        except Exception as e:
            logger.error(f"Error consolidating semantic memory: {str(e)}")
    
    async def _save_episodic_memories(self):
        """Save episodic memories to disk."""
        try:
            path = os.path.join(self.vector_db_path, "episodic", "memories.json")
            async with aiofiles.open(path, 'w') as f:
                await f.write(json.dumps(self.episodic_memory, indent=2))
        except Exception as e:
            logger.error(f"Error saving episodic memories: {str(e)}")
    
    async def _save_semantic_memories(self):
        """Save semantic memories to disk."""
        try:
            path = os.path.join(self.vector_db_path, "semantic", "memories.json")
            async with aiofiles.open(path, 'w') as f:
                await f.write(json.dumps(self.semantic_memory, indent=2))
        except Exception as e:
            logger.error(f"Error saving semantic memories: {str(e)}")
    
    async def _save_procedural_memories(self):
        """Save procedural memories to disk."""
        try:
            path = os.path.join(self.vector_db_path, "procedural", "memories.json")
            async with aiofiles.open(path, 'w') as f:
                await f.write(json.dumps(self.procedural_memory, indent=2))
        except Exception as e:
            logger.error(f"Error saving procedural memories: {str(e)}")
    
    async def _save_vector_store(self):
        """Save vector store to disk."""
        if not self.vector_store_initialized:
            return
        
        try:
            path = os.path.join(self.vector_db_path, "vector_store.npz")
            
            # Convert to numpy arrays
            texts = np.array(self.vector_store["texts"], dtype=object)
            embeddings = self.vector_store["embeddings"]
            metadata = np.array(self.vector_store["metadata"], dtype=object)
            
            # Save to disk
            np.savez_compressed(path, texts=texts, embeddings=embeddings, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
