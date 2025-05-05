"""
Semantic Memory implementation for storing and retrieving general knowledge.

This module provides specialized functionality for managing semantic memories,
which represent factual knowledge, concepts, definitions, and relationships
about the world or specific domains.
"""

import logging
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)

class SemanticMemory:
    """
    Specialized memory system for storing and retrieving semantic knowledge.
    
    Semantic memory stores factual knowledge, concepts, definitions, and relationships
    about the world or specific domains. It provides the foundational knowledge
    required for reasoning and understanding.
    """
    
    def __init__(self, memory_store: MemoryStore):
        """
        Initialize the semantic memory system.
        
        Args:
            memory_store: The main memory store instance
        """
        self.memory_store = memory_store
        logger.info("Semantic memory system initialized")
    
    async def store_concept(self, concept: str, information: str, 
                         related_concepts: Optional[List[str]] = None) -> str:
        """
        Store a concept in semantic memory.
        
        Args:
            concept: The concept name or term
            information: Information about the concept
            related_concepts: Optional list of related concepts
            
        Returns:
            Memory ID of the stored concept
        """
        timestamp = datetime.now().isoformat()
        
        # Check if concept already exists
        existing_concept = await self.retrieve_concept(concept)
        
        if existing_concept:
            # Update existing concept
            concept_id = existing_concept["id"]
            
            # Merge information
            current_info = existing_concept["information"]
            
            # Only update if new information adds value
            if information not in current_info:
                # Combine information
                updated_info = current_info + " " + information
                
                # Trim if too long
                if len(updated_info) > 1000:
                    updated_info = updated_info[:1000] + "..."
                
                # Update the concept
                self.memory_store.semantic_memory[concept] = {
                    "id": concept_id,
                    "concept": concept,
                    "information": updated_info,
                    "occurrences": existing_concept["occurrences"] + 1,
                    "created": existing_concept["created"],
                    "last_updated": timestamp,
                    "related_concepts": existing_concept["related_concepts"]
                }
                
                # Update related concepts if provided
                if related_concepts:
                    related_list = [{"concept": c, "similarity": 0.8} for c in related_concepts]
                    # Merge with existing related concepts
                    existing_related = {rc["concept"]: rc for rc in existing_concept["related_concepts"]}
                    for rel in related_list:
                        existing_related[rel["concept"]] = rel
                    
                    # Convert back to list
                    self.memory_store.semantic_memory[concept]["related_concepts"] = list(existing_related.values())
            else:
                # Just increment occurrences
                self.memory_store.semantic_memory[concept]["occurrences"] += 1
                self.memory_store.semantic_memory[concept]["last_updated"] = timestamp
        else:
            # Create new concept
            concept_id = f"semantic_{uuid.uuid4().hex[:8]}"
            
            self.memory_store.semantic_memory[concept] = {
                "id": concept_id,
                "concept": concept,
                "information": information[:1000] if len(information) > 1000 else information,
                "occurrences": 1,
                "created": timestamp,
                "last_updated": timestamp,
                "related_concepts": [{"concept": c, "similarity": 0.8} for c in (related_concepts or [])]
            }
        
        # Save semantic memories
        await self.memory_store._save_semantic_memories()
        
        return concept_id
    
    async def retrieve_concept(self, concept: str) -> Optional[Dict]:
        """
        Retrieve a concept from semantic memory.
        
        Args:
            concept: The concept to retrieve
            
        Returns:
            The concept data if found, None otherwise
        """
        return await self.memory_store.retrieve_semantic_concept(concept)
    
    async def retrieve_related_concepts(self, concept: str, max_depth: int = 2) -> List[Dict]:
        """
        Retrieve a concept and its related concepts to a specified depth.
        
        Args:
            concept: The concept to start from
            max_depth: Maximum depth of related concepts to retrieve
            
        Returns:
            The concept and its related concepts in a graph structure
        """
        if max_depth <= 0:
            return []
        
        # Get the initial concept
        concept_data = await self.retrieve_concept(concept)
        
        if not concept_data:
            return []
        
        # Start building the graph
        graph = {
            concept: {
                "data": concept_data,
                "related": {}
            }
        }
        
        # Process the first level of related concepts
        await self._retrieve_related_recursive(concept, graph, current_depth=1, max_depth=max_depth)
        
        # Convert to a list format
        result = []
        for concept_name, concept_info in graph.items():
            concept_entry = {
                "concept": concept_name,
                "data": concept_info["data"],
                "related": []
            }
            
            # Add related concepts
            for related_name, related_info in concept_info["related"].items():
                concept_entry["related"].append({
                    "concept": related_name,
                    "data": related_info["data"],
                    "similarity": related_info.get("similarity", 0)
                })
            
            result.append(concept_entry)
        
        return result
    
    async def _retrieve_related_recursive(self, concept: str, graph: Dict, 
                                       current_depth: int, max_depth: int) -> None:
        """
        Recursively retrieve related concepts up to a maximum depth.
        
        Args:
            concept: The current concept
            graph: The graph structure being built
            current_depth: Current depth in the recursion
            max_depth: Maximum depth to explore
        """
        if current_depth >= max_depth or concept not in graph:
            return
        
        # Get the related concepts
        for related in graph[concept]["data"].get("related_concepts", []):
            related_concept = related["concept"]
            similarity = related.get("similarity", 0)
            
            # Skip if already in the graph to avoid cycles
            if related_concept in graph:
                continue
            
            # Retrieve the related concept
            related_data = await self.retrieve_concept(related_concept)
            
            if related_data:
                # Add to the graph
                graph[related_concept] = {
                    "data": related_data,
                    "related": {},
                    "similarity": similarity
                }
                
                # Add the bidirectional relationship to the original concept
                graph[concept]["related"][related_concept] = {
                    "data": related_data,
                    "similarity": similarity
                }
                
                # Recursively process this concept's relations
                await self._retrieve_related_recursive(related_concept, graph, 
                                                    current_depth + 1, max_depth)
    
    async def retrieve_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve concepts containing the keyword.
        
        Args:
            keyword: The keyword to search for
            limit: Maximum number of concepts to retrieve
            
        Returns:
            List of matching concepts
        """
        results = []
        keyword_lower = keyword.lower()
        
        # Search in concept names and information
        for concept_name, concept_data in self.memory_store.semantic_memory.items():
            if (keyword_lower in concept_name.lower() or 
                keyword_lower in concept_data["information"].lower()):
                
                results.append(concept_data)
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def retrieve_most_frequent(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve the most frequently accessed concepts.
        
        Args:
            limit: Maximum number of concepts to retrieve
            
        Returns:
            List of concepts ordered by frequency
        """
        # Sort concepts by occurrence count
        sorted_concepts = sorted(
            self.memory_store.semantic_memory.values(),
            key=lambda x: x.get("occurrences", 0),
            reverse=True
        )
        
        return sorted_concepts[:limit]
    
    async def retrieve_recently_updated(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve the most recently updated concepts.
        
        Args:
            limit: Maximum number of concepts to retrieve
            
        Returns:
            List of concepts ordered by last update time
        """
        # Sort concepts by last updated timestamp
        sorted_concepts = sorted(
            self.memory_store.semantic_memory.values(),
            key=lambda x: x.get("last_updated", ""),
            reverse=True
        )
        
        return sorted_concepts[:limit]
    
    async def find_similar_concepts(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Find concepts similar to the query.
        
        Args:
            query: The search query
            limit: Maximum number of concepts to retrieve
            
        Returns:
            List of similar concepts
        """
        # If vector store is available, use it for similarity search
        if self.memory_store.vector_store_initialized:
            # Get all concepts
            concept_names = list(self.memory_store.semantic_memory.keys())
            
            if not concept_names:
                return []
            
            try:
                # Generate embedding for the query
                query_embedding = self.memory_store.embedding_model.encode([query])[0]
                
                # Generate embeddings for all concepts
                concept_embeddings = self.memory_store.embedding_model.encode(concept_names)
                
                # Calculate similarity scores
                similarities = []
                for i, concept in enumerate(concept_names):
                    embedding = concept_embeddings[i]
                    similarity = float(np.dot(query_embedding, embedding))
                    
                    similarities.append((concept, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get the top concepts
                results = []
                for concept_name, similarity in similarities[:limit]:
                    concept_data = self.memory_store.semantic_memory[concept_name]
                    concept_data["similarity"] = similarity
                    results.append(concept_data)
                
                return results
                
            except Exception as e:
                logger.error(f"Error finding similar concepts: {str(e)}")
        
        # Fallback to keyword search
        return await self.retrieve_by_keyword(query, limit)
    
    async def update_concept_relations(self) -> bool:
        """
        Update relationships between concepts.
        
        This method triggers the semantic memory consolidation process.
        
        Returns:
            True if successfully updated, False otherwise
        """
        try:
            await self.memory_store._consolidate_semantic_memory()
            return True
        except Exception as e:
            logger.error(f"Error updating concept relations: {str(e)}")
            return False
    
    async def delete_concept(self, concept: str) -> bool:
        """
        Delete a concept from semantic memory.
        
        Args:
            concept: The concept to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if concept in self.memory_store.semantic_memory:
            # Remove the concept
            del self.memory_store.semantic_memory[concept]
            
            # Remove references in related concepts
            for other_concept, data in self.memory_store.semantic_memory.items():
                related = data.get("related_concepts", [])
                data["related_concepts"] = [r for r in related if r["concept"] != concept]
            
            # Save semantic memories
            await self.memory_store._save_semantic_memories()
            
            return True
        
        return False
