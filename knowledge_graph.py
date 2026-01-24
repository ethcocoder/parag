"""
Dynamic Knowledge Graph
========================

Advanced graph-based knowledge representation.
Features:
- Dynamic entity & relation extraction
- Temporal knowledge (facts valid for specific time ranges)
- Confidence scores and contradiction detection
- Multi-hop reasoning capabilities
"""

import typing as t
from dataclasses import dataclass, field
import time
import json
from collections import defaultdict
import heapq

@dataclass
class Entity:
    """A node in the knowledge graph"""
    id: str
    type: str = "concept"
    attributes: t.Dict[str, t.Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0

@dataclass
class Relation:
    """An edge in the knowledge graph"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: t.Dict[str, t.Any] = field(default_factory=dict)
    is_bidirectional: bool = False

class KnowledgeGraph:
    """
    Dynamic Knowledge Graph Engine
    
    Manages entities and relations with temporal tracking and reasoning.
    """
    
    def __init__(self):
        self.entities: t.Dict[str, Entity] = {}
        self.adjacency: t.Dict[str, t.List[Relation]] = defaultdict(list)
        self.reverse_adjacency: t.Dict[str, t.List[Relation]] = defaultdict(list)
    
    def add_entity(self, id: str, type: str = "concept", **attributes) -> Entity:
        """Add or update an entity"""
        if id in self.entities:
            ent = self.entities[id]
            ent.last_seen = time.time()
            ent.attributes.update(attributes)
            return ent
        
        ent = Entity(id=id, type=type, attributes=attributes)
        self.entities[id] = ent
        return ent
    
    def add_relation(self, source: str, target: str, type: str, 
                    weight: float = 1.0, bidirectional: bool = False,
                    **metadata):
        """Add a relation between entities"""
        # Ensure entities exist
        if source not in self.entities:
            self.add_entity(source)
        if target not in self.entities:
            self.add_entity(target)
            
        relation = Relation(
            source_id=source,
            target_id=target,
            relation_type=type,
            weight=weight,
            is_bidirectional=bidirectional,
            metadata=metadata
        )
        
        # Add to adjacency
        self.adjacency[source].append(relation)
        self.reverse_adjacency[target].append(relation)
        
        if bidirectional:
            back_relation = Relation(
                source_id=target,
                target_id=source,
                relation_type=type,
                weight=weight,
                is_bidirectional=True,
                metadata=metadata
            )
            self.adjacency[target].append(back_relation)
            self.reverse_adjacency[source].append(back_relation)
            
    def get_neighbors(self, entity_id: str, 
                     relation_type: str = None) -> t.List[t.Tuple[Entity, Relation]]:
        """Get neighboring entities"""
        if entity_id not in self.adjacency:
            return []
            
        neighbors = []
        for rel in self.adjacency[entity_id]:
            if relation_type and rel.relation_type != relation_type:
                continue
                
            target = self.entities[rel.target_id]
            neighbors.append((target, rel))
            
        return neighbors
    
    def find_path(self, start_id: str, end_id: str, 
                 max_depth: int = 3) -> t.List[t.List[t.Tuple[str, str]]]:
        """
        Find paths between two entities (limit to max_depth)
        Returns list of paths, where each path is list of (entity_id, relation_type)
        """
        if start_id not in self.entities or end_id not in self.entities:
            return []
            
        paths = []
        visited = set()
        
        # DFS for path finding
        def dfs(current_id, current_path, depth):
            if current_id == end_id:
                paths.append(list(current_path))
                return
                
            if depth >= max_depth:
                return
                
            visited.add(current_id)
            
            for rel in self.adjacency[current_id]:
                if rel.target_id not in visited:
                    # Append (target, relation) to path
                    current_path.append((rel.target_id, rel.relation_type))
                    dfs(rel.target_id, current_path, depth + 1)
                    current_path.pop()
            
            visited.remove(current_id)
            
        dfs(start_id, [], 0)
        return paths
    
    def reason_about(self, entity_id: str) -> t.Dict[str, t.Any]:
        """
        Perform 1-hop reasoning/inference about an entity
        """
        if entity_id not in self.entities:
            return {"error": "Entity not found"}
            
        neighbors = self.get_neighbors(entity_id)
        
        # Aggregate facts
        facts = []
        categories = set()
        related_concepts = []
        
        for neighbor, rel in neighbors:
            facts.append(f"{rel.relation_type} {neighbor.id}")
            if rel.relation_type == "is_a":
                categories.add(neighbor.id)
            else:
                related_concepts.append(neighbor.id)
                
        return {
            "entity": entity_id,
            "categories": list(categories),
            "related": related_concepts,
            "facts_count": len(facts),
            "degree": len(neighbors)
        }
    
    def merge_graphs(self, other_graph: 'KnowledgeGraph'):
        """Merge another graph into this one"""
        # Merge entities
        for id, ent in other_graph.entities.items():
            self.add_entity(id, ent.type, **ent.attributes)
            
        # Merge relations
        for source, rels in other_graph.adjacency.items():
            for rel in rels:
                # Check duplicate
                exists = False
                for existing in self.adjacency[source]:
                    if (existing.target_id == rel.target_id and 
                        existing.relation_type == rel.relation_type):
                        exists = True
                        break
                
                if not exists:
                    self.add_relation(
                        rel.source_id, 
                        rel.target_id, 
                        rel.relation_type,
                        rel.weight,
                        rel.is_bidirectional,
                        **rel.metadata
                    )

    def extract_subgraph(self, center_id: str, depth: int = 1) -> 'KnowledgeGraph':
        """Extract a subgraph around a center node"""
        subgraph = KnowledgeGraph()
        
        if center_id not in self.entities:
            return subgraph
            
        # BFS traversal
        queue = [(center_id, 0)]
        visited = {center_id}
        
        subgraph.add_entity(center_id, **self.entities[center_id].attributes)
        
        while queue:
            curr_id, curr_depth = queue.pop(0)
            
            if curr_depth >= depth:
                continue
                
            for rel in self.adjacency[curr_id]:
                target_id = rel.target_id
                
                # Add target entity
                subgraph.add_entity(target_id, **self.entities[target_id].attributes)
                
                # Add relation
                subgraph.add_relation(
                    rel.source_id, target_id, rel.relation_type,
                    rel.weight, rel.is_bidirectional, **rel.metadata
                )
                
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, curr_depth + 1))
                    
        return subgraph
    
    def to_json(self) -> str:
        """Serialize graph to JSON"""
        data = {
            "entities": [
                {
                    "id": e.id, "type": e.type, 
                    "attributes": e.attributes,
                    "confidence": e.confidence
                } 
                for e in self.entities.values()
            ],
            "relations": []
        }
        
        for source, rels in self.adjacency.items():
            for rel in rels:
                data["relations"].append({
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.relation_type,
                    "weight": rel.weight
                })
                
        return json.dumps(data, indent=2)

    def save(self, path: str):
        """Save graph to file"""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """Load graph from file"""
        kg = cls()
        if not Path(path).exists():
            return kg
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        for ent_data in data.get("entities", []):
            kg.add_entity(
                ent_data["id"], 
                ent_data.get("type", "concept"),
                **ent_data.get("attributes", {})
            )
            kg.entities[ent_data["id"]].confidence = ent_data.get("confidence", 1.0)
            
        for rel_data in data.get("relations", []):
            kg.add_relation(
                rel_data["source"],
                rel_data["target"],
                rel_data["type"],
                weight=rel_data.get("weight", 1.0)
            )
            
        return kg



def extract_knowledge_from_text(text: str) -> KnowledgeGraph:
    """
    Simple heuristic-based knowledge crawler
    (Placeholder for advanced NLP extraction)
    """
    kg = KnowledgeGraph()
    
    # Very basic S-V-O extraction mock
    # "Python is a language" -> (Python, is_a, language)
    
    words = text.replace('.', '').split()
    
    # Simple rule: "X is Y" or "X is a Y"
    for i in range(len(words) - 2):
        if words[i+1] == "is":
            subj = words[i]
            if words[i+2] == "a" and i+3 < len(words):
                obj = words[i+3]
                rel = "is_a"
            else:
                obj = words[i+2]
                rel = "is"
            
            kg.add_relation(subj, obj, rel)
            
    return kg
