"""
Smart Query Processor for TreeHop Enhanced

Handles intelligent query preprocessing including:
- Multi-facet query decomposition
- Entity extraction and expansion
- Syntactic complexity analysis
- Query optimization for better retrieval
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class QueryAnalysis:
    """Analysis results for input query"""
    original_query: str
    complexity_score: float  # 0-1 scale
    entity_count: int
    question_type: str  # factual, multi_hop, comparison, etc.
    facets: List[str]
    entities: List[str]
    optimized_query: str

class SmartQueryProcessor:
    """Advanced query processor with multi-facet analysis"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Question type patterns
        self.question_patterns = {
            'factual': [r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b'],
            'multi_hop': [r'\bcompare\b', r'\bdifference between\b', r'\brelationship\b'],
            'comparison': [r'\bbetter\b', r'\bversus\b', r'\bvs\b', r'\bcompared to\b'],
            'causal': [r'\bwhy\b', r'\bhow does\b', r'\bcause\b', r'\bresult\b'],
            'temporal': [r'\bbefore\b', r'\bafter\b', r'\bduring\b', r'\bsequence\b']
        }
        
        # Entity patterns (simple rule-based)
        self.entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|Company|Corporation)\b',  # Organizations
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|State|Country)\b'  # Locations
        ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""
        
        # Basic preprocessing
        clean_query = self._clean_query(query)
        
        # Complexity analysis
        complexity = self._calculate_complexity(clean_query)
        
        # Question type detection
        question_type = self._detect_question_type(clean_query)
        
        # Entity extraction
        entities = self._extract_entities(clean_query)
        
        # Query facet decomposition
        facets = self._decompose_facets(clean_query, question_type)
        
        # Query optimization
        optimized = self._optimize_query(clean_query, entities, facets)
        
        return QueryAnalysis(
            original_query=query,
            complexity_score=complexity,
            entity_count=len(entities),
            question_type=question_type,
            facets=facets,
            entities=entities,
            optimized_query=optimized
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Normalize punctuation
        query = re.sub(r'[^\w\s\?\.\,\-]', '', query)
        
        return query
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)"""
        factors = []
        
        # Length factor
        token_count = len(self.tokenizer.tokenize(query))
        length_score = min(token_count / 50.0, 1.0)
        factors.append(length_score)
        
        # Syntactic complexity
        clause_count = len(re.findall(r'\b(?:and|or|but|which|that|where|when)\b', query.lower()))
        syntax_score = min(clause_count / 5.0, 1.0)
        factors.append(syntax_score)
        
        # Question word count
        question_words = len(re.findall(r'\b(?:what|who|when|where|why|how|which)\b', query.lower()))
        question_score = min(question_words / 3.0, 1.0)
        factors.append(question_score)
        
        return np.mean(factors)
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question being asked"""
        query_lower = query.lower()
        
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return q_type
        
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities using rule-based patterns"""
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _decompose_facets(self, query: str, question_type: str) -> List[str]:
        """Decompose query into searchable facets"""
        facets = []
        
        # Split by conjunctions
        conjunctive_splits = re.split(r'\b(?:and|or)\b', query, flags=re.IGNORECASE)
        
        for split in conjunctive_splits:
            split = split.strip()
            if len(split) > 10:  # Minimum meaningful length
                facets.append(split)
        
        # If no splits found, use the original query
        if not facets:
            facets = [query]
        
        # Add question-type specific facets
        if question_type == 'comparison':
            # Extract items being compared
            items = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            if len(items) >= 2:
                for item in items[:2]:
                    facets.append(f"information about {item}")
        
        return facets[:5]  # Limit to 5 facets
    
    def _optimize_query(self, query: str, entities: List[str], facets: List[str]) -> str:
        """Optimize query for better retrieval"""
        optimized = query
        
        # Add entity context if missing
        if entities and not any(entity.lower() in query.lower() for entity in entities):
            main_entity = entities[0]
            optimized = f"{query} related to {main_entity}"
        
        # Expand abbreviations (simple rules)
        abbreviations = {
            r'\bUS\b': 'United States',
            r'\bUK\b': 'United Kingdom', 
            r'\bAI\b': 'artificial intelligence',
            r'\bML\b': 'machine learning'
        }
        
        for abbrev, expansion in abbreviations.items():
            optimized = re.sub(abbrev, expansion, optimized)
        
        return optimized
    
    def get_adaptive_hop_count(self, analysis: QueryAnalysis) -> int:
        """Determine optimal hop count based on query analysis"""
        base_hops = 2
        
        # Increase hops for complex queries
        if analysis.complexity_score > 0.7:
            base_hops += 2
        elif analysis.complexity_score > 0.4:
            base_hops += 1
        
        # Increase for multi-hop question types
        if analysis.question_type in ['multi_hop', 'comparison', 'causal']:
            base_hops += 1
        
        # Increase for entity-rich queries
        if analysis.entity_count > 3:
            base_hops += 1
        
        return min(base_hops, 5)  # Cap at 5 hops 