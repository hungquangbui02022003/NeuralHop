"""
Confidence Scorer for TreeHop Enhanced

Provides intelligent confidence scoring for:
- Passage relevance and quality assessment
- Evidence strength evaluation  
- Selective LLM integration decisions
- Early stopping criteria
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ConfidenceScore:
    """Confidence scoring results"""
    overall_score: float  # 0-1 scale
    relevance_score: float
    information_novelty: float
    entity_coverage: float
    evidence_strength: float
    should_use_llm: bool
    confidence_level: str  # low, medium, high

class ConfidenceScorer:
    """Advanced confidence scoring for passage evaluation"""
    
    def __init__(self, low_threshold: float = 0.3, high_threshold: float = 0.7):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # Weights for multi-objective scoring
        self.score_weights = {
            'relevance': 0.4,
            'novelty': 0.2,
            'entity_coverage': 0.2,
            'evidence_strength': 0.2
        }
    
    def score_passages(self, 
                      query_embedding: torch.Tensor,
                      passage_embeddings: torch.Tensor,
                      passages: List[str],
                      query_entities: List[str],
                      retrieved_so_far: List[str] = None) -> List[ConfidenceScore]:
        """Score multiple passages for confidence and quality"""
        
        scores = []
        retrieved_so_far = retrieved_so_far or []
        
        for i, (passage_emb, passage_text) in enumerate(zip(passage_embeddings, passages)):
            # Calculate individual components
            relevance = self._calculate_relevance_score(query_embedding, passage_emb)
            novelty = self._calculate_information_novelty(passage_text, retrieved_so_far)
            entity_cov = self._calculate_entity_coverage(passage_text, query_entities)
            evidence = self._calculate_evidence_strength(passage_text)
            
            # Weighted overall score
            overall = (
                self.score_weights['relevance'] * relevance +
                self.score_weights['novelty'] * novelty +
                self.score_weights['entity_coverage'] * entity_cov +
                self.score_weights['evidence_strength'] * evidence
            )
            
            # Determine if LLM verification needed
            should_use_llm = overall < self.low_threshold
            
            # Confidence level categorization
            if overall >= self.high_threshold:
                confidence_level = 'high'
            elif overall >= self.low_threshold:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            score = ConfidenceScore(
                overall_score=overall,
                relevance_score=relevance,
                information_novelty=novelty,
                entity_coverage=entity_cov,
                evidence_strength=evidence,
                should_use_llm=should_use_llm,
                confidence_level=confidence_level
            )
            
            scores.append(score)
        
        return scores
    
    def _calculate_relevance_score(self, query_emb: torch.Tensor, passage_emb: torch.Tensor) -> float:
        """Calculate semantic relevance between query and passage"""
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if passage_emb.dim() == 1:
            passage_emb = passage_emb.unsqueeze(0)
        
        # Cosine similarity
        similarity = cosine_similarity(
            query_emb.detach().cpu().numpy(),
            passage_emb.detach().cpu().numpy()
        )[0][0]
        
        return max(0.0, min(1.0, float(similarity)))
    
    def _calculate_information_novelty(self, passage: str, previous_passages: List[str]) -> float:
        """Calculate how much new information this passage provides"""
        if not previous_passages:
            return 1.0
        
        # Simple word-level novelty calculation
        passage_words = set(passage.lower().split())
        
        total_previous_words = set()
        for prev_passage in previous_passages:
            total_previous_words.update(prev_passage.lower().split())
        
        if not total_previous_words:
            return 1.0
        
        # Calculate ratio of new words
        new_words = passage_words - total_previous_words
        novelty_ratio = len(new_words) / len(passage_words) if passage_words else 0.0
        
        return max(0.1, min(1.0, novelty_ratio))  # Minimum 0.1 to avoid complete rejection
    
    def _calculate_entity_coverage(self, passage: str, query_entities: List[str]) -> float:
        """Calculate how well passage covers query entities"""
        if not query_entities:
            return 0.5  # Neutral score if no entities to check
        
        passage_lower = passage.lower()
        covered_entities = 0
        
        for entity in query_entities:
            if entity.lower() in passage_lower:
                covered_entities += 1
        
        coverage_ratio = covered_entities / len(query_entities)
        return coverage_ratio
    
    def _calculate_evidence_strength(self, passage: str) -> float:
        """Calculate strength of evidence in passage"""
        evidence_indicators = [
            # Strong evidence words
            r'\b(?:study|research|found|demonstrated|proved|evidence|data|statistics)\b',
            # Quantitative indicators  
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d{4}\b',  # Years
            r'\b\d+(?:,\d{3})*\b',  # Large numbers
            # Authority indicators
            r'\b(?:professor|doctor|expert|scientist|researcher)\b',
            r'\b(?:university|institute|laboratory|journal)\b',
            # Factual indicators
            r'\b(?:according to|based on|reported|published|confirmed)\b'
        ]
        
        import re
        evidence_score = 0.0
        total_possible = len(evidence_indicators)
        
        for pattern in evidence_indicators:
            if re.search(pattern, passage.lower()):
                evidence_score += 1.0
        
        # Normalize and add length bonus for detailed passages
        base_score = evidence_score / total_possible
        
        # Length bonus (longer passages with evidence are often more comprehensive)
        word_count = len(passage.split())
        length_bonus = min(0.2, word_count / 1000)  # Max 0.2 bonus
        
        final_score = min(1.0, base_score + length_bonus)
        return final_score
    
    def should_continue_hopping(self, 
                               current_scores: List[ConfidenceScore],
                               hop_count: int,
                               max_hops: int) -> bool:
        """Determine if more hops are needed based on confidence scores"""
        
        if hop_count >= max_hops:
            return False
        
        if not current_scores:
            return True
        
        # Calculate average confidence
        avg_confidence = np.mean([score.overall_score for score in current_scores])
        
        # Early stopping criteria
        early_stop_conditions = [
            avg_confidence >= self.high_threshold,  # High confidence achieved
            hop_count >= 3 and avg_confidence >= self.low_threshold,  # Minimum quality after 3 hops
            all(score.confidence_level == 'high' for score in current_scores[:3])  # Top 3 all high quality
        ]
        
        return not any(early_stop_conditions)
    
    def select_best_passages(self, 
                           scores: List[ConfidenceScore],
                           passages: List[str],
                           max_passages: int = 10) -> Tuple[List[str], List[ConfidenceScore]]:
        """Select best passages based on confidence scores"""
        
        # Sort by overall score
        scored_passages = list(zip(scores, passages))
        scored_passages.sort(key=lambda x: x[0].overall_score, reverse=True)
        
        # Select top passages
        selected = scored_passages[:max_passages]
        selected_scores = [item[0] for item in selected]
        selected_passages = [item[1] for item in selected]
        
        return selected_passages, selected_scores
    
    def get_llm_verification_candidates(self, 
                                       scores: List[ConfidenceScore],
                                       passages: List[str]) -> List[Tuple[str, ConfidenceScore]]:
        """Get passages that need LLM verification"""
        candidates = []
        
        for score, passage in zip(scores, passages):
            if score.should_use_llm:
                candidates.append((passage, score))
        
        return candidates 