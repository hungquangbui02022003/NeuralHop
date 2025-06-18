"""
Intelligent Post-Processor for TreeHop Enhanced

Handles advanced post-processing including:
- Passage clustering and deduplication
- Evidence confidence scoring
- Selective LLM verification
- Answer synthesis optimization
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict

from .confidence_scorer import ConfidenceScore

@dataclass
class ProcessedResult:
    """Results from intelligent post-processing"""
    deduplicated_passages: List[str]
    clustered_passages: Dict[int, List[str]]
    evidence_summary: str
    confidence_summary: Dict[str, float]
    llm_verification_results: Optional[Dict] = None
    processing_metadata: Dict = None

class IntelligentPostProcessor:
    """Advanced post-processing for enhanced TreeHop results"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 cluster_count: int = 3,
                 min_evidence_score: float = 0.4):
        self.similarity_threshold = similarity_threshold
        self.cluster_count = cluster_count
        self.min_evidence_score = min_evidence_score
        
        # Evidence extraction patterns
        self.evidence_patterns = {
            'factual': [
                r'according to [^,\.]+',
                r'studies show that [^,\.]+',
                r'research indicates [^,\.]+',
                r'data suggests [^,\.]+',
                r'evidence demonstrates [^,\.]+',
            ],
            'statistical': [
                r'\d+(?:\.\d+)?%[^,\.]*',
                r'\d+(?:,\d{3})*[^,\.]*',
                r'increased by \d+[^,\.]*',
                r'decreased by \d+[^,\.]*',
            ],
            'authoritative': [
                r'(?:Dr\.|Professor|Expert) [A-Z][a-z]+ [A-Z][a-z]+[^,\.]*',
                r'[A-Z][a-z]+ University[^,\.]*',
                r'published in [^,\.]+',
                r'peer-reviewed [^,\.]+',
            ],
            'temporal': [
                r'in \d{4}[^,\.]*',
                r'since \d{4}[^,\.]*',
                r'from \d{4} to \d{4}[^,\.]*',
                r'recent studies[^,\.]*',
            ]
        }
    
    def process_results(self, 
                       passages: List[str],
                       confidence_scores: List[ConfidenceScore],
                       passage_embeddings: torch.Tensor,
                       query: str,
                       use_llm_verification: bool = True) -> ProcessedResult:
        """Comprehensive post-processing of TreeHop results"""
        
        # Step 1: Deduplication
        deduplicated_passages, dedup_scores, dedup_embeddings = self._deduplicate_passages(
            passages, confidence_scores, passage_embeddings
        )
        
        # Step 2: Clustering for organization
        clustered_passages = self._cluster_passages(
            deduplicated_passages, dedup_embeddings
        )
        
        # Step 3: Evidence extraction and synthesis
        evidence_summary = self._extract_and_synthesize_evidence(
            deduplicated_passages, dedup_scores
        )
        
        # Step 4: Confidence analysis
        confidence_summary = self._analyze_confidence_distribution(dedup_scores)
        
        # Step 5: LLM verification if needed and requested
        llm_verification_results = None
        if use_llm_verification and self._should_use_llm_verification(dedup_scores):
            llm_verification_results = self._prepare_llm_verification_request(
                deduplicated_passages, dedup_scores, query
            )
        
        # Metadata for analysis
        processing_metadata = {
            'original_passage_count': len(passages),
            'deduplicated_count': len(deduplicated_passages),
            'cluster_distribution': {k: len(v) for k, v in clustered_passages.items()},
            'average_confidence': np.mean([s.overall_score for s in dedup_scores]) if dedup_scores else 0.0,
            'evidence_extraction_count': len(self._extract_all_evidence(deduplicated_passages)),
            'requires_llm_verification': llm_verification_results is not None
        }
        
        return ProcessedResult(
            deduplicated_passages=deduplicated_passages,
            clustered_passages=clustered_passages,
            evidence_summary=evidence_summary,
            confidence_summary=confidence_summary,
            llm_verification_results=llm_verification_results,
            processing_metadata=processing_metadata
        )
    
    def _deduplicate_passages(self, 
                             passages: List[str],
                             scores: List[ConfidenceScore],
                             embeddings: torch.Tensor) -> Tuple[List[str], List[ConfidenceScore], torch.Tensor]:
        """Remove duplicate or highly similar passages"""
        
        if len(passages) <= 1:
            return passages, scores, embeddings
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings.detach().cpu().numpy())
        
        # Track which passages to keep
        keep_indices = []
        
        # Sort by confidence score (keep highest quality)
        sorted_indices = sorted(range(len(scores)), 
                              key=lambda i: scores[i].overall_score, reverse=True)
        
        for idx in sorted_indices:
            # Check if this passage is similar to any we're already keeping
            is_duplicate = False
            
            for kept_idx in keep_indices:
                if similarity_matrix[idx][kept_idx] > self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(idx)
        
        # Return deduplicated results
        dedup_passages = [passages[i] for i in keep_indices]
        dedup_scores = [scores[i] for i in keep_indices]
        dedup_embeddings = embeddings[keep_indices]
        
        return dedup_passages, dedup_scores, dedup_embeddings
    
    def _cluster_passages(self, 
                         passages: List[str],
                         embeddings: torch.Tensor) -> Dict[int, List[str]]:
        """Cluster passages by topic/theme"""
        
        if len(passages) <= self.cluster_count:
            return {i: [passage] for i, passage in enumerate(passages)}
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings.detach().cpu().numpy())
        
        # Group passages by cluster
        clustered = defaultdict(list)
        for passage, label in zip(passages, cluster_labels):
            clustered[int(label)].append(passage)
        
        return dict(clustered)
    
    def _extract_and_synthesize_evidence(self, 
                                        passages: List[str],
                                        scores: List[ConfidenceScore]) -> str:
        """Extract and synthesize key evidence from passages"""
        
        all_evidence = self._extract_all_evidence(passages)
        
        # Weight evidence by passage confidence
        weighted_evidence = []
        
        for i, passage in enumerate(passages):
            passage_evidence = self._extract_evidence_from_passage(passage)
            confidence_weight = scores[i].overall_score if i < len(scores) else 0.5
            
            for evidence_type, evidence_list in passage_evidence.items():
                for evidence in evidence_list:
                    weighted_evidence.append({
                        'text': evidence,
                        'type': evidence_type,
                        'weight': confidence_weight,
                        'source_passage_idx': i
                    })
        
        # Sort by weight and type importance
        type_importance = {'factual': 4, 'authoritative': 3, 'statistical': 2, 'temporal': 1}
        
        weighted_evidence.sort(
            key=lambda x: (type_importance.get(x['type'], 0), x['weight']), 
            reverse=True
        )
        
        # Synthesize top evidence into summary
        summary_parts = []
        used_types = set()
        
        for evidence in weighted_evidence[:10]:  # Top 10 pieces of evidence
            if evidence['type'] not in used_types or len(summary_parts) < 3:
                summary_parts.append(f"• {evidence['text'].strip()}")
                used_types.add(evidence['type'])
        
        if summary_parts:
            return "Key Evidence:\n" + "\n".join(summary_parts)
        else:
            return "No significant evidence patterns detected."
    
    def _extract_all_evidence(self, passages: List[str]) -> Dict[str, List[str]]:
        """Extract all evidence from all passages"""
        all_evidence = defaultdict(list)
        
        for passage in passages:
            passage_evidence = self._extract_evidence_from_passage(passage)
            for evidence_type, evidence_list in passage_evidence.items():
                all_evidence[evidence_type].extend(evidence_list)
        
        return dict(all_evidence)
    
    def _extract_evidence_from_passage(self, passage: str) -> Dict[str, List[str]]:
        """Extract evidence from a single passage"""
        evidence = defaultdict(list)
        
        for evidence_type, patterns in self.evidence_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, passage, re.IGNORECASE)
                evidence[evidence_type].extend(matches)
        
        return dict(evidence)
    
    def _analyze_confidence_distribution(self, scores: List[ConfidenceScore]) -> Dict[str, float]:
        """Analyze confidence score distribution"""
        
        if not scores:
            return {'error': 'No scores provided'}
        
        overall_scores = [s.overall_score for s in scores]
        relevance_scores = [s.relevance_score for s in scores]
        novelty_scores = [s.information_novelty for s in scores]
        entity_scores = [s.entity_coverage for s in scores]
        evidence_scores = [s.evidence_strength for s in scores]
        
        return {
            'average_overall': np.mean(overall_scores),
            'median_overall': np.median(overall_scores),
            'std_overall': np.std(overall_scores),
            'average_relevance': np.mean(relevance_scores),
            'average_novelty': np.mean(novelty_scores),
            'average_entity_coverage': np.mean(entity_scores),
            'average_evidence_strength': np.mean(evidence_scores),
            'high_confidence_ratio': sum(1 for s in scores if s.confidence_level == 'high') / len(scores),
            'low_confidence_ratio': sum(1 for s in scores if s.confidence_level == 'low') / len(scores)
        }
    
    def _should_use_llm_verification(self, scores: List[ConfidenceScore]) -> bool:
        """Determine if LLM verification is needed"""
        
        if not scores:
            return True
        
        overall_scores = [s.overall_score for s in scores]
        avg_confidence = np.mean(overall_scores)
        low_confidence_ratio = sum(1 for s in scores if s.confidence_level == 'low') / len(scores)
        confidence_variance = np.var(overall_scores)
        
        verification_criteria = [
            avg_confidence < 0.6,
            low_confidence_ratio > 0.3,
            confidence_variance > 0.1
        ]
        
        return any(verification_criteria)
    
    def _prepare_llm_verification_request(self, 
                                         passages: List[str],
                                         scores: List[ConfidenceScore],
                                         query: str) -> Dict:
        """Prepare LLM verification request with context"""
        
        verification_candidates = []
        for i, (passage, score) in enumerate(zip(passages, scores)):
            if score.should_use_llm or score.confidence_level == 'low':
                verification_candidates.append({
                    'passage': passage,
                    'index': i,
                    'confidence_score': score.overall_score,
                    'issues': self._identify_confidence_issues(score)
                })
        
        verification_prompt = self._generate_verification_prompt(
            query, verification_candidates
        )
        
        return {
            'verification_candidates': verification_candidates,
            'verification_prompt': verification_prompt,
            'candidate_count': len(verification_candidates),
            'request_type': 'confidence_verification'
        }
    
    def _identify_confidence_issues(self, score: ConfidenceScore) -> List[str]:
        """Identify specific confidence issues"""
        issues = []
        
        if score.relevance_score < 0.5:
            issues.append('low_relevance')
        if score.information_novelty < 0.3:
            issues.append('redundant_information')
        if score.entity_coverage < 0.4:
            issues.append('poor_entity_coverage')
        if score.evidence_strength < 0.3:
            issues.append('weak_evidence')
        
        return issues
    
    def _generate_verification_prompt(self, 
                                     query: str,
                                     candidates: List[Dict]) -> str:
        """Generate prompt for LLM verification"""
        
        prompt = f"""Please verify and assess the following passages for the query: "{query}"

Instructions:
1. Rate each passage's relevance to the query (1-10)
2. Identify any factual inconsistencies or errors
3. Suggest improvements or additional context needed
4. Provide an overall assessment

Passages to verify:

"""
        
        for i, candidate in enumerate(candidates):
            prompt += f"Passage {i+1} (Current confidence: {candidate['confidence_score']:.2f}):\n"
            prompt += f"{candidate['passage']}\n"
            prompt += f"Identified issues: {', '.join(candidate['issues'])}\n\n"
        
        prompt += """
Please provide your assessment in the following format:
- Passage X: [Rating] - [Assessment] - [Suggestions]
- Overall recommendation: [Keep/Revise/Remove for each passage]
"""
        
        return prompt 