"""
Enhanced TreeHop Model with Advanced Update Mechanisms

Features:
- Enhanced updateGate formula with contextual weighting
- Multi-objective passage scoring
- Adaptive attention mechanisms
- Dynamic hop optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math

from tree_hop.model import TreeHopNode as OriginalTreeHopNode, AttentionHead2D
from .confidence_scorer import ConfidenceScorer, ConfidenceScore

class EnhancedAttentionHead2D(AttentionHead2D):
    """Enhanced 2D attention with contextual weighting"""
    
    def __init__(self, input_dim: int, attention_dim: int = None):
        super().__init__(input_dim, attention_dim)
        
        # Additional contextual weighting components
        self.context_weight_layer = nn.Linear(input_dim, 1)
        self.diversity_layer = nn.Linear(input_dim, 1)
        self.confidence_layer = nn.Linear(input_dim, 1)
        
    def forward(self, Q, K, V, context_info: Optional[Dict] = None):
        """Enhanced forward with contextual information"""
        # Original attention computation
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        
        if Q.dim() == 3:
            QK = torch.einsum("bud,bvd->buv", Q, K)
        elif Q.dim() == 2:
            QK = Q * K
            if QK.dim() == 2:
                QK = QK.sum(-1)
        else:
            raise ValueError(f"Unsupported dimension: {Q.dim()}")
        
        # Enhanced weighting if context provided
        if context_info is not None:
            # Calculate contextual weights
            context_weight = torch.sigmoid(self.context_weight_layer(K))
            diversity_weight = torch.sigmoid(self.diversity_layer(K))
            confidence_weight = torch.sigmoid(self.confidence_layer(K))
            
            # Combine weights
            combined_weight = (context_weight * diversity_weight * confidence_weight).squeeze(-1)
            
            # Apply to attention scores
            if QK.dim() == 2:
                QK = QK * combined_weight
            elif QK.dim() == 3:
                QK = QK * combined_weight.unsqueeze(1)
        
        A = self.softmax(QK)
        
        if A.dim() == 3:
            O = torch.einsum("buv,bvd->bud", A, V)
        elif A.dim() == 2:
            if V.dim() == 3:
                A = A.unsqueeze(-1)
            O = A * V
        
        return O

class EnhancedTreeHopNode(OriginalTreeHopNode):
    """Enhanced TreeHop node with improved update mechanisms"""
    
    def __init__(self, input_dim: int, attention_dim: int = None):
        # Initialize parent class components manually to avoid conflicts
        nn.Module.__init__(self)
        
        attention_dim = attention_dim or input_dim
        self.attention_dim = attention_dim
        self.input_dim = input_dim
        
        # Enhanced attention mechanism
        self.update_attn = EnhancedAttentionHead2D(input_dim, attention_dim)
        
        # Enhanced update gate with more sophisticated control
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Contextual weighting components
        self.alpha_layer = nn.Linear(input_dim, 1)  # For Q-K weighting
        self.beta_layer = nn.Linear(input_dim, 1)   # For attention scaling
        
        # Multi-objective scoring components
        self.relevance_scorer = nn.Linear(input_dim, 1)
        self.novelty_scorer = nn.Linear(input_dim, 1)
        self.entity_scorer = nn.Linear(input_dim, 1)
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, q: torch.Tensor, ctx: torch.Tensor, 
                context_info: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Enhanced forward pass with contextual weighting
        
        Args:
            q: Query embedding
            ctx: Context embeddings 
            context_info: Additional context information
            
        Returns:
            Tuple of (updated_query, metadata)
        """
        # Calculate enhanced update gate
        gate_input = torch.cat([q.expand_as(ctx), ctx], dim=-1)
        update_gate = self.update_gate(gate_input)
        
        # Calculate dynamic weighting factors
        alpha = torch.sigmoid(self.alpha_layer(ctx))  # Q-K weighting
        beta = torch.sigmoid(self.beta_layer(ctx))    # Attention scaling
        
        # Enhanced attention with context
        update_attn_out = self.update_attn(q, ctx, ctx, context_info)
        
        # Enhanced update formula: h = αQ - βK + γ*attention_scale(update_gate)
        # where α, β, γ are learned contextual weights
        gamma = beta  # Reuse beta for attention scaling
        
        if q.dim() == 2:
            alpha = alpha.squeeze(-1)
            beta = beta.squeeze(-1) 
            gamma = gamma.squeeze(-1)
        
        # Apply enhanced update rule
        h = (alpha * q - beta * ctx + gamma * update_attn_out) * update_gate
        
        # Calculate multi-objective scores
        relevance_scores = torch.sigmoid(self.relevance_scorer(h))
        novelty_scores = torch.sigmoid(self.novelty_scorer(h))
        entity_scores = torch.sigmoid(self.entity_scorer(h))
        
        # Confidence estimation
        confidence_scores = self.confidence_estimator(h)
        
        # Metadata for analysis
        metadata = {
            'update_gate': update_gate,
            'alpha_weights': alpha,
            'beta_weights': beta,
            'gamma_weights': gamma,
            'relevance_scores': relevance_scores,
            'novelty_scores': novelty_scores,
            'entity_scores': entity_scores,
            'confidence_scores': confidence_scores
        }
        
        return h, metadata

class TreeHopEnhanced(nn.Module):
    """Enhanced TreeHop model with adaptive mechanisms"""
    
    def __init__(self, 
                 encoder_model,
                 input_dim: int = 1024,
                 attention_dim: int = None,
                 max_hops: int = 5,
                 confidence_threshold: float = 0.7):
        super().__init__()
        
        self.encoder = encoder_model
        self.input_dim = input_dim
        self.max_hops = max_hops
        self.confidence_threshold = confidence_threshold
        
        # Enhanced TreeHop node
        self.tree_hop_node = EnhancedTreeHopNode(input_dim, attention_dim)
        
        # Confidence scorer
        self.confidence_scorer = ConfidenceScorer()
        
        # Adaptive hop predictor
        self.hop_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, max_hops),
            nn.Softmax(dim=-1)
        )
        
        # Performance caching
        self.query_cache = {}
        self.embedding_cache = {}
    
    def forward(self, 
                query: str,
                passages: List[str],
                passage_embeddings: torch.Tensor,
                query_entities: List[str] = None,
                adaptive_hops: bool = True) -> Dict:
        """
        Enhanced TreeHop forward pass
        
        Args:
            query: Input query string
            passages: List of passage texts
            passage_embeddings: Precomputed passage embeddings
            query_entities: Extracted entities from query
            adaptive_hops: Whether to use adaptive hop count
            
        Returns:
            Dictionary with results and metadata
        """
        # Encode query
        query_embedding = self._encode_text(query)
        
        # Predict optimal hop count if adaptive
        if adaptive_hops:
            hop_probs = self.hop_predictor(query_embedding)
            predicted_hops = torch.argmax(hop_probs).item() + 1
            max_hops = min(predicted_hops, self.max_hops)
        else:
            max_hops = self.max_hops
        
        # Initialize tracking variables
        current_query = query_embedding
        retrieved_passages = []
        all_scores = []
        hop_metadata = []
        
        # Multi-hop retrieval loop
        for hop in range(max_hops):
            # Score current passages
            confidence_scores = self.confidence_scorer.score_passages(
                query_embedding=current_query,
                passage_embeddings=passage_embeddings,
                passages=passages,
                query_entities=query_entities or [],
                retrieved_so_far=retrieved_passages
            )
            
            # Check early stopping
            if not self.confidence_scorer.should_continue_hopping(
                confidence_scores, hop, max_hops):
                break
            
            # Select best passages for this hop
            selected_passages, selected_scores = self.confidence_scorer.select_best_passages(
                confidence_scores, passages, max_passages=10
            )
            
            # Update query using TreeHop mechanism
            if selected_passages:
                # Get embeddings for selected passages
                selected_indices = [passages.index(p) for p in selected_passages]
                selected_embeddings = passage_embeddings[selected_indices]
                
                # Prepare context information
                context_info = {
                    'hop_count': hop,
                    'confidence_scores': [s.overall_score for s in selected_scores],
                    'entities': query_entities or []
                }
                
                # Apply TreeHop update
                updated_query, metadata = self.tree_hop_node(
                    current_query, selected_embeddings, context_info
                )
                
                # Select best updated query
                best_idx = torch.argmax(metadata['confidence_scores']).item()
                current_query = updated_query[best_idx:best_idx+1]
                
                # Track results
                retrieved_passages.extend(selected_passages[:3])  # Top 3 per hop
                all_scores.extend(selected_scores[:3])
                hop_metadata.append(metadata)
        
        # Final passage selection and ranking
        final_passages, final_scores = self.confidence_scorer.select_best_passages(
            all_scores, retrieved_passages, max_passages=10
        )
        
        # Calculate multi-objective final scores
        final_relevance = [s.relevance_score for s in final_scores]
        final_novelty = [s.information_novelty for s in final_scores]
        final_entity_coverage = [s.entity_coverage for s in final_scores]
        final_evidence = [s.evidence_strength for s in final_scores]
        
        # Determine LLM verification candidates
        llm_candidates = self.confidence_scorer.get_llm_verification_candidates(
            final_scores, final_passages
        )
        
        return {
            'query': query,
            'final_passages': final_passages,
            'confidence_scores': final_scores,
            'hop_count': len(hop_metadata),
            'predicted_hops': predicted_hops if adaptive_hops else max_hops,
            'multi_objective_scores': {
                'relevance': final_relevance,
                'novelty': final_novelty, 
                'entity_coverage': final_entity_coverage,
                'evidence_strength': final_evidence
            },
            'llm_verification_needed': len(llm_candidates) > 0,
            'llm_candidates': llm_candidates,
            'hop_metadata': hop_metadata,
            'average_confidence': np.mean([s.overall_score for s in final_scores]) if final_scores else 0.0
        }
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Use the encoder to get embeddings
        with torch.no_grad():
            embedding = self.encoder.encode([text], convert_to_tensor=True)
            if embedding.dim() == 2:
                embedding = embedding[0]  # Remove batch dimension
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.query_cache.clear()
        self.embedding_cache.clear() 