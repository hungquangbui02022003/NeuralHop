"""
TreeHop Enhanced: Advanced Multi-Hop Retrieval with Intelligent Optimization

This module provides enhanced TreeHop functionality with:
- Smart query preprocessing and decomposition
- Adaptive hop count and confidence-based processing  
- Intelligent post-processing and selective LLM integration
- Performance optimizations for production use
"""

from .query_processor import SmartQueryProcessor
from .enhanced_model import TreeHopEnhanced
from .post_processor import IntelligentPostProcessor
from .confidence_scorer import ConfidenceScorer
from .adaptive_retriever import AdaptiveRetriever

__version__ = "2.0.0"
__all__ = [
    "SmartQueryProcessor",
    "TreeHopEnhanced", 
    "IntelligentPostProcessor",
    "ConfidenceScorer",
    "AdaptiveRetriever"
] 