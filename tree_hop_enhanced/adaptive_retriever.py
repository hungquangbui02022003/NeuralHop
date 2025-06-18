"""
Adaptive Retriever for TreeHop Enhanced

Features:
- Performance optimization with caching
- Batch processing for multiple queries
- Parallel retrieval for independent hops
- Adaptive indexing and memory management
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import faiss
import hashlib
from dataclasses import dataclass, field

from .query_processor import SmartQueryProcessor, QueryAnalysis
from .enhanced_model import TreeHopEnhanced
from .post_processor import IntelligentPostProcessor, ProcessedResult
from .confidence_scorer import ConfidenceScorer

@dataclass
class RetrievalResult:
    """Results from adaptive retrieval"""
    query: str
    passages: List[str]
    scores: List[float]
    confidence_scores: List[any]
    processing_time: float
    hop_count: int
    cache_hits: int
    metadata: Dict = field(default_factory=dict)

@dataclass
class BatchRetrievalResult:
    """Results from batch retrieval"""
    results: List[RetrievalResult]
    total_processing_time: float
    average_processing_time: float
    cache_hit_ratio: float
    metadata: Dict = field(default_factory=dict)

class PerformanceCache:
    """High-performance caching system for embeddings and results"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._embedding_cache = {}
        self._result_cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
        
    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        with self._lock:
            if text in self._embedding_cache:
                current_time = time.time()
                if current_time - self._access_times[text] < self.ttl:
                    self._access_times[text] = current_time
                    return self._embedding_cache[text]
                else:
                    # Expired, remove from cache
                    del self._embedding_cache[text]
                    del self._access_times[text]
        return None
    
    def set_embedding(self, text: str, embedding: torch.Tensor):
        """Cache embedding"""
        with self._lock:
            if len(self._embedding_cache) >= self.max_size:
                self._evict_lru()
            
            self._embedding_cache[text] = embedding
            self._access_times[text] = time.time()
    
    def get_result(self, query_hash: str) -> Optional[RetrievalResult]:
        """Get cached result"""
        with self._lock:
            if query_hash in self._result_cache:
                current_time = time.time()
                if current_time - self._access_times[query_hash] < self.ttl:
                    self._access_times[query_hash] = current_time
                    return self._result_cache[query_hash]
                else:
                    del self._result_cache[query_hash]
                    del self._access_times[query_hash]
        return None
    
    def set_result(self, query_hash: str, result: RetrievalResult):
        """Cache result"""
        with self._lock:
            if len(self._result_cache) >= self.max_size:
                self._evict_lru()
            
            self._result_cache[query_hash] = result
            self._access_times[query_hash] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used items"""
        if not self._access_times:
            return
        
        # Find oldest access time
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from all caches
        if oldest_key in self._embedding_cache:
            del self._embedding_cache[oldest_key]
        if oldest_key in self._result_cache:
            del self._result_cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self):
        """Clear all caches"""
        with self._lock:
            self._embedding_cache.clear()
            self._result_cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'embedding_cache_size': len(self._embedding_cache),
                'result_cache_size': len(self._result_cache),
                'total_cache_size': len(self._access_times),
                'max_size': self.max_size
            }

class AdaptiveRetriever:
    """High-performance adaptive retriever with optimization features"""
    
    def __init__(self, 
                 encoder_model,
                 passages: List[str],
                 max_workers: int = 4,
                 cache_size: int = 10000,
                 enable_parallel: bool = True):
        
        # Core components
        self.query_processor = SmartQueryProcessor()
        self.enhanced_model = TreeHopEnhanced(encoder_model)
        self.post_processor = IntelligentPostProcessor()
        self.confidence_scorer = ConfidenceScorer()
        
        # Performance components
        self.cache = PerformanceCache(max_size=cache_size)
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        
        # Load passages
        self.passages = passages
        
        # Pre-compute passage embeddings if not cached
        self.passage_embeddings = self._get_or_compute_passage_embeddings()
        
        # Performance tracking
        self.retrieval_stats = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'average_hops': 0.0
        }
    
    def retrieve(self, 
                query: str,
                max_passages: int = 10,
                use_cache: bool = True,
                adaptive_hops: bool = True,
                use_llm_verification: bool = False) -> RetrievalResult:
        """
        Single query retrieval with full optimization
        """
        start_time = time.time()
        cache_hits = 0
        
        # Generate query hash for caching
        query_hash = self._generate_query_hash(query, max_passages, adaptive_hops)
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get_result(query_hash)
            if cached_result is not None:
                cache_hits = 1
                self._update_stats(time.time() - start_time, cache_hits, cached_result.hop_count)
                return cached_result
        
        # Analyze query
        query_analysis = self.query_processor.analyze_query(query)
        
        # Get adaptive hop count
        if adaptive_hops:
            max_hops = self.query_processor.get_adaptive_hop_count(query_analysis)
        else:
            max_hops = 3
        
        # Enhanced TreeHop retrieval
        treehop_result = self.enhanced_model.forward(
            query=query_analysis.optimized_query,
            passages=self.passages,
            passage_embeddings=self.passage_embeddings,
            query_entities=query_analysis.entities,
            adaptive_hops=adaptive_hops
        )
        
        # Post-processing
        processed_result = self.post_processor.process_results(
            passages=treehop_result['final_passages'],
            confidence_scores=treehop_result['confidence_scores'],
            passage_embeddings=self._get_passage_embeddings_for_texts(treehop_result['final_passages']),
            query=query,
            use_llm_verification=use_llm_verification
        )
        
        # Create final result
        processing_time = time.time() - start_time
        
        result = RetrievalResult(
            query=query,
            passages=processed_result.deduplicated_passages[:max_passages],
            scores=[score.overall_score for score in treehop_result['confidence_scores'][:max_passages]],
            confidence_scores=treehop_result['confidence_scores'][:max_passages],
            processing_time=processing_time,
            hop_count=treehop_result['hop_count'],
            cache_hits=cache_hits,
            metadata={
                'query_analysis': query_analysis,
                'treehop_result': treehop_result,
                'processed_result': processed_result,
                'adaptive_hops_used': max_hops
            }
        )
        
        # Cache result
        if use_cache:
            self.cache.set_result(query_hash, result)
        
        # Update stats
        self._update_stats(processing_time, cache_hits, treehop_result['hop_count'])
        
        return result
    
    def batch_retrieve(self, 
                      queries: List[str],
                      max_passages: int = 10,
                      use_cache: bool = True,
                      adaptive_hops: bool = True,
                      use_llm_verification: bool = False,
                      parallel: bool = None) -> BatchRetrievalResult:
        """
        Batch retrieval with parallel processing
        """
        if parallel is None:
            parallel = self.enable_parallel
        
        start_time = time.time()
        results = []
        
        if parallel and len(queries) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_query = {
                    executor.submit(
                        self.retrieve, 
                        query, 
                        max_passages, 
                        use_cache, 
                        adaptive_hops, 
                        use_llm_verification
                    ): query for query in queries
                }
                
                for future in as_completed(future_to_query):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # Create error result
                        query = future_to_query[future]
                        error_result = RetrievalResult(
                            query=query,
                            passages=[],
                            scores=[],
                            confidence_scores=[],
                            processing_time=0.0,
                            hop_count=0,
                            cache_hits=0,
                            metadata={'error': str(e)}
                        )
                        results.append(error_result)
        else:
            # Sequential processing
            for query in queries:
                result = self.retrieve(
                    query, max_passages, use_cache, adaptive_hops, use_llm_verification
                )
                results.append(result)
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        average_time = total_time / len(queries) if queries else 0.0
        total_cache_hits = sum(r.cache_hits for r in results)
        cache_hit_ratio = total_cache_hits / len(queries) if queries else 0.0
        
        return BatchRetrievalResult(
            results=results,
            total_processing_time=total_time,
            average_processing_time=average_time,
            cache_hit_ratio=cache_hit_ratio,
            metadata={
                'parallel_processing': parallel,
                'query_count': len(queries),
                'successful_retrievals': sum(1 for r in results if not r.metadata.get('error')),
                'total_hops': sum(r.hop_count for r in results),
                'cache_stats': self.cache.get_stats()
            }
        )
    
    def _get_or_compute_passage_embeddings(self) -> torch.Tensor:
        """Get or compute passage embeddings with caching"""
        embeddings = []
        
        for passage in self.passages:
            cached_emb = self.cache.get_embedding(passage)
            if cached_emb is not None:
                embeddings.append(cached_emb)
            else:
                # Compute embedding
                emb = self.enhanced_model._encode_text(passage)
                self.cache.set_embedding(passage, emb)
                embeddings.append(emb)
        
        return torch.stack(embeddings)
    
    def _get_passage_embeddings_for_texts(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for specific passage texts"""
        embeddings = []
        
        for text in texts:
            # Find in original passages
            try:
                idx = self.passages.index(text)
                embeddings.append(self.passage_embeddings[idx])
            except ValueError:
                # Compute new embedding if not found
                emb = self.enhanced_model._encode_text(text)
                embeddings.append(emb)
        
        return torch.stack(embeddings) if embeddings else torch.empty(0, self.enhanced_model.input_dim)
    
    def _generate_query_hash(self, query: str, max_passages: int, adaptive_hops: bool) -> str:
        """Generate hash for query caching"""
        content = f"{query}_{max_passages}_{adaptive_hops}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_stats(self, processing_time: float, cache_hits: int, hop_count: int):
        """Update retrieval statistics"""
        self.retrieval_stats['total_retrievals'] += 1
        self.retrieval_stats['cache_hits'] += cache_hits
        self.retrieval_stats['total_time'] += processing_time
        
        # Update average hops
        total_retrievals = self.retrieval_stats['total_retrievals']
        current_avg = self.retrieval_stats['average_hops']
        self.retrieval_stats['average_hops'] = (current_avg * (total_retrievals - 1) + hop_count) / total_retrievals
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        stats = self.retrieval_stats.copy()
        
        if stats['total_retrievals'] > 0:
            stats['cache_hit_ratio'] = stats['cache_hits'] / stats['total_retrievals']
            stats['average_processing_time'] = stats['total_time'] / stats['total_retrievals']
        else:
            stats['cache_hit_ratio'] = 0.0
            stats['average_processing_time'] = 0.0
        
        stats['cache_stats'] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        self.enhanced_model.clear_cache()
    
    def optimize_performance(self):
        """Optimize for better performance"""
        # Pre-warm cache with common queries if available
        # Optimize tensor operations
        if torch.cuda.is_available():
            self.passage_embeddings = self.passage_embeddings.cuda()
    
    async def async_retrieve(self, 
                           query: str,
                           max_passages: int = 10,
                           use_cache: bool = True,
                           adaptive_hops: bool = True,
                           use_llm_verification: bool = False) -> RetrievalResult:
        """Async version of retrieve for non-blocking calls"""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            None,
            self.retrieve,
            query,
            max_passages,
            use_cache,
            adaptive_hops,
            use_llm_verification
        )
        
        return result 