"""
Enhanced Evaluation Script for TreeHop Enhanced

This script evaluates the enhanced TreeHop system with:
- Smart query preprocessing
- Adaptive hop count determination
- Intelligent post-processing
- Selective LLM integration
- Performance optimization
"""

import argparse
import json
import time
import torch
from typing import List, Dict, Any
import logging
from pathlib import Path

from tree_hop_enhanced import (
    SmartQueryProcessor, 
    TreeHopEnhanced, 
    IntelligentPostProcessor,
    ConfidenceScorer,
    AdaptiveRetriever
)
from src.bge_m3.model import BGE_M3
from metrics import Metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedEvaluationPipeline:
    """Complete evaluation pipeline for TreeHop Enhanced"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3",
                 enable_adaptive_hops: bool = True,
                 enable_llm_verification: bool = False,
                 cache_size: int = 10000,
                 max_workers: int = 4):
        
        logger.info("Initializing Enhanced TreeHop Evaluation Pipeline...")
        
        # Load encoder model
        logger.info(f"Loading encoder model: {model_name}")
        self.encoder = BGE_M3(model_name)
        
        # Configuration
        self.enable_adaptive_hops = enable_adaptive_hops
        self.enable_llm_verification = enable_llm_verification
        self.cache_size = cache_size
        self.max_workers = max_workers
        
        # Initialize metrics
        self.metrics = Metrics()
        
        logger.info("Enhanced TreeHop Evaluation Pipeline initialized successfully!")
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load evaluation dataset"""
        logger.info(f"Loading dataset from: {dataset_path}")
        
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} examples from dataset")
        return data
    
    def load_passages(self, passages_path: str) -> List[str]:
        """Load passage corpus"""
        logger.info(f"Loading passages from: {passages_path}")
        
        passages = []
        with open(passages_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # Parse JSONL format
                    try:
                        passage_data = json.loads(line.strip())
                        if 'text' in passage_data:
                            passages.append(passage_data['text'])
                        elif 'contents' in passage_data:
                            passages.append(passage_data['contents'])
                        else:
                            passages.append(line.strip())
                    except json.JSONDecodeError:
                        passages.append(line.strip())
        
        logger.info(f"Loaded {len(passages)} passages")
        return passages
    
    def evaluate_dataset(self, 
                        dataset: List[Dict],
                        passages: List[str],
                        output_path: str = None,
                        batch_size: int = 1) -> Dict[str, Any]:
        """Evaluate entire dataset with enhanced TreeHop"""
        
        logger.info(f"Starting evaluation on {len(dataset)} examples...")
        
        # Initialize adaptive retriever
        retriever = AdaptiveRetriever(
            encoder_model=self.encoder,
            passages=passages,
            max_workers=self.max_workers,
            cache_size=self.cache_size,
            enable_parallel=True
        )
        
        # Track results
        all_results = []
        performance_metrics = {
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hit_ratio': 0.0,
            'average_hop_count': 0.0,
            'llm_verification_usage': 0.0
        }
        
        # Process in batches for efficiency
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_queries = [example['question'] for example in batch]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
            
            # Batch retrieval
            if len(batch_queries) > 1:
                batch_results = retriever.batch_retrieve(
                    queries=batch_queries,
                    max_passages=10,
                    use_cache=True,
                    adaptive_hops=self.enable_adaptive_hops,
                    use_llm_verification=self.enable_llm_verification,
                    parallel=True
                )
                
                # Process batch results
                for j, result in enumerate(batch_results.results):
                    example = batch[j]
                    enhanced_result = self._process_single_result(example, result)
                    all_results.append(enhanced_result)
                
                # Update performance metrics
                performance_metrics['total_processing_time'] += batch_results.total_processing_time
                performance_metrics['cache_hit_ratio'] = (
                    performance_metrics['cache_hit_ratio'] * i + 
                    batch_results.cache_hit_ratio * len(batch)
                ) / (i + len(batch))
                
            else:
                # Single query
                example = batch[0]
                result = retriever.retrieve(
                    query=example['question'],
                    max_passages=10,
                    use_cache=True,
                    adaptive_hops=self.enable_adaptive_hops,
                    use_llm_verification=self.enable_llm_verification
                )
                
                enhanced_result = self._process_single_result(example, result)
                all_results.append(enhanced_result)
                
                performance_metrics['total_processing_time'] += result.processing_time
        
        # Calculate final metrics
        performance_metrics['average_processing_time'] = (
            performance_metrics['total_processing_time'] / len(dataset)
        )
        
        performance_metrics['average_hop_count'] = sum(
            r['hop_count'] for r in all_results
        ) / len(all_results)
        
        performance_metrics['llm_verification_usage'] = sum(
            1 for r in all_results if r.get('used_llm_verification', False)
        ) / len(all_results)
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(all_results)
        
        # Combine all metrics
        final_metrics = {
            **performance_metrics,
            **accuracy_metrics,
            'total_examples': len(dataset),
            'retriever_stats': retriever.get_performance_stats()
        }
        
        # Save results if output path provided
        if output_path:
            self._save_results(all_results, final_metrics, output_path)
        
        logger.info("Evaluation completed successfully!")
        return final_metrics
    
    def _process_single_result(self, example: Dict, result) -> Dict:
        """Process single retrieval result with ground truth comparison"""
        
        # Extract ground truth
        ground_truth_answers = example.get('answers', [])
        if isinstance(ground_truth_answers, str):
            ground_truth_answers = [ground_truth_answers]
        
        # Get retrieved passages
        retrieved_passages = result.passages
        
        # Calculate F1 score using the existing metrics
        f1_scores = []
        for gt_answer in ground_truth_answers:
            # Simple F1 calculation (this could be enhanced with better answer generation)
            predicted_answer = " ".join(retrieved_passages[:3])  # Use top 3 passages as answer
            f1 = self.metrics.f1_score(predicted_answer, gt_answer)
            f1_scores.append(f1)
        
        best_f1 = max(f1_scores) if f1_scores else 0.0
        
        return {
            'question': example['question'],
            'ground_truth_answers': ground_truth_answers,
            'retrieved_passages': retrieved_passages,
            'passage_scores': result.scores,
            'confidence_scores': [
                {
                    'overall': score.overall_score,
                    'relevance': score.relevance_score,
                    'novelty': score.information_novelty,
                    'entity_coverage': score.entity_coverage,
                    'evidence_strength': score.evidence_strength,
                    'confidence_level': score.confidence_level
                } for score in result.confidence_scores
            ],
            'processing_time': result.processing_time,
            'hop_count': result.hop_count,
            'cache_hits': result.cache_hits,
            'f1_score': best_f1,
            'used_llm_verification': result.metadata.get('processed_result', {}).get('llm_verification_results') is not None,
            'metadata': result.metadata
        }
    
    def _calculate_accuracy_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy and other quality metrics"""
        
        f1_scores = [r['f1_score'] for r in results]
        
        # Basic accuracy metrics
        accuracy_metrics = {
            'average_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'median_f1': sorted(f1_scores)[len(f1_scores)//2] if f1_scores else 0.0,
            'max_f1': max(f1_scores) if f1_scores else 0.0,
            'min_f1': min(f1_scores) if f1_scores else 0.0
        }
        
        # Confidence correlation metrics
        avg_confidence_scores = []
        for result in results:
            if result['confidence_scores']:
                avg_conf = sum(cs['overall'] for cs in result['confidence_scores']) / len(result['confidence_scores'])
                avg_confidence_scores.append(avg_conf)
            else:
                avg_confidence_scores.append(0.0)
        
        # Calculate correlation between confidence and F1
        if len(avg_confidence_scores) == len(f1_scores) and len(f1_scores) > 1:
            try:
                import numpy as np
                correlation = np.corrcoef(avg_confidence_scores, f1_scores)[0, 1]
                accuracy_metrics['confidence_f1_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            except:
                accuracy_metrics['confidence_f1_correlation'] = 0.0
        else:
            accuracy_metrics['confidence_f1_correlation'] = 0.0
        
        return accuracy_metrics
    
    def _save_results(self, results: List[Dict], metrics: Dict, output_path: str):
        """Save evaluation results to file"""
        
        output_data = {
            'evaluation_metrics': metrics,
            'detailed_results': results,
            'configuration': {
                'enable_adaptive_hops': self.enable_adaptive_hops,
                'enable_llm_verification': self.enable_llm_verification,
                'cache_size': self.cache_size,
                'max_workers': self.max_workers
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save as JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Also save a summary
        summary_file = output_file.with_suffix('.summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to: {summary_file}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Enhanced TreeHop Evaluation')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to evaluation dataset (JSONL format)')
    parser.add_argument('--passages', type=str, required=True,
                       help='Path to passage corpus (JSONL format)')
    parser.add_argument('--output', type=str, default='results/enhanced_evaluation.json',
                       help='Output path for results')
    parser.add_argument('--model', type=str, default='BAAI/bge-m3',
                       help='Encoder model name')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--disable-adaptive-hops', action='store_true',
                       help='Disable adaptive hop count')
    parser.add_argument('--enable-llm-verification', action='store_true',
                       help='Enable LLM verification (slower but potentially more accurate)')
    parser.add_argument('--cache-size', type=int, default=10000,
                       help='Cache size for performance optimization')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads')
    
    args = parser.parse_args()
    
    # Initialize evaluation pipeline
    pipeline = EnhancedEvaluationPipeline(
        model_name=args.model,
        enable_adaptive_hops=not args.disable_adaptive_hops,
        enable_llm_verification=args.enable_llm_verification,
        cache_size=args.cache_size,
        max_workers=args.max_workers
    )
    
    # Load data
    dataset = pipeline.load_dataset(args.dataset)
    passages = pipeline.load_passages(args.passages)
    
    # Run evaluation
    results = pipeline.evaluate_dataset(
        dataset=dataset,
        passages=passages,
        output_path=args.output,
        batch_size=args.batch_size
    )
    
    # Print summary
    print("\n" + "="*50)
    print("ENHANCED TREEHOP EVALUATION RESULTS")
    print("="*50)
    print(f"Total Examples: {results['total_examples']}")
    print(f"Average F1 Score: {results['average_f1']:.4f}")
    print(f"Average Processing Time: {results['average_processing_time']:.4f}s")
    print(f"Average Hop Count: {results['average_hop_count']:.2f}")
    print(f"Cache Hit Ratio: {results['cache_hit_ratio']:.4f}")
    print(f"LLM Verification Usage: {results['llm_verification_usage']:.4f}")
    print(f"Confidence-F1 Correlation: {results['confidence_f1_correlation']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main() 