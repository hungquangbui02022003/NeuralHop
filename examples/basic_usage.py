"""
TreeHop - Basic Usage Example
Demonstrates original TreeHop functionality with multi-hop retrieval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_hop import TreeHopModel
from passage_retrieval import MultiHopRetriever
import json

def main():
    """Simple TreeHop demo for quick testing"""
    
    EVALUATE_DATASET = "multihop_rag"
    
    # Check if dataset exists
    passages_file = f"embedding_data/{EVALUATE_DATASET}/eval_passages.jsonl"
    if not os.path.exists(passages_file):
        print(f"Error: Dataset not found at {passages_file}")
        print("Please run 'python init_multihop_rag.py' first to initialize the dataset")
        return
    
    print("Loading TreeHop model...")
    
    try:
        # load TreeHop model from HuggingFace
        tree_hop_model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")
        
        # load retriever
        retriever = MultiHopRetriever(
            "BAAI/bge-m3",
            passages=f"embedding_data/{EVALUATE_DATASET}/eval_passages.jsonl",
            passage_embeddings=f"embedding_data/{EVALUATE_DATASET}/eval_content_dense.npy",
            tree_hop_model=tree_hop_model,
            projection_size=1024,
            save_or_load_index=True,
            indexing_batch_size=10240,
            index_device="cuda"
        )
        
        print("TreeHop model loaded successfully!")
        
        # Example query
        query = "Did Engadget report a discount on the 13.6-inch MacBook Air before The Verge reported a discount on Samsung Galaxy Buds 2?"
        
        print(f"\nQuery: {query}")
        print("Performing multi-hop retrieval...")
        
        # Perform retrieval
        result = retriever.multihop_search_passages(
            query,
            n_hop=2,
            top_n=5
        )
        
        print(f"\nRetrieved {len(result.passage)} passages:")
        for i, passage in enumerate(result.passage[:3]):  # Show top 3
            print(f"{i+1}. {passage['title']}: {passage['text'][:100]}...")
            
        print("\nTreeHop demo completed successfully!")
        
    except Exception as e:
        print(f"Error running TreeHop demo: {e}")
        print("Make sure you have initialized the dataset and dependencies")

if __name__ == "__main__":
    main() 