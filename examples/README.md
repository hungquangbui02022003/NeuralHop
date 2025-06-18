# TreeHop Examples

This directory contains usage examples for both original TreeHop and TreeHop Enhanced.

## Quick Start Examples

### Basic TreeHop Usage
```bash
python basic_usage.py
```
Demonstrates the original TreeHop functionality with multi-hop retrieval.

### Enhanced TreeHop Usage  
```bash
python enhanced_usage.py
```
Showcases advanced features including:
- Adaptive hop count determination
- Smart query preprocessing
- Multi-objective confidence scoring
- Performance optimization with caching

## Prerequisites

Before running examples, ensure you have:

1. **Installed dependencies:**
   ```bash
   # For basic usage
   pip install -r ../requirements.txt
   
   # For enhanced features
   pip install -r ../requirements_enhanced.txt
   ```

2. **Initialized datasets:**
   ```bash
   cd ..
   python init_multihop_rag.py
   python init_train_vectors.py
   ```

## Example Structure

- `basic_usage.py` - Original TreeHop demonstration
- `enhanced_usage.py` - TreeHop Enhanced showcase with all advanced features

## Running Examples

From the project root directory:
```bash
# Basic TreeHop
python examples/basic_usage.py

# Enhanced TreeHop  
python examples/enhanced_usage.py
```

## Customization

Both examples can be modified to:
- Use different datasets (2wiki, musique, multihop_rag)
- Adjust hop counts and retrieval parameters
- Test with custom queries
- Experiment with different models

See the source code for detailed configuration options. 