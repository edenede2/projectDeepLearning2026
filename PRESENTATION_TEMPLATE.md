# GraphSAGE: Inductive Representation Learning on Large Graphs
## Deep Learning Final Project Presentation

---

## Slide 1: Title Slide

**GraphSAGE Implementation for Node Classification**

*Re-implementation of Hamilton et al. (2017)*

- **Student Name:** [Your Name]
- **Course:** Deep Learning
- **Date:** [Presentation Date]

---

## Slide 2: Problem Statement

### What is Node Classification?
- Given a graph with nodes and edges
- Some nodes have labels (training data)
- Goal: Predict labels for unseen nodes

### Challenges
- Traditional methods require full graph at training time
- No generalization to new nodes (transductive only)
- Scalability issues with large graphs

---

## Slide 3: GraphSAGE Innovation

### Key Contribution: Inductive Learning
- **Before GraphSAGE:** Node embeddings fixed at training time
- **GraphSAGE:** Learn to *generate* embeddings from local neighborhoods

### The "SAGE" Approach
- **S**ample: Fixed-size neighborhood sampling
- **A**ggregate: Combine neighbor information
- **GE**nerate: Create node embeddings

---

## Slide 4: Architecture Overview

```
Input Node → Sample K-hop Neighbors → Aggregate → Transform → Output
```

### Key Components
1. **Neighborhood Sampling:** K=2 hops, S₁=25, S₂=10 neighbors
2. **Aggregator Function:** Mean pooling (also: LSTM, pool, GCN)
3. **Multi-layer GNN:** 2 layers with ReLU and Dropout
4. **Classification Head:** Linear layer + Softmax/Sigmoid

---

## Slide 5: Algorithm Walkthrough

### Forward Pass (per node v)
```
1. h⁰ᵥ = xᵥ (initial features)
2. For k = 1 to K:
   a. Sample neighbors N(v) 
   b. h̃ᵏₙ = AGGREGATE({hᵏ⁻¹ᵤ : u ∈ N(v)})
   c. hᵏᵥ = σ(W · CONCAT(hᵏ⁻¹ᵥ, h̃ᵏₙ))
3. Output: zᵥ = hᴷᵥ (normalized)
```

### Loss Function
- **Multi-class:** Cross-Entropy Loss
- **Multi-label:** Binary Cross-Entropy with Logits

---

## Slide 6: Datasets

| Dataset | Nodes | Edges | Features | Classes | Task |
|---------|-------|-------|----------|---------|------|
| **Cora** | 2,708 | 10,556 | 1,433 | 7 | Citation classification (sanity check) |
| **PPI** | 56,944 | 818,716 | 50 | 121 | Protein function prediction |
| **Reddit** | 232,965 | 114.6M | 602 | 41 | Community classification |

### ⚠️ Important: Paper's Citation Dataset
The original paper used **Web of Science**, NOT Cora.
- Cora is included as a **sanity check only**
- We do NOT compare Cora results to paper

---

## Slide 7: Training Protocols

### Per-Dataset Protocol

| Dataset | Protocol | Description |
|---------|----------|-------------|
| **Cora** | Transductive | Full-graph message passing; loss on train_mask only. **Sanity check only** |
| **Reddit** | Inductive (mini-batch) | NeighborLoader on FULL graph; input_nodes=train_mask |
| **PPI** | Inductive (multi-graph) | Separate train/val/test graphs |

### Key Implementation Details
- ❌ Do NOT train on induced train-only subgraph for Reddit
- ✅ Use NeighborLoader on FULL graph with train_mask as input_nodes
- ✅ Loss computed only on seed nodes (first batch_size per subgraph)

---

## Slide 8: Implementation Details

### Model Architecture
```python
GraphSAGE(
  SAGEConv(in_features, 256) + ReLU + Dropout(0.5)
  SAGEConv(256, num_classes)
)
```

### Paper-Faithful Settings
- **BatchNorm:** OFF (paper did not use it)
- **Aggregator:** Mean (GraphSAGE-mean)
- **Neighbors:** [25, 10] per hop
- **Optimizer:** Adam (lr=0.01, weight_decay=5e-4)
- **Early stopping:** patience=20

---

## Slide 9: Results Comparison

### Paper-Comparable Results Only

| Dataset | Our F1 | Paper F1 | Difference | Status |
|---------|--------|----------|------------|--------|
| **PPI** | **72.59%** | 59.8% | **+12.79%** | Exceeded paper 🎉 |
| **Reddit** | *Not executed* | 95.0% | — | Requires GPU ⚠️ |

### Cora (Sanity Check Only - NOT Comparable to Paper)

| Dataset | Our F1 | Notes |
|---------|--------|-------|
| **Cora** | **79.90%** | Sanity check only; paper used Web of Science |

---

## Slide 10: Training Curves

### Observations
- **Cora**: Quick convergence (27 epochs), early stopping at F1=79.90%
- **PPI**: Steady improvement over 100 epochs, F1=72.59%
- **Reddit**: Not executed (CPU too slow - requires GPU)

[Insert training_curves.png visualization]

---

## Slide 11: Why Did PPI Exceed Paper Results?

### PPI Ablation Study Results
Our F1: **72.59%** vs Paper: **59.8%** = **+12.79%** improvement

### Factor Analysis
| Factor | Impact | Finding |
|--------|--------|---------|
| **Modern PyG** | +13.78% | SAGEConv optimizations, training pipeline |
| **Metric Style** | ~0.4% | Per-graph vs global F1 minimal difference |
| **BatchNorm OFF** | +0.42% | Slightly better without BN |
| **Learning Rate** | 0.01 best | LR=0.01 optimal (0.7396 F1) |

### Conclusion
- Modern PyG implementation is the **primary driver** of improvement
- Training regime and architectural details are secondary factors

---

## Slide 12: Methodology Improvements

### What We Fixed from Initial Implementation
1. **Reddit Protocol:** Changed from induced subgraph to full-graph NeighborLoader
2. **Cora Comparison:** Removed incorrect paper comparison (Web of Science ≠ Cora)
3. **Multi-seed Runs:** Added mean±std reporting for reproducibility
4. **Centralized Config:** All hyperparameters in one dict
5. **Seed Setting:** torch, numpy, random all seeded

### Sanity Checks Implemented
- ✅ Random label test (F1 should drop to chance)
- ✅ Overfit small batch test (should reach ~100%)

---

## Slide 13: Code Structure

```
projectDeepLearning2026/
├── GraphSAGE_Project.ipynb  # Main notebook (updated)
├── PRESENTATION_TEMPLATE.md  # This file (corrected)
├── data/                     # Downloaded datasets
│   ├── Cora/
│   ├── PPI/
│   └── Reddit/
├── checkpoints/              # Saved models + results
│   ├── graphsage_cora.pt
│   ├── graphsage_ppi.pt
│   ├── graphsage_reddit.pt
│   └── all_results.json      # Comprehensive results dict
├── results_comparison.png    # Bar chart (paper-comparable only)
└── training_curves.png       # Loss/F1 curves
```

---

## Slide 14: Key Takeaways

1. **GraphSAGE enables inductive learning** on graphs
2. **Paper's citation benchmark is Web of Science**, not Cora
3. **Reddit requires full-graph NeighborLoader** (not induced subgraph)
4. **Our PPI results exceed paper** by +12.79% F1 (72.59% vs 59.8%)
5. **Modern PyG implementation** is the main driver of improvement
6. **Reproducibility matters:** multi-seed experiments, comprehensive logging

---

## Slide 15: References

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). 
   *Inductive Representation Learning on Large Graphs*. 
   NeurIPS 2017.

2. PyTorch Geometric Documentation:
   https://pytorch-geometric.readthedocs.io/

3. Original GraphSAGE Code:
   https://github.com/williamleif/GraphSAGE

---

## Slide 16: Q&A

**Questions?**

### Key Points to Remember
- Paper's Citation = Web of Science, NOT Cora
- Reddit uses full-graph NeighborLoader  
- PPI exceeded paper by +12.79% (72.59% vs 59.8%)
- Main factor: Modern PyG implementation improvements

---

## Appendix: Technical Notes

### Environment
- Python 3.10+
- PyTorch 2.x
- PyTorch Geometric 2.x
- CUDA (optional)

### Reproducibility
- Seeds: [42, 123, 456] for multi-run
- All results from checkpoints/all_results.json
- Plots generated programmatically from results dict

### Methodology Clarifications
1. Cora: Transductive, sanity check only, NOT comparable to paper
2. Reddit: Inductive mini-batch on FULL graph (paper-faithful)
3. PPI: Inductive multi-graph (paper-faithful)
4. BatchNorm: OFF by default (paper didn't use it)
5. Threshold for PPI: logits > 0 (equiv. sigmoid > 0.5)

---

*End of Presentation Template*
