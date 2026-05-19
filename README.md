# Hybrid Information Retrieval System with Reciprocal Rank Fusion (RRF)

This repository contains the implementation and evaluation of a Hybrid Information Retrieval (IR) system. The project focuses on bridging the gap between exact keyword matching and deep semantic understanding by fusing **BM25 lexical retrieval** with **Dense semantic retrieval** across various $k$-value thresholds up to $k=100$.

---

## 📊 Experimental Results & Key Insights

Our evaluation compares the BM25 baseline against two dense retrieval encoders (**all-MiniLM-L6-v2** and **all-mpnet-base-v2**), as well as their respective hybrid combinations using Reciprocal Rank Fusion (RRF).

### Performance Metrics Comparison

| Model Combination | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Baseline** | 0.347 | 0.627 | 0.693 | 0.787 | *Stable* |
| **Hybrid (BM25 + MiniLM)** | **0.587** | 0.707 | 0.827 | 0.893 | *High* |
| **Hybrid (BM25 + MPNet)** | 0.573 | **0.707** | **0.840** | **0.907** | **Highest** |

### 🔍 Key Findings

1. **Precision vs. Generalization (Dense Encoders):** 
   Our analysis confirms that while the lighter **MiniLM** demonstrates slightly higher precision at the very top rank ($k=1$), **MPNet** exhibits superior semantic capture as $k$ increases. MPNet's retrieval capability scales more robustly with a higher document budget, eventually outperforming MiniLM at $k=20$ and $k=100$.
2. **The Hybrid Advantage:** 
   Regardless of the underlying dense model used, **Hybrid Retrieval (RRF) consistently outpaces any single baseline**. RRF effectively leverages the mutual exclusivity of keyword and semantic errors, using early lexical signals to anchor precision while allowing dense models to maximize long-tail recall.
3. **Efficiency Trade-off:** 
   In resource-constrained settings, the `BM25 + MiniLM` configuration yields highly competitive results close to the `MPNet` variant, offering a highly cost-effective alternative for real-time deployment.

---

## 🛠️ Project Structure

* `evaluation.ipynb` — Core Jupyter Notebook containing data loading, RRF fusion logic, and evaluation metrics.
* `plots/` — Generated evaluation curves showcasing performance trajectories up to $k=100$.
* `data/` *(Not tracked)* — Contains raw retrieval rankings (`bm25_rankings_full.csv`, `dense_results_minilm.csv`, `dense_results_mpnet.csv`) and standard annotations (`qrels`).

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the required libraries:

```bash
pip install numpy pandas matplotlib
