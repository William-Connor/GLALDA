# GLALDA: Multi-scale Graph-level Attention for Predicting lncRNA-Disease Associations

GLALDA is a graph neural network (GNN) prediction framework designed to accurately predict potential associations between long non-coding RNAs (lncRNAs) and diseases. It achieves this by integrating multi-scale graph information, edge feature data, and a dual-level attention mechanism that operates at both the subgraph and global levels.

---
## Data

The dataset used for training and evaluating GLALDA primarily derives from established benchmark data.

### Dataset Statistics

The heterogeneous network is constructed using the following entities and known associations:

* **Nodes**:

  * **lncRNAs**: 240 

  * **Diseases**: 412 

  * **miRNAs**: 495 




* **Associations**:

  * **lncRNA-Disease**: 2,697 associations 

  * **lncRNA-miRNA**: 1,002 associations 

  * **miRNA-Disease**: 13,562 associations 





### Primary Data Sources

The raw biological data was integrated from several curated databases:


- **Lnc2Cancer**: Experimentally supported lncRNA-cancer associations.



- **LncRNADisease**: A database for long non-coding RNA-associated diseases.



- **GeneRIF**: Functional annotation of gene functions.



- **starBase v2.0**: Decoding miRNA-ceRNA and miRNA-ncRNA interaction networks.



- **HMDD v2.0**: Experimentally supported human microRNA and disease associations.


---

## Core Contributions


- **Dual-level Attention Mechanism**: Combines a subgraph-level attention mechanism to capture local structural features and interactions with a global graph-level attention mechanism for deep feature information.


 
- **Multi-scale Feature Fusion**: Employs a cross-attention mechanism to deeply fuse feature representations from graphs of different scales, effectively combining local node details with global context.



- **Edge Feature Integration**: Incorporates numerical link strength between nodes as edge features in the attention calculations to better capture complex graph structures and dependencies.



- **Multi-similarity Fusion Strategy**: Integrates functional, semantic, cosine, and kernel similarity measures. It utilizes K-Nearest Neighbors (KNN) and power normalization to emphasize significant relationships while reducing noise and computational burden.



---

## Model Architecture

The GLALDA workflow consists of three primary steps:

1. **Multi-similarity Fusion**: Multiple similarity matrices are standardized, fused, and enhanced to make high-similarity connections more prominent.


2. **Heterogeneous Network Construction**: The model builds a multi-level global scale graph of lncRNA-disease-miRNA and dual-factor local scale graphs for lncRNA-disease and disease-miRNA.


3. **Feature Extraction and Prediction**:

   - **Positional Encoding**: Uses the Laplacian spectrum as positional encoding to encapsulate the global graph structure.

   - **Attention Learning**: Parallel feature extraction is performed using subgraph-level and global graph-level attention mechanisms.

   - **Cross-Attention Fusion**: A cross-attention mechanism integrates key information from different network levels and perspectives.

   - **Scoring**: The resulting comprehensive feature representations are fed into a scoring network (classifier) using a sigmoid activation function to evaluate the probability of association.






---

### Installation & Code

> 
> **[https://github.com/William-Connor/GLALDA](https://github.com/William-Connor/GLALDA)**





