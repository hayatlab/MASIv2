# MASIv2

MASIv2: marker-assisted standardization and integration for multi-modal single-cell and spatial omics data

Manuscript: Annotating and integrating multi-modal single-cell and spatial omics data with one general framework

### 1. Brief description
MASIv2 is an extended version of MASI, which utlizes marker genes and graph structure to annotate and integrate multi-modal single-cell and spatial omics data.

![alt text](https://github.com/hayatlab/masiv2/blob/main/MASIv2/MASIv2_Figure1.jpg?raw=true)

### 2. Install requirement packages
    pip install scanpy networkx python-louvain scikit-learn scipy
    ##MASIv2 also requires dgl library.
    ##Please install dgl (https://www.dgl.ai/pages/start.html) according your system.
    
### 3. Usage
    import MASI as masi
    
### 4. Reproduce results in manuscript
Please see tutorials at https://github.com/hayatlab/masiv2/tree/main/tutorial

### 5. Citation
Xu et al. "Annotating and integrating multi-modal single-cell and spatial omics data with one general framework". <a href="https://www.biorxiv.org/content/10.1101/2022.03.28.486110v1">bioRxiv</a>
