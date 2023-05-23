# MASIv2

MASIv2: marker-assisted standardization and integration for multi-modal single-cell and spatial omics data

Manuscript: MASIv2 enables standardization and integration of multi-modal single-cell and spatial omics data with one general framework

### 1. Brief description
MASIv2 is an extended version of MASI, which utlizes marker genes and graph structure to annotate and integrate multi-modal single-cell and spatial omics data.

![alt text](https://github.com/hayatlab/masiv2/blob/main/MASIv2/MASIv2_Figure1.jpg?raw=true)

### 2. Install requirement packages
    pip install scanpy networkx python-louvain scikit-learn scipy
    ##MASIv2 also requires dgl library.
    ##Please install dgl according to your system (https://www.dgl.ai/pages/start.html).
    
### 3. Usage
    import MASIv2 as masi
    
### 4. Reproduce results in manuscript
Please see tutorials at https://github.com/hayatlab/MASIv2/tree/main/tutorial

### 5. Citation
Xu et al. "MASIv2 enables standardization and integration of multi-modal single-cell and spatial omics data with one general framework". <a href="https://www.biorxiv.org/content/10.1101/2023.05.15.540808v1">bioRxiv</a>
