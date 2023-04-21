# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:31:17 2020

@author: Yang Xu
"""

import gc
import scipy
import community
import collections

import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer

import warnings
warnings.filterwarnings("ignore")

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.base import DGLError
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax

from torchcontrib.optim import SWA

##-----------------------------------------------------------------------------
##main functions
def gene2cell(ad=None, cell_markers=None,use_weight=False,thresh=0.5,
              if_tfidf=True,if_thresh=True,use_knn=False):
    ##TF-IDF transformation
    X = ad.X.copy()
    if scipy.sparse.issparse(X):
        X = X.todense()
    
    if if_tfidf:
        tf_transformer = TfidfTransformer(use_idf=True).fit(X)
        X= tf_transformer.transform(X).todense()
    
    labels = pd.DataFrame(np.zeros((X.shape[0],len(cell_markers))))
    labels.columns = cell_markers.keys()
    exprsed = pd.DataFrame(np.zeros((X.shape[0],len(cell_markers))))
    exprsed.columns = cell_markers.keys()
    celltype_size = {}
    
    ##create artifical labels for each cell
    if use_weight == True:
        for k, v in cell_markers.items():
            celltype_size[k]=0
            sums=0
            n = np.zeros((X.shape[0]))
            marker_index = -1
            for i in v:
                marker_index += 1
                if i in ad.var.index:
                    if if_thresh:
                        expr95 = np.percentile(X[:,ad.var.index == i],95)
                        thresh = thresh * expr95
                        l = np.array(X[:,ad.var.index == i])
                        l[X[:,ad.var.index == i]<=thresh]=0
                    else:
                        l = np.array(X[:,ad.var.index == i])
                    ##consider marker weight
                    l = l*(1-marker_index/(len(v)*2))##default 2
                    
                    n[np.array(l>0).reshape(X.shape[0])] += 1
                    sums += 1
                    labels[k] += l.reshape(X.shape[0])
            n = n/sums
            celltype_size[k]=sums
            exprsed[k] = n.reshape(X.shape[0]) 

    else:
        for k, v in cell_markers.items():
            celltype_size[k]=0
            sums=0
            n = np.zeros((X.shape[0]))
            for i in v:
                if i in ad.var.index:
                    expr95 = np.percentile(X[:,ad.var.index == i],95)
                    thresh = thresh * expr95
                    l = np.array(X[:,ad.var.index == i])
                    l[X[:,ad.var.index == i]<=thresh]=0
                    n[np.array(l>0).reshape(X.shape[0])] += 1
                    sums += 1
                    labels[k] += l.reshape(X.shape[0])
            n = n/sums
            celltype_size[k]=sums
            exprsed[k] = n.reshape(X.shape[0])        
    
    if use_knn:##not used in final method
        
        ad.obsm['X_score']=labels.values
        subsample = ad[ad.obs['source']=='reference']
        subsample = downsample_to_smallest_category(subsample, 'cell_type', min_cells=500, 
                                                    keep_small_categories=True)#default 500
        neigh = KNeighborsClassifier(n_neighbors=5,weights='distance')
        #neigh = SVC(random_state=1,kernel='rbf',class_weight="balanced",probability=True)
        neigh.fit(subsample.obsm['X_score'], subsample.obs['cell_type'])
        new_labels = neigh.predict_proba(ad.obsm['X_score'])
        new_labels = pd.DataFrame(new_labels)
        new_labels.columns = neigh.classes_
        
        labels = labels.reindex(sorted(labels.columns), axis=1)
        new_labels = new_labels.reindex(sorted(new_labels.columns), axis=1)
    
    else:
        assess1 = np.argmax((labels*exprsed).values,axis=1)
        vals1 = 0
        for k,v in collections.Counter(assess1).items():
            if v >= 5:
                vals1 += 1
                            
        assess1 = vals1
    
        assess2 = np.argmax((labels).values,axis=1)
        vals2 = 0
        for k,v in collections.Counter(assess2).items():
            if v >= 5:
                vals2 += 1
                       
        assess2 = vals2
    
        assess = [assess1,assess2]
    
        new_labels = [labels*exprsed,labels][assess.index(max(assess))]
    
    print(labels.shape)
    return labels, new_labels

def gene2mat(ad=None, cell_markers=None,thresh=0.5,if_tfidf=True,if_thresh=True):
    ##TF-IDF transformation
    X = ad.X.copy()
    if scipy.sparse.issparse(X):
        X = X.todense()
    
    if if_tfidf:
        tf_transformer = TfidfTransformer(use_idf=True).fit(X)
        X= tf_transformer.transform(X).todense()
        
    maxlen = 0
    for k,v in cell_markers.items():
        if len(v)>=maxlen:
            maxlen=len(v)
    
    marker_expr = np.zeros((X.shape[0],maxlen,len(cell_markers)))
    b1 = 0
    for k, v in cell_markers.items():
        b2 = 0
        for i in v:
            if i in ad.var.index:
                expr95 = np.percentile(X[:,ad.var.index == i],95)
                thresh = thresh * expr95
                l = np.array(X[:,ad.var.index == i])
                if if_thresh:
                    l[X[:,ad.var.index == i]<=thresh]=0
                marker_expr[:,b2,b1] = l.reshape(X.shape[0])
                b2 += 1
            else:
                b2 += 1
        b1 += 1
        
    return marker_expr

def downsample_to_smallest_category(adata,column="cell_type",random_state=None,
                                    min_cells=10,keep_small_categories=False):
    
    counts = adata.obs[column].value_counts(sort=False)
    min_size = min(counts[counts >= min_cells])
    sample_selection = None
    for sample, num_cells in counts.items():
        if num_cells <= min_cells:
            if keep_small_categories:
                sel = adata.obs.index.isin(
                    adata.obs[adata.obs[column] == sample].index)
            else:
                continue
        else:
            sel = adata.obs.index.isin(
                adata.obs[adata.obs[column] == sample]
                .sample(min_size, random_state=random_state)
                .index
            )
        if sample_selection is None:
            sample_selection = sel
        else:
            sample_selection |= sel
    return adata[sample_selection].copy()

##-----------------------------------------------------------------------------
##MASIv2 updates
'''
A modified Attention-based graph net
''' 
class AGNNConv(nn.Module):

    def __init__(
        self, init_beta=1.0, learn_beta=True, temp=1.0, allow_zero_in_degree=False
    ):
        super(AGNNConv, self).__init__()
        self.temp=temp
        self._allow_zero_in_degree = allow_zero_in_degree
        if learn_beta:
            self.beta = nn.Parameter(torch.Tensor([init_beta]))
        else:
            self.register_buffer("beta", torch.Tensor([init_beta]))

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute AGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *)` and :math:`(N_{out}, *)`, the :math:`*` in the later
            tensor must equal the previous one.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata["h"] = feat_src
            graph.srcdata["norm_h"] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple) or graph.is_block:
                graph.dstdata["norm_h"] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v("norm_h", "norm_h", "cos"))
            cos = graph.edata.pop("cos")
            e = self.beta * cos
            attn = edge_softmax(graph, e/self.temp)
            
            graph.edata["p"] = attn
            
            graph.update_all(fn.u_mul_e("h", "p", "m"), fn.sum("m", "h"))
            return graph.dstdata.pop("h")
        
'''
Prepare data into graph
'''
def process(nodes_data,edges_data):
    
    node_features = torch.from_numpy(nodes_data)
    edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
    edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
    edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())

    graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
    graph.ndata['feat'] = node_features
    graph.edata['weight'] = edge_features
    
    graph = dgl.add_self_loop(graph)
    
    return graph

'''
Major classes in MASIv2
'''
class MarkerWeight(nn.Module):
    def __init__(self,marker_dim=20,output_dim=16):
        super(MarkerWeight, self).__init__()
        self.output_dim = output_dim
        self.marker_dim = marker_dim
        self.weight = torch.nn.Parameter(torch.ones(self.marker_dim,self.output_dim))
        
    def forward(self, x):
        y = (x * self.weight.reshape(1,self.marker_dim,self.output_dim))
        return y
    
class RefCentroid(nn.Module):
    def __init__(self,ref_center=None):
        super(RefCentroid, self).__init__()
        self.weight = torch.nn.Parameter(torch.from_numpy(ref_center),requires_grad=False)
        
    def forward(self, x):
        y = torch.matmul(x,self.weight)
        return y
   
class Scaler(nn.Module):
    def __init__(self,ref_center=None):
        super(Scaler, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(ref_center.shape[1]))
        self.bias = torch.nn.Parameter(torch.zeros(ref_center.shape[1]))
        
    def forward(self, x):
        y = self.weight*x+self.bias
        return y
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        n_b = x.data.size(0)
        m,c = self.shape
        return x.view(n_b,m,c)
    
'''
Resource light Model with no cell type centroids
'''
class GCNClassifier_noref(nn.Module):
    def __init__(self,marker_dim=20,output_dim=16,t1=1.0,t2=1.0):
        super(GCNClassifier_noref, self).__init__()
        self.output_dim = output_dim
        self.marker_dim = marker_dim
        self.markerencoder = MarkerWeight(marker_dim=self.marker_dim,output_dim=self.output_dim)
        self.gc = AGNNConv(allow_zero_in_degree=True,learn_beta=False,temp=t1)
        self.batchnorm = nn.BatchNorm1d(self.output_dim)
        self.activation = nn.LeakyReLU(0.25)
        self.softmax = nn.Softmax(dim=1)
        self.temp = torch.tensor(t2, requires_grad=False)
    
    def forward(self, g, x):
        f = self.gc(g, x)
        f = self.gc(g, f)

        f = torch.reshape(f, (f.shape[0],self.marker_dim, self.output_dim))
        f = self.markerencoder(f)
        f = f.sum(dim=1)
        f = self.batchnorm(f)
        f = self.activation(f)
        
        out = self.gc(g, f)
        out = self.gc(g, out)
        
        out = self.softmax(out/self.temp)
        return f,out

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list
    
def train_noref(g, model,epochs,pseudoy):
    
    weights_space=[]
    
    model = model.double()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.01, max_lr=0.5)
    optswa = SWA(opt)
    labels = np.argmax(pseudoy,1)
    labels = torch.tensor(labels).long().to(device)
    features = g.ndata['feat'].to(device)
    
    for e in range(epochs):
        
        noise_inputs = torch.clone(features)
        #noise_inputs[torch.cuda.FloatTensor(features.shape).uniform_() >= 0.2]=0 #default 0.1
        _,out = model(g, noise_inputs)

        loss = nn.NLLLoss(reduction='mean')(torch.log(out),labels)
        
        #Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        if e > 50 and e % 5 == 0:
            optswa.update_swa()
            weights=[]
            for p in model.parameters():
                weights.append(p)
            marker_weights=torch.Tensor.cpu(weights[0]).detach().numpy()
            weights_space.append(marker_weights)
        gc.collect()
        
    optswa.swap_swa_sgd()
    return weights_space

'''
Resource light Model with cell type centroids
'''
class GCNClassifier(nn.Module):
    def __init__(self,marker_dim=20,output_dim=16,t1=1.0,t2=5.0,t3=0.1,ref_center = None):
        super(GCNClassifier, self).__init__()
        self.output_dim = output_dim
        self.marker_dim = marker_dim
        self.markerencoder = MarkerWeight(marker_dim=self.marker_dim,output_dim=self.output_dim)
        self.gc = AGNNConv(allow_zero_in_degree=True,learn_beta=False,temp=t1)
        self.batchnorm = nn.BatchNorm1d(self.output_dim)
        self.activation = nn.LeakyReLU(0.25)
        self.t2 = torch.tensor(t2, requires_grad=False)
        self.t3 = torch.tensor(t3, requires_grad=False)
        self.centroid = RefCentroid(ref_center)
    
    def forward(self, g, x):
        
        fea = self.gc(g, x)
        fea = self.gc(g, fea)
        
        fea = torch.reshape(fea, (fea.shape[0],self.marker_dim, self.output_dim))
        fea = self.markerencoder(fea)
        f = fea.sum(dim=1)
        fea = torch.reshape(fea, (fea.shape[0],self.marker_dim*self.output_dim))
        f = self.batchnorm(f)
        f = self.activation(f)
        
        p = F.softmax(f/self.t3,1)
        p=self.centroid(p)
        
        out = self.gc(g, f)
        out = self.gc(g, out)
        out = F.softmax(out/self.t2,1)
        
        return fea,f,p,out

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list
  
def train(g, model,epochs,pseudoy):
    
    weights_space=[]
    
    model = model.double()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.to(device)
    model.to(device)
    model.centroid.to(device)
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.01, max_lr=0.5)
    optswa = SWA(opt)
    labels = np.argmax(pseudoy,1)
    labels = torch.tensor(labels).long().to(device)
    #labels = torch.tensor(pseudoy).to(device)
    features = g.ndata['feat'].to(device)
    
    for e in range(epochs):
        
        noise_inputs = torch.clone(features)
        f,_,p,out = model(g, noise_inputs)
        
        clf_loss = nn.NLLLoss(reduction='mean')(torch.log(out),labels)
        #clf_loss = nn.KLDivLoss(reduction='mean')(torch.log(out),F.softmax(labels/model.t2,1))
        rec_loss = nn.MSELoss(reduction='mean')(f,p)
        #rec_loss = nn.MSELoss(reduction='mean')(f,p.detach())
        #rec_loss = CosineSimilarity(f,p)
        
        loss = clf_loss*0.4 + rec_loss*0.6
        
        #Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        if e > 50 and e % 5 == 0:
            optswa.update_swa()
            weights=[]
            for p in model.parameters():
                weights.append(p)
            marker_weights=torch.Tensor.cpu(weights[0]).detach().numpy()
            weights_space.append(marker_weights)
        gc.collect()
        
    optswa.swap_swa_sgd()
    
    weights_space.append(marker_weights)
    return weights_space

'''
Parallel computing
'''
def parallelAGN_noref(ad=None,scores=None,labels=None,feat=None,
                      epochs=200,batch_size=50000,t1=1.0,t2=2.0):
    
    ad.obsm['X_score']=scores#.values
    ad.obsm['X_label']=labels.values
    ad.obsm['X_node']=feat
    
    index = np.array([i for i in range(scores.shape[0])])
    
    r = np.random.permutation(scores.shape[0])
    r_index = index[r]
    ad = ad[r]
    
    ad_list= []
    for j in range(scores.shape[0]//batch_size+1):
        ad_list.append(ad[j*batch_size:(j+1)*batch_size])
    
    all_weights=[]
    for i in range(len(ad_list)):
        subad = ad_list[i]
        sc.pp.neighbors(subad, n_neighbors=10, use_rep='X_node',metric='cosine',key_added='Adj')
        adj = subad.obsp['Adj_connectivities']
        adj = np.array(adj.todense()).flatten()
        src_node = np.repeat([i for i in range(subad.X.shape[0])],repeats=subad.X.shape[0])
        dst_node = np.tile([i for i in range(subad.X.shape[0])],subad.X.shape[0])

        edge_data = pd.DataFrame({'Src':src_node,'Dst':dst_node,'Weight':adj})
        edge_data = edge_data[edge_data['Weight']>0]
        
        graph = dgl.graph((edge_data['Src'].values.tolist(), edge_data['Dst'].values.tolist()))
        
        mat = subad.obsm['X_score']
        node_features = torch.from_numpy(mat)
        graph.ndata['feat'] = node_features
        graph = dgl.add_self_loop(graph)
        
        student = GCNClassifier_noref(output_dim=labels.shape[1],marker_dim=20,t1=t1,t2=t2)
        w=train_noref(graph, student,epochs,subad.obsm['X_label'])
        all_weights=all_weights+w
        
        student.to(torch.device("cpu"))
        graph = graph.to(torch.device("cpu"))
        student.temp = torch.tensor(1.0, requires_grad=False)

        #student.eval()

        scores,labels = student(graph,graph.ndata['feat'])
        scores = torch.Tensor.cpu(scores).detach().numpy()
        labels = torch.Tensor.cpu(labels).detach().numpy()
        
        ad_list[i].obsm['X_score']=scores
        ad_list[i].obsm['X_label']=labels

        
    merged = ad_list[0]
    for m in ad_list[1:]:
        merged = merged.concatenate(m)
    merged = merged[r_index.argsort()]
        
    return merged,all_weights

def parallelAGN(ad=None,scores=None,labels=None,feat=None,marker_dim=50,epochs=200,
                batch_size=50000,t1=1.0,t2=15.0,t3=1.0,ref_center = None,
                res=1.0,key_node_metric='degree centrality'):
    
    ad.obsm['X_score']=scores#.values
    ad.obsm['X_label']=labels.values
    ad.obsm['X_node']=feat
    
    index = np.array([i for i in range(scores.shape[0])])
    
    r = np.random.permutation(scores.shape[0])
    r_index = index[r]
    ad = ad[r]
    
    ad_list= []
    for j in range(scores.shape[0]//batch_size+1):
        ad_list.append(ad[j*batch_size:(j+1)*batch_size])
    
    all_weights=[]
    for i in range(len(ad_list)):
        subad = ad_list[i]
        sc.pp.neighbors(subad, n_neighbors=10, use_rep='X_node',metric='cosine',key_added='Adj')
        adj = subad.obsp['Adj_connectivities']
        adj = np.array(adj.todense()).flatten()
        src_node = np.repeat([i for i in range(subad.X.shape[0])],repeats=subad.X.shape[0])
        dst_node = np.tile([i for i in range(subad.X.shape[0])],subad.X.shape[0])

        edge_data = pd.DataFrame({'Src':src_node,'Dst':dst_node,'Weight':adj})
        edge_data = edge_data[edge_data['Weight']>0]
        
        G = nx.from_pandas_edgelist(edge_data,'Src','Dst')
        partition = community.best_partition(G,resolution=res)
        
        if key_node_metric=='degree centrality':
            within_degree = {}
            for node, community_id in partition.items():
                degree = G.degree(node)
                if community_id not in within_degree:
                    within_degree[community_id] = {}
                within_degree[community_id][node] = degree
            
            key_nodes = {}
            for community_id, degrees in within_degree.items():
                key_node = max(degrees, key=degrees.get)
                key_nodes[community_id] = key_node
                
        elif key_node_metric=='betweenness centrality':
            betweenness = nx.betweenness_centrality(G)

            community_betweenness = {}
            for node, community_id in partition.items():
                if community_id not in community_betweenness:
                    community_betweenness[community_id] = {}
                community_betweenness[community_id][node] = betweenness[node]

            key_nodes = {}
            for community_id, centrality_scores in community_betweenness.items():
                key_node = max(centrality_scores, key=centrality_scores.get)
                key_nodes[community_id] = key_node
                
        else:
            eigenvector = nx.eigenvector_centrality(G)

            community_eigenvector = {}
            for node, community_id in partition.items():
                if community_id not in community_eigenvector:
                    community_eigenvector[community_id] = {}
                community_eigenvector[community_id][node] = eigenvector[node]

            key_nodes = {}
            for community_id, centrality_scores in community_eigenvector.items():
                key_node = max(centrality_scores, key=centrality_scores.get)
                key_nodes[community_id] = key_node
        
        key_cell_index = [v for k,v in key_nodes.items()]
        key_cell = np.zeros(subad.X.shape[0])
        key_cell[key_cell_index]=1
        
        graph = dgl.graph((edge_data['Src'].values.tolist(), 
                           edge_data['Dst'].values.tolist()))
        
        mat = subad.obsm['X_score']
        node_features = torch.from_numpy(mat)
        graph.ndata['feat'] = node_features
        graph = dgl.add_self_loop(graph)
        
        student = GCNClassifier(output_dim=labels.shape[1],marker_dim=marker_dim,
                                t1=t1,t2=t2,t3=t3,ref_center = ref_center)
        w=train(graph, student,epochs,subad.obsm['X_label'])
        all_weights.append(w)
        
        student.to(torch.device("cpu"))
        graph = graph.to(torch.device("cpu"))
        student.t2 = torch.tensor(1.0, requires_grad=False)

        #student.eval()

        _,scores,_,labels = student(graph,graph.ndata['feat'])
        scores = torch.Tensor.cpu(scores).detach().numpy()
        labels = torch.Tensor.cpu(labels).detach().numpy()
        ad_list[i].obsm['X_score']=scores
        ad_list[i].obsm['X_label']=labels
        ad_list[i].obs['Node_label']=key_cell

        
    merged = ad_list[0]
    for m in ad_list[1:]:
        merged = merged.concatenate(m)
    merged = merged[r_index.argsort()]
        
    return merged,all_weights

'''
Linear correction for alignment
'''     
def centroid_align(source,target):
    
    source_input = pd.DataFrame(source.obsm['X_score'])
    source_input['cell_type']=source.obs['cell_type'].values
    source_input = source_input.groupby('cell_type').mean()

    target_input = pd.DataFrame(target.obsm['X_score'])
    target_input['cell_type']=target.obs['cell_type'].values
    target_input = target_input.groupby('cell_type').mean()
    
    source_unique_label = pd.unique(source.obs['cell_type'])
    source_input = source_input.reindex(source_unique_label)
    
    target_unique_label = pd.unique(target.obs['cell_type'])
    target_input = target_input.reindex(target_unique_label)
    
    commons = list(set(target_input.index.tolist()).intersection(source_input.index.tolist()))
    target_input = target_input.T[commons].T
    source_input = source_input.T[commons].T
    
    reg = LinearRegression().fit(target_input.values,source_input.values)
    return reg

def reg_align(source,target):
    
    source_unique_label = pd.unique(source.obs['cell_type'])
    target_unique_label = pd.unique(target.obs['cell_type'])
    cell_type = [i for i in source_unique_label if i in target_unique_label]
    
    r = np.random.permutation(source.X.shape[0])
    source = source[r]
    r = np.random.permutation(target.X.shape[0])
    target = target[r]
    
    source_input = np.expand_dims(source[source.obs['cell_type']==cell_type[0]].obsm['X_score'][0,:], axis=1)
    target_input = np.expand_dims(target[target.obs['cell_type']==cell_type[0]].obsm['X_score'][0,:], axis=1)
    for c in cell_type[1:]:
        source_input = np.concatenate((source_input, 
                                       np.expand_dims(source[source.obs['cell_type']==c].obsm['X_score'][0,:], axis=1)), 
                                       axis=1)
        target_input = np.concatenate((target_input, 
                                       np.expand_dims(target[target.obs['cell_type']==c].obsm['X_score'][0,:], axis=1)),
                                       axis=1)
    
    for e in range(199):
        
        r = np.random.permutation(source.X.shape[0])
        source = source[r]
        r = np.random.permutation(target.X.shape[0])
        target = target[r]
        for c in cell_type:
            source_input = np.concatenate((source_input, 
                                           np.expand_dims(source[source.obs['cell_type']==c].obsm['X_score'][0,:], axis=1)), 
                                           axis=1)
            target_input = np.concatenate((target_input, 
                                           np.expand_dims(target[target.obs['cell_type']==c].obsm['X_score'][0,:], axis=1)),
                                           axis=1)
            
    
    reg = LinearRegression().fit(target_input.T,source_input.T)
    return reg

def find_key_cell(source=None,batch_size=50000,res=2.0,key_node_metric='degree centrality'):
    
    index = np.array([i for i in range(source.X.shape[0])])
    
    r = np.random.permutation(source.X.shape[0])
    r_index = index[r]
    source = source[r]
    
    ad_list= []
    for j in range(source.X.shape[0]//batch_size+1):
        ad_list.append(source[j*batch_size:(j+1)*batch_size])
    
    for i in range(len(ad_list)):
        subad = ad_list[i]
    
        sc.pp.neighbors(subad, n_neighbors=10, use_rep='X_score',metric='cosine',key_added='Adj')
        adj = subad.obsp['Adj_connectivities']
        #adj = np.array(adj.todense()).flatten()
        #src_node = np.repeat([i for i in range(subad.X.shape[0])],repeats=subad.X.shape[0])
        #dst_node = np.tile([i for i in range(subad.X.shape[0])],subad.X.shape[0])
        
        #edge_data = pd.DataFrame({'Src':src_node,'Dst':dst_node,'Weight':adj})
        #edge_data = edge_data[edge_data['Weight']>0]
        
        G = nx.from_scipy_sparse_array(adj)
        #G = nx.from_pandas_edgelist(edge_data,'Src','Dst')
        partition = community.best_partition(G,resolution=res)
        
        if key_node_metric=='degree centrality':
            within_degree = {}
            for node, community_id in partition.items():
                degree = G.degree(node)
                if community_id not in within_degree:
                    within_degree[community_id] = {}
                within_degree[community_id][node] = degree
        
            key_nodes = {}
            for community_id, degrees in within_degree.items():
                key_node = max(degrees, key=degrees.get)
                key_nodes[community_id] = key_node
            
        elif key_node_metric=='betweenness centrality':
            betweenness = nx.betweenness_centrality(G)
            
            community_betweenness = {}
            for node, community_id in partition.items():
                if community_id not in community_betweenness:
                    community_betweenness[community_id] = {}
                community_betweenness[community_id][node] = betweenness[node]
                
            key_nodes = {}
            for community_id, centrality_scores in community_betweenness.items():
                key_node = max(centrality_scores, key=centrality_scores.get)
                key_nodes[community_id] = key_node
            
        else:
            eigenvector = nx.eigenvector_centrality(G)
            
            community_eigenvector = {}
            for node, community_id in partition.items():
                if community_id not in community_eigenvector:
                    community_eigenvector[community_id] = {}
                community_eigenvector[community_id][node] = eigenvector[node]

            key_nodes = {}
            for community_id, centrality_scores in community_eigenvector.items():
                key_node = max(centrality_scores, key=centrality_scores.get)
                key_nodes[community_id] = key_node
    
        key_cell_index = [v for k,v in key_nodes.items()]
        key_cell = np.zeros(subad.X.shape[0])
        key_cell[key_cell_index]=1
    
        ad_list[i].obs['Node_label']=key_cell

    
    merged = ad_list[0]
    for m in ad_list[1:]:
        merged = merged.concatenate(m)
    merged = merged[r_index.argsort()]
    
    return merged
