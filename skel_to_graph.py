# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:13:43 2016

@author: yuliya
"""

from skimage import util
from scipy import ndimage
import numpy as np
import networkx as nx
from nx_merge_nodes import merge_nodes

        
def numb(skel_mat) :
    """
    Counts the number of neighboring 1 for every nonzero
    element in 3D.
    
    Parameters
    ------
    skel_mat : 3d binary array
    
    Returns
    ------
    arr : The array of uint8 of the same size as skel_mat 
    with the elements equal to the number of neighbors 
    for the current point.
    
    Examples
    ------
    >>> a = np.random.random_integers(0,1,(3,3,3))
    >>> a 
    array([[[0, 0, 1],
            [0, 1, 1],
            [1, 0, 1]],        
           [[1, 0, 1],
            [1, 0, 1],
            [1, 1, 0]],        
           [[1, 1, 1],
            [1, 0, 0],
            [1, 1, 0]]])
    >>> neigh = numb(a)
    >>> neigh 
    array([[[ 0,  0,  4],
            [ 0, 10,  6],
            [ 4,  0,  4]],            
           [[ 5,  0,  6],
            [10,  0,  9],
            [ 7, 10,  0]],            
           [[ 4,  7,  3],
            [ 8,  0,  0],
            [ 5,  6,  0]]], dtype=uint8)
    """
    c_pad = util.pad(skel_mat,((1,1),(1,1),(1,1)), 'constant')
    mask = c_pad > 0
    fil = 3**3 * ndimage.uniform_filter(c_pad.astype('float'), 
                                        size = (3,3,3)) - 1
    arr = (fil * mask)[1:-1,1:-1,1:-1].astype('uint8')
    return arr
    

def label(skel_mat):
    """
    Creates the graph of nodes corresponding to every
    nonzero element of the input binary matrix. The nodes have
    the attributes 'index' that is the index of the element in
    the initial matrix; 'neig' is the number of neighbors of
    the element.
    
    Parameters 
    ---------
    skel_mat : 3d binary matrix

    Returns
    ---------
    G : a graph of the nodes ; each node has the attributes
    'index' and 'neig'
    
    Examples
    ---------
    >>>b = np.zeros((3,3,3))
    >>>b[1,1,:] = 1
    >>>b
    array([[[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],
           [[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 0.,  0.,  0.]],
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]])
    >>>G = label(b)
    >>>G.nodes()
    [1, 2, 3]
    >>>G.node[1]
    {'index': (1, 1, 0), 'neig': 1}
    """
    
    non_z = np.transpose(np.nonzero(skel_mat))
    coor = tuple(map(tuple,non_z))
    n = numb(skel_mat)
    G = nx.Graph()
    for i in range(np.size(non_z, axis=0)):
        G.add_node(i+1, neig = n[coor[i]], index = coor[i])
    return G
    
    
def adj_matrix(Graph, skel_mat):    
    """
    Returns adjacency matrix for the given skeleton
    binary matrix and the corresponding graph of nodes.
    
    Parameters
    --------
    Graph : a graph of nodes; each node correspond to the
    nonzero element of skeleton matrix and has the attrubutes
    'index' and 'neig'
    
    skel_mat : 3d binary matrix
    
    Returns
    -------
    a_m : 2d array of the dimentions n by n, where n is equal to
    the number of nodes; matrix is symmetric, nonzero elements 
    indicate the connections between the nodes.
    
    Examples
    -------
    >>>b = np.zeros((3,3,3))
    >>>b[1,1,:] = 1
    >>>b
    array([[[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],
           [[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 0.,  0.,  0.]],
           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]])
    >>>G = label(b)
    >>>adj_m = adj_matrix(G,b)
    >>>adj_m
    array([[ 0.,  1.,  0.],
           [ 1.,  0.,  1.],
           [ 0.,  1.,  0.]])
    """
    
    point = nx.get_node_attributes(Graph, 'index')
    node_label = dict( (j,i) for i,j in point.iteritems())
    a_pad = util.pad(skel_mat,((1,1),(1,1),(1,1)), 'constant')
    n = nx.number_of_nodes(Graph)
    a_m = np.zeros((n,n))
    for i in Graph.nodes():
        vol = np.zeros(a_pad.shape)
        co = point[i]
        vol[co[0]:co[0]+3, co[1]:co[1]+3, 
            co[2]:co[2]+3] = a_pad[co[0]:co[0]+3, co[1]:co[1]+3, co[2]:co[2]+3]
        nz_vol = np.transpose(np.nonzero(vol)) - 1  
        el = tuple(map(tuple,nz_vol))
        for elem in el:
            j = node_label[elem]
            a_m [i-1,j-1] = 1   
    a_m = a_m - np.eye(n)
    return a_m    
    

def edges(Graph, a_m):
    """
    Returns connected graph, creating the edges in the
    input graph of nodes and the adjacency matrix; 
    the edges have the attribute 'length' that is set to 1
    (edges connect neighboring pixels)
    
    Parameters
    -------
    Graph : a graph of nodes
    
    a_m : adjacency matrix; 2d array n by n, where n equals 
    the number of nodes in the input graph
    
    Returns
    -------
    G : connected graph
    
    Examples
    -------
    >>>G.nodes()
    [1, 2, 3]   
    >>>adj_m
    array([[ 0.,  1.,  0.],
           [ 1.,  0.,  1.],
           [ 0.,  1.,  0.]])
    >>>G_con = edges(G, adj_m)
    >>>G_con.edges()
    [(1, 2), (2, 3)]
    """
    G = Graph.copy()
    non_z = np.transpose(np.nonzero(a_m))
    for tup in non_z:
        G.add_edge(tup[0]+1, tup[1]+1, length=1)
    return G



def branches(Graph):
    """
    Modifies the connections of the input graph in a way
    to create the branches by removing the connection points
    and creating now edges between the surrounding points.
    The attribute 'length' depending on the number of removed
    connection nodes.    
    
    Parameters
    --------
    Graph : a connected graph with large number of nodes and 
    edges
    
    Returns
    --------
    G : a reduced graph
    
    Examples
    --------
    >>>G_con.nodes()
    [1, 2, 3]
    >>>G_con.edges()
    [(1, 2), (2, 3)]
    >>>nx.get_edge_attributes(G_con, 'length')
    {(1, 2): 1, (2, 3): 1}
    >>>G_red = branches(G_con)
    >>>G_red.nodes()
    [1, 2]
    >>>G_red.edges()
    [(1, 2)]
    >>>nx.get_edge_attributes(G_red, 'length')
    {(1, 2): 2}
    """
    G = Graph.copy()
    neigh = nx.get_node_attributes(G, 'neig')
    for i in G.nodes():
        if (neigh[i]==2) :
            leng = nx.get_edge_attributes(G,'length')
            n = nx.all_neighbors(G,i)
            u = n.next()
            v = n.next()
            l1 = leng[(min(u,i), max(u,i))]
            l2 = leng[(min(v,i), max(v,i))]
            G.add_edge(min(u,v), max(u,v), length = l1+l2) 
            G.remove_node(i)
    G = nx.convert_node_labels_to_integers(G,1)
    return G    
    



def clust(Graph):
    """
    Returns the graph that merges artificial loops into a
    single node. Detects the nodes included to the triangles
    and merges them. Uses the extern function merge_nodes.
    
    Parameters
    --------
    Graph : input graph with artificial loops
    
    Returns
    -------
    G : a graph without loops; triangles of neighboring nodes
    are replaced by a single node
￼

    """
    G = Graph.copy()
    size = G.number_of_nodes()           
    for i in G.nodes():
        neigh = nx.get_node_attributes(G, 'neig')
        index = nx.get_node_attributes(G, 'index')
        if (i in G.nodes() and nx.triangles(G, i))>0:
            n = nx.all_neighbors(G,i)
            l = [i]
            for k in n:
                if ((neigh[k]>2) and 
                    (nx.get_edge_attributes(G, 'length')[min(i,k), max(i,k)]<2)):
                    l = np.append(l, k)
            merge_nodes(G,l,size+1,index = index[i], neig = neigh[i])
            size+=1
        if (i==G.number_of_nodes()):
            break
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    return G
