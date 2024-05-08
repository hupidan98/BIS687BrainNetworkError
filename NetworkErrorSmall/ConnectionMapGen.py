import numpy as np
import pandas as pd
import random

# Method for generating a connection map of neural network. If there are 100 cells in the network, this will be a 100*100 matrixs, [i,j]entry = w means the connection weight w，if w > 0, this is a excititory connection with weight w, if w < 0, this is a inhibitory connection with weight -w.



def MultilevelMapGen(num_node, deterministic_weight = None, max_weight = 0.01, node_level_num = None, interlevel_conn_prob = 0.8):
    connection_map = np.zeros((num_node, num_node))
    if num_node == 2:
        connection_map[0, 1] = random.uniform(-max_weight, max_weight)
        return connection_map
    if node_level_num == None:
        # default to 2 layer
        second_level_count = int(num_node/3)
        node_level_num = [num_node-1-second_level_count, second_level_count, 1]
    # Connection
    node_layer_idx = []
    layer_starts = 0
    layer_ends = 0
    for layer_count in node_level_num:
        layer_ends += layer_count
        layer_idx = list(range(layer_starts, layer_ends))
        node_layer_idx.append(layer_idx)
        layer_starts += layer_count
    for i in range(len(node_layer_idx) - 1): 
        this_layer = node_layer_idx[i]
        next_layer = node_layer_idx[i + 1]
        for j in this_layer:
            for k in next_layer:
                rand_det = random.uniform(0, 1)
                if i == (len(node_layer_idx) - 2):
                    rand_det = 0
                if rand_det < interlevel_conn_prob:
                    weight = random.uniform(-max_weight, max_weight)
                    if deterministic_weight:
                        weight = deterministic_weight
                    connection_map[j, k] = weight
    return connection_map

# Generating a Feedforward network of given number of cells (num_node), the cells are evenly dividened into node_level_num of layers (default to 2). If deterministic_weight is set to a given number, all interlayer connections will have weight of that given number, otherwise, random weight from distribution uniform(-max_weight, +max_weight) will be set to each inerlayer connection, max_weight has dafult value of 0.01. interlevel_conn_prob is the probablity of whether two cells in adjcent layers is conenction, 1 means a fully connected network, and 0 means no cells will be connected.
def FeedForwardMapGen(num_node, deterministic_weight = None, max_weight = 0.01, node_level_num = None, interlevel_conn_prob = 0.8):
    connection_map = np.zeros((num_node, num_node))
    if num_node == 2:
        connection_map[0, 1] = random.uniform(-max_weight, max_weight)
        return connection_map
    if node_level_num == None:
        # default to 2 layer
        second_level_count = int(num_node/2)
        node_level_num = [num_node-second_level_count, second_level_count]
    # Connection
    node_layer_idx = []
    layer_starts = 0
    layer_ends = 0
    for layer_count in node_level_num:
        layer_ends += layer_count
        layer_idx = list(range(layer_starts, layer_ends))
        node_layer_idx.append(layer_idx)
        layer_starts += layer_count
    for i in range(len(node_layer_idx) - 1): 
        this_layer = node_layer_idx[i]
        next_layer = node_layer_idx[i + 1]
        for j in this_layer:
            for k in next_layer:
                rand_det = random.uniform(0, 1)
                if rand_det < interlevel_conn_prob:
                    weight = random.uniform(-max_weight, max_weight)
                    if deterministic_weight:
                        weight = deterministic_weight
                    connection_map[j, k] = weight
    return connection_map
                
        
        
# Generating a ScaleFree network of given number of cells (num_node). If deterministic_weight is set to a given number, all connections will have weight of that given number, otherwise, random weight from distribution uniform(-max_weight, +max_weight) will be set to each connection, max_weight has dafult value of 0.01. 
def ScaleFreeMapGen(num_node, deterministic_weight = None, max_weight = 0.01):
    connection_map = np.zeros((num_node, num_node))
    scale_para = 1.8
    prob_sum = 0
    for i in range(1, num_node+1):
        prob_sum += 1/(i ** scale_para)
    # for Scale Free, the probability is the inverse of # of connection
    for i in range(num_node):
        # Find out how many out going connection of a given node
        rn = random.uniform(0, prob_sum)
        num_conn = 1
        while rn > 0:
            rn = rn - 1/(num_conn ** scale_para)
            num_conn += 1
            if num_conn == num_node:
                break
        # Which nodes the current nodes are connected to
        node_conn = random.sample(range(num_node), num_conn)
        # Put the weight into the connection matrix
        for j in node_conn:
            # weight from distribution uniform(-0.1, 0.1), can change here later to determine distribution
            weight = random.uniform(-max_weight, max_weight)
            if deterministic_weight:
                weight = deterministic_weight
            connection_map[i, j-1] = weight
    return connection_map
     

# Generate a SmallWorld network of given number of cells (num_node). If deterministic_weight is set to a given number, all connections will have weight of that given number, otherwise, random weight from distribution uniform(-max_weight, +max_weight) will be set to each connection, max_weight has dafult value of 0.01. 
def SmallWorldMapGen(num_node, deterministic_weight = None, max_weight = 0.01):
    connection_map = np.zeros((num_node, num_node))
    # Parameters for SmallWorld Network, can be changed later
    N = num_node
    m = int(np.log(num_node)) + 1 # number of neighbour
    M = N * m
    beta = 0.2
    # Find out what are some connections needs to be rewired
    num_rewireing = int(M * beta)
    conn_rewire = random.sample(range(M), num_rewireing)
    # Start connection, 
    counter = 0
    for i in range(num_node):
        for j in range(m):
            # weight from distribution uniform(-0.1, 0.1), can change here later to determine distribution
            weight = random.uniform(-max_weight, max_weight)
            if deterministic_weight:
                weight = deterministic_weight
            if counter in conn_rewire:
                #wire to a random one
                connection_map[i, (i + random.sample(range(1, num_node), 1)[0]) % num_node] = weight
            else:
                connection_map[i, (i + j) % num_node] = weight
            counter += 1
    return connection_map
            

def RingMapGen(num_node, deterministic_weight = None):
    connection_map = np.zeros((num_node, num_node))
    for i in range(num_node):
        weight = random.uniform(-max_weight, max_weight)
        if deterministic_weight:
            weight = deterministic_weight
        connection_map[i, (i+1) % num_node] = weight
    


def replaceNodewithNetwork(org_network, n, added_network):
    out_by_n = org_network[n, :]
    incoming_to_n = org_network[:, n]
    dim = org_network.shape[0] + added_network.shape[0]-1
    new_network = np.zeros((dim, dim))
    # Relocate
    added_dim = added_network.shape[0]
    new_network[:n,:n] = org_network[:n,:n]
    new_network[added_dim+n:, added_dim+n:] = org_network[n+1:, n+1:]
    new_network[:n, added_dim+n:] = org_network[:n, n+1:]
    new_network[added_dim+n: , :n] = org_network[n+1: , :n]
    # Put the added_network in
    new_network[n:added_dim+n, n:added_dim+n] = added_network
    # Randomly put incoming and outgoing connection into the added network
    idx = 0
    for i in out_by_n:
        if idx < n:
            rn = random.sample(range(n, n + added_dim), 1)[0]
            new_network[rn , idx] = i
        elif idx >n:
            rn = random.sample(range(n, n + added_dim), 1)[0]
            new_network[rn , idx + added_dim - 1] = i
        idx +=1
    idx = 0
    for i in incoming_to_n:
        if idx < n:
            rn = random.sample(range(n, n + added_dim), 1)[0]
            new_network[idx , rn] = i
        elif idx > n:
            rn = random.sample(range(n, n + added_dim), 1)[0]
            new_network[idx + added_dim - 1, rn] = i
        idx +=1
        
    return new_network