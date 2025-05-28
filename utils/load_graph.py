import numpy as np

def get_J(file_name):
    # Load the sparse matrix
    with open(file_name, 'r') as file:
        next(file) #start reading from second line
        edges = [line.strip().split() for line in file]
    
    # Convert nodes to integers and find the maximum node number to determine matrix size
    nodes = set(int(edge[0]) for edge in edges) | set(int(edge[1]) for edge in edges)
    max_node = max(nodes)

    J = np.zeros((max_node, max_node))

    for edge in edges:
        node1, node2, weight = int(edge[0]), int(edge[1]), float(edge[2])
        J[node1 - 1, node2 - 1] = weight  # Adjust for zero-based indexing
        J[node2 - 1, node1 - 1] = weight  # If the graph is undirected

    return J