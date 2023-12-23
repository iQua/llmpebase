"""
An implementation of the tree thought structure in which thoughts of 
one reasoning process are organized as a tree.

1. Depth-First Growth (DFG):
The tree is expanded as deeply as possible along each branch before backtracking. 
This approach is used in algorithms like XGBoost.
Example: https://www.geeksforgeeks.org/preorder-from-inorder-and-postorder-traversals/

2. Breadth-First Growth (BFG):
The tree is expanded level by level, creating a balanced tree. Each node at a 
certain depth is expanded before the tree grows deeper.

3. Leaf-Wise (Best-First) Growth (LWG):
Used in LightGBM, this method focuses on expanding the leaf that will reduce the 
loss the most. It can result in deeper, more asymmetric trees.

"""
import logging

import networkx as nx

from llmpebase.model.thought_structure import base
from networkx.algorithms.dag import dag_longest_path


class DFGTreeThoughtStructure(base.BaseThoughtStructure):
    """
    A tree thought structure in which the tree is grown in a depth-wise manner
    or depth-first manner.
    """

    def get_grow_node(self):
        """Get the node to be grown next in the tree."""
        # Collect existing nodes with Depth First Search (DFS) algorithm
        dfs_nodes = nx.dfs_preorder_nodes(self.graph, source=self.root.identity)
        node = None
        for node_id in dfs_nodes:
            if self.node_pool[node_id].growth == "Growable":
                node = self.node_pool[node_id]
                break
        return node


class BFGTreeThoughtStructure(base.BaseThoughtStructure):
    """
    A tree thought structure in which the tree is grown in a leaf-wise manner or
    best first manner.
    """

    def get_grow_node(self):
        """Grow the thought structure in the tree version."""
        # Collect existing nodes with Breath First Search (DFS) algorithm
        bfs_nodes = [self.root.identity] + [
            successor
            for successors in dict(
                nx.bfs_successors(self.graph, self.root.identity)
            ).values()
            for successor in successors
        ]
        node = None
        for node_id in bfs_nodes:
            if self.node_pool[node_id].growth == "Growable":
                node = self.node_pool[node_id]
                break
        return node


class LWGTreeThoughtStructure(base.BaseThoughtStructure):
    """
    A tree thought structure in which the tree is grown in a leaf-wise manner or
    best first manner.
    """

    def get_grow_node(self):
        """Grow the thought structure in the tree version."""
        # Collect existing nodes with Depth First Search (DFS) algorithm
        # Visit the nodes with the Breath First Search algorithm
        level_nodes = nx.bfs_layers(self.graph, self.root.identity)
        level_nodes = list(level_nodes)
        # Get the current depth of the tree
        longest_path = dag_longest_path(self.graph)
        num_depth = len(longest_path)

        # When the depth reaches the maximum, stop growing
        if num_depth >= self.max_length:
            return None

        # Visit the thought value of node in the current level
        nodes = level_nodes[num_depth - 1]
        # Get the one with highest thought value
        max_value = 0
        max_node = None
        for node_id in nodes:
            node = self.node_pool[node_id]
            if node.growth == "Growable":
                if (
                    node.identity == self.root.identity
                    or node.thought_score > max_value
                ):
                    max_node = node
                    max_value = node.thought_score
        if max_node is not None:
            logging.info(
                "Node %s has the largest value %s.", max_node.identity, max_value
            )
        return max_node


def get(growth_type: str):
    """Get the thought structure."""
    growth_type = growth_type.lower()

    logging.info("Using %s growth for the Thought Tree Structure.", growth_type)

    if growth_type == "dfg":
        return DFGTreeThoughtStructure
    if growth_type == "bfg":
        return BFGTreeThoughtStructure
    if growth_type == "lwg":
        return LWGTreeThoughtStructure

    raise ValueError(f"Unknown growth type: {growth_type}")
