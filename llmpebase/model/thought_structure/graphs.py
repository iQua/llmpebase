"""
An implementation of the graph thought structure in which thoughts of 
one reasoning process are organized as a graph, i.e., Graph of Thoughts (GoT).

The basic idea of the graph structure follows the paper:
Graph of Thoughts: Solving Elaborate Problems with Large Language Models.

To implement this thought structure, we utilize the breath first growth to 
explore the structure and build the graph layer by layer.

However, as the original GoT proposed in the paper does not support the reasoning
on mathematical problems, we make some changes to GoT by redesigning the 
1. Aggregation Transformations
2. Refining Transformations
"""

import logging
from typing import Dict, List

import networkx as nx

from llmpebase.model.thought_structure import base


class GraphTreeThoughtStructure(base.BaseThoughtStructure):
    """
    A graph thought structure.
    """

    def get_grow_node(self):
        """Get the node to be grown next in the tree."""
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

    def is_duplicated_path(
        self,
        node1_id: int,
        node2_id: int,
    ):
        """
        We remove the constrain on the duplicated path to
        search all nodes to find the similar thoughts.
        """
        return True

    def extend_node(
        self,
        thought: str,
        thought_score: float,
        node_ids: List[int],
        thought_similarity: Dict[str, float],
        similarity_prompts: Dict[str, str],
        **kwargs,
    ):
        """
        Extend the node by
        1. adding similar thoughts to it,
        2. create a edge that connects from previous node to it.
        """
        # This is the previous node id that produces the thought
        prev_node_id = kwargs["prev_node_id"]
        # In the graph structure, once a similar thought is found and the node holding
        # this thought exists in the next depth, we directly connect the 'prev_node_id'
        # to this node.
        prev_depth = len(self.get_node_path(self.root.identity, prev_node_id))
        prev_nodes = []
        to_nodes = []
        for node_id in node_ids:
            depth = len(
                self.get_node_path(self.root.identity, self.node_pool[node_id].identity)
            )
            if depth == prev_depth:
                prev_nodes.append(node_id)
            if depth == prev_depth + 1:
                to_nodes.append(node_id)

        prev_nodes = [prev_node_id] if prev_node_id in prev_nodes else prev_nodes

        node_ids = prev_nodes if len(to_nodes) == 0 else to_nodes
        node_id = node_ids[0]
        sim_score = thought_similarity[node_id]
        sim_prompt = similarity_prompts[node_id]

        # Add the thought to the node
        self.node_pool[node_id].backup_though(
            thought, thought_score, similarity_score=sim_score, prompt=sim_prompt
        )

        # Create the node edge between the prev node and the current node
        # If there exist similar nodes in the next depth
        # If there is no self loop
        # If there does not exist the edge
        if (
            len(to_nodes) != 0
            and prev_node_id != node_id
            and not self.graph.has_edge(prev_node_id, node_id)
        ):
            self.graph.add_edge(prev_node_id, node_id)
            logging.info("  Created an edge from %s to %s ", prev_node_id, node_id)

        logging.info("  Backed it up to an existing node %s", node_id)

        return node_id
