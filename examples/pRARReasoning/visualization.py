"""
An implementation of the visualizer of the p-RAR.

Note that the plot of the p-RAR thought structure is based on the 'position' 
of the node.

p-RAR approach contains many types of nodes with the 'node_name':
    - For the thought structure
        with 'node_name':
            1). Root Node
            2). Intermediate Node
            3). NormalGenerationThought
            4). PolicySummarizationThought
            5). PolicyExclusionThought
            6). PolicyAssessmentThought
            7). PolicyExistenceThought
        with 'position':
            1). Root
            2). Intermediate
            3). Sink
            4). NormalGenerationIntermediate: s, skyblue
            5). PolicySummarizationIntermediate: p, lavender
            6). PolicyExclusionIntermediate: s, royalblue
            7). PolicyAssessmentIntermediate: h, beige
            8). PolicyExistenceIntermediate: d, darkgray


    - For the policy tree
        with 'node_name':
            1). Root Policy Node
            2). IntermediatePolicy Node
        with 'position':
            1). PolicyRoot
            2). PolicyIntermediate
            3). PolicySink

p-RAR approach contains many types of edges with the 'edge_type':
    - For the thought structure
        1). Reasoning
        2). ThoughtGenerationReasoning 
        3). PolicySummarizationReasoning
        4). PolicyExclusionReasoning
        5). PolicyAssessmentReasoning
        6). PolicyExistenceReasoning
    - For the policy tree
        1). Policy Forwarding
"""

from typing import List

import networkx as nx

from llmpebase.model.thought_structure.structure_generic import BasicNode
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer


# Note:
# 1. you may need to access the website
# https://python-charts.com/colors/ to get the color code
# 2. you may need to access the website
# https://graphviz.org/docs/layouts/ to get the graphviz layout
node_config = {
    "Root": {
        "node_color": "#8FBC8F",
        "node_shape": "o",
        "node_size": 900,
        "alpha": 0.8,
    },
    "Intermediate": {
        "node_color": "#6495ED",
        "node_shape": "s",
        "node_size": 850,
        "alpha": 0.9,
    },
    "Sink": {
        "node_color": "#F5DEB3",
        "node_shape": "8",
        "node_size": 900,
        "alpha": 0.8,
    },
    "NormalGenerationIntermediate": {
        "node_color": "#87CEEB",
        "node_shape": "s",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PolicySummarizationIntermediate": {
        "node_color": "#E6E6FA",
        "node_shape": "p",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PolicyExclusionIntermediate": {
        "node_color": "#4169E1",
        "node_shape": "s",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PolicyAssessmentIntermediate": {
        "node_color": "#F5F5DC",
        "node_shape": "h",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PolicyExistenceIntermediate": {
        "node_color": "#A9A9A9",
        "node_shape": "d",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PolicyRoot": {
        "node_color": "#FFE4E1",
        "node_shape": "o",
        "node_size": 900,
        "alpha": 0.8,
    },
    "PolicyIntermediate": {
        "node_color": "#DB7093",
        "node_shape": "s",
        "node_size": 850,
        "alpha": 0.9,
    },
    "PolicySink": {
        "node_color": "#DB7093",
        "node_shape": "8",
        "node_size": 900,
        "alpha": 0.8,
    },
}

edge_config = {
    "Root": {
        "edge_color": "black",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "Intermediate": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "Sink": {
        "edge_color": "green",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "NormalGenerationIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashdot",
        "arrowsize": 10,
    },
    "PolicySummarizationIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashdot",
        "arrowsize": 8,
    },
    "PolicyExclusionIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashed",
        "arrowsize": 8,
    },
    "PolicyAssessmentIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashed",
        "arrowsize": 8,
    },
    "PolicyExistenceIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashed",
        "arrowsize": 8,
    },
    "PolicyRoot": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "PolicyIntermediate": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "PolicySink": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
}


class PRARVisualizer(BasicStructureVisualizer):
    """The visualizer for the p-RAR thought structure."""

    def create_node_draw_labels(self, graph: nx.DiGraph, node_pool: List[BasicNode]):
        """Create the labels of nodes for drawing the graph."""
        # Create the labels to be plotted
        labels = dict()
        for node_id in graph.nodes:
            node = node_pool[node_id]
            if node.position == "Root":
                labels[node.identity] = "Q"
            if node.position == "PolicyRoot":
                labels[node.identity] = "Task"

            if node.position == "PolicySummarizationIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIS P-{node.step_idx}"

            if node.position == "NormalGenerationIntermediate":
                labels[node.identity] = f"N-{node.identity}\nS-{node.step_idx}"

            if node.position == "PolicyExclusionIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIE S-{node.step_idx}"

            if node.position == "PolicyAssessmentIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIA S-{node.step_idx}"

            if node.position == "PolicyExistenceIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIC S-{node.step_idx}"

            if node.position == "Intermediate":
                labels[node.identity] = f"N-{node.identity}\nS-{node.step_idx}"

            if node.position == "Sink":
                labels[node.identity] = f"N-{node.identity}\nS-{node.step_idx}"

            if node.position == "PolicyIntermediate":
                labels[node.identity] = f"N-{node.identity}\nP-{node.step_idx}"

            if node.position == "PolicySink":
                labels[node.identity] = f"N-{node.identity}\nP-{node.step_idx}"

        return labels
