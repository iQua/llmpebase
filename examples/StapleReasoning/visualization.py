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
            4). PlanSummarizationThought
            5). PlanExclusionThought
            6). PlanAssessmentThought
            7). PlanExistenceThought
        with 'position':
            1). Root
            2). Intermediate
            3). Sink
            4). NormalGenerationIntermediate: s, skyblue
            5). PlanSummarizationIntermediate: p, lavender
            6). PlanExclusionIntermediate: s, royalblue
            7). PlanAssessmentIntermediate: h, beige
            8). PlanExistenceIntermediate: d, darkgray


    - For the plan tree
        with 'node_name':
            1). Root Plan Node
            2). IntermediatePlan Node
        with 'position':
            1). PlanRoot
            2). PlanIntermediate
            3). PlanSink

p-RAR approach contains many types of edges with the 'edge_type':
    - For the thought structure
        1). Reasoning
        2). ThoughtGenerationReasoning 
        3). PlanSummarizationReasoning
        4). PlanExclusionReasoning
        5). PlanAssessmentReasoning
        6). PlanExistenceReasoning
    - For the plan tree
        1). Plan Forwarding
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
    "PlanSummarizationIntermediate": {
        "node_color": "#E6E6FA",
        "node_shape": "p",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PlanExclusionIntermediate": {
        "node_color": "#4169E1",
        "node_shape": "s",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PlanAssessmentIntermediate": {
        "node_color": "#F5F5DC",
        "node_shape": "h",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PlanExistenceIntermediate": {
        "node_color": "#A9A9A9",
        "node_shape": "d",
        "node_size": 1000,
        "alpha": 0.9,
    },
    "PlanRoot": {
        "node_color": "#FFE4E1",
        "node_shape": "o",
        "node_size": 900,
        "alpha": 0.8,
    },
    "PlanIntermediate": {
        "node_color": "#DCDCDC",
        "node_shape": "s",
        "node_size": 850,
        "alpha": 0.9,
    },
    "PlanSink": {
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
    "PlanSummarizationIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashdot",
        "arrowsize": 8,
    },
    "PlanExclusionIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashed",
        "arrowsize": 8,
    },
    "PlanAssessmentIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashed",
        "arrowsize": 8,
    },
    "PlanExistenceIntermediate": {
        "edge_color": "black",
        "width": 1.5,
        "style": "dashed",
        "arrowsize": 8,
    },
    "PlanRoot": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "PlanIntermediate": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
    "PlanSink": {
        "edge_color": "gray",
        "width": 1.5,
        "style": "solid",
        "arrowsize": 10,
    },
}


class StapleVisualizer(BasicStructureVisualizer):
    """The visualizer for the p-RAR thought structure."""

    def create_node_draw_labels(self, graph: nx.DiGraph, node_pool: List[BasicNode]):
        """Create the labels of nodes for drawing the graph."""
        # Create the labels to be plotted
        labels = dict()
        for node_id in graph.nodes:
            node = node_pool[node_id]
            if node.position == "Root":
                labels[node.identity] = "Q"
            if node.position == "PlanRoot":
                labels[node.identity] = "Task"

            if node.position == "PlanSummarizationIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIS P-{node.step_idx}"

            if node.position == "NormalGenerationIntermediate":
                labels[node.identity] = f"N-{node.identity}\nS-{node.step_idx}"

            if node.position == "PlanExclusionIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIE S-{node.step_idx}"

            if node.position == "PlanAssessmentIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIA S-{node.step_idx}"

            if node.position == "PlanExistenceIntermediate":
                labels[node.identity] = f"N-{node.identity}\nIC S-{node.step_idx}"

            if node.position == "Intermediate":
                labels[node.identity] = f"N-{node.identity}\nS-{node.step_idx}"

            if node.position == "Sink":
                labels[node.identity] = f"N-{node.identity}\nS-{node.step_idx}"

            if node.position == "PlanIntermediate":
                labels[node.identity] = f"N-{node.identity}\nP-{node.step_idx}"

            if node.position == "PlanSink":
                labels[node.identity] = f"N-{node.identity}\nP-{node.step_idx}"

        return labels
