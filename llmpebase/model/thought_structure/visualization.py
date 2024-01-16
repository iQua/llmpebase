"""
An implementation to visualize the thought structure.
"""
import os
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from llmpebase.model.thought_structure.structure_generic import BasicNode


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
}

node_labels_config = {
    "font_size": 10,
    "font_color": "black",
    "font_family": "sans-serif",
    "font_weight": "normal",
}

edge_labels_config = {
    "font_size": 6,
    "font_family": "sans-serif",
    "font_color": "black",
    "font_weight": "normal",
}


class BasicStructureVisualizer:
    """A visualizer to visualize the thought structure."""

    def __init__(
        self, logging_config: str, visualization_foldername: str = "thought_structure"
    ):
        self.visualization_path = logging_config["visualization_path"]
        self.visualization_foldername = visualization_foldername

        self.layout = (
            "dot"
            if "thought_structure" not in logging_config
            or "layout" not in logging_config["thought_structure"]
            else logging_config["thought_structure"]["layout"]
        )

    def draw_node(
        self,
        ax: plt.axes,
        graph: nx.DiGraph,
        pos: dict,
        node: BasicNode,
    ):
        """Draw a node of the graph."""
        node_id = node.identity
        pos_type = node.position
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[node_id],
            ax=ax,
            **node_config[pos_type],
        )
        return ax

    def draw_node_edges(
        self,
        ax: plt.axes,
        graph: nx.DiGraph,
        pos: dict,
        node: BasicNode,
    ):
        """Draw the edges of one node of the graph."""

        pos_type = node.position
        node_id = node.identity

        # Draw edges of the node
        edges = [(node_id, n) for n in graph.neighbors(node_id)]
        # The 'node_shape' is commented out as it will make
        # the arrow not stick to the node
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges,
            ax=ax,
            node_size=node_config[pos_type]["node_size"],
            # node_shape=node_config[pos_type]["node_shape"],
            **edge_config[pos_type],
        )

        return ax

    def create_node_draw_labels(self, graph: nx.DiGraph, node_pool: List[BasicNode]):
        """Create the labels of nodes for drawing the graph."""
        # Create the labels to be plotted

        labels = {
            node_pool[node_id].identity: "Q"
            if graph.in_degree(node_id) == 0
            else f"N-{node_pool[node_id].identity}\n S-{node_pool[node_id].step_idx}"
            for node_id in graph.nodes
        }
        return labels

    def create_edge_draw_labels(self, graph: nx.DiGraph, node_pool: List[BasicNode]):
        """Create the labels of edges for drawing the graph."""
        # Create the labels to be plotted
        labels = {}
        return labels

    def draw_edge_labels(self, graph, node_pool, pos, ax):
        """Draw edge labels of the graph."""
        # Plot the labels of the edges
        labels = self.create_edge_draw_labels(graph=graph, node_pool=node_pool)
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=labels, ax=ax, **edge_labels_config
        )

        return ax

    def draw_graph(
        self,
        ax: plt.axes,
        graph: nx.DiGraph,
        node_pool: List[BasicNode],
    ):
        """Visualize the thought structure.
        This function plots the structure in the tree format, which
        relies on the layout of `graphviz`.
        """

        # Get the positions
        pos = graphviz_layout(graph, prog=self.layout)

        for node_id in graph.nodes:
            node = node_pool[node_id]

            node_id = node.identity
            # Draw the node
            ax = self.draw_node(ax=ax, graph=graph, pos=pos, node=node)
            # Draw edges of the node
            ax = self.draw_node_edges(ax=ax, graph=graph, pos=pos, node=node)

        # Plot the labels of the nodes
        labels = self.create_node_draw_labels(graph=graph, node_pool=node_pool)
        nx.draw_networkx_labels(graph, pos, labels, ax=ax, **node_labels_config)

        # Plot the labels of the edges
        ax = self.draw_edge_labels(graph=graph, node_pool=node_pool, pos=pos, ax=ax)

        return ax

    def visualize(
        self,
        graph: nx.DiGraph,
        node_pool: List[BasicNode],
        save_name: str = None,
    ):
        """Plot the thought structure."""

        whole_fig, ax = plt.subplots()
        ax = self.draw_graph(ax, graph=graph, node_pool=node_pool)
        ax.axis("off")

        plt.show(block=False)
        plt.pause(1)

        self.save_fig(fig=whole_fig, save_name=save_name)

        plt.close("all")

    def save_fig(self, fig: plt.figure, save_name: str):
        """Save the figure."""
        save_path = f"{self.visualization_path}/{self.visualization_foldername}"
        os.makedirs(save_path, exist_ok=True)

        fig.tight_layout()

        # Save the figure
        save_name = save_name if save_name is not None else "structure_visualization"
        fig.savefig(f"{save_path}/{save_name}.png")
        fig.savefig(f"{save_path}/{save_name}.pdf")
