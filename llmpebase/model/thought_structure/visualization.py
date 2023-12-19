"""
An implementation to visualize the thought structure.
"""
import os
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from llmpebase.model.thought_structure.structure_generic import BasicNode


node_config = {
    "Root": {
        "node_color": "#228B22",
        "node_shape": "o",
        "node_size": 1100,
        "alpha": 0.9,
    },
    "Intermediate": {
        "node_color": "#6495ED",
        "node_shape": "s",
        "node_size": 1200,
        "alpha": 0.9,
    },
    "Stop": {
        "node_color": "#F5DEB3",
        "node_shape": "8",
        "node_size": 1100,
        "alpha": 0.9,
    },
}

edge_config = {
    "Root": {
        "edge_color": "black",
        "width": 1.111111110,
        "style": "solid",
        "arrowsize": 20,
    },
    "Intermediate": {
        "edge_color": "gray",
        "width": 1.0,
        "style": "solid",
        "arrowsize": 20,
    },
    "Stop": {
        "edge_color": "green",
        "width": 1.0,
        "style": "solid",
        "arrowsize": 20,
    },
}

labels_config = {
    "font_size": 10,
    "font_color": "black",
    "font_family": "sans-serif",
    "font_weight": "normal",
}


class BasicStructureVisualizer:
    """A visualizer to visualize the thought structure."""

    def __init__(
        self, logging_config: str, visualization_name: str = "thought_structure"
    ):
        visualization_path = logging_config["visualization_path"]
        self.visualization_name = visualization_name
        self.save_path = f"{visualization_path}/{self.visualization_name}"
        os.makedirs(self.save_path, exist_ok=True)

        self.num_visual = 1

    def draw_graph(
        self,
        ax: plt.axes,
        graph: nx.Graph,
        node_pool: List[BasicNode],
    ):
        """Visualize the thought structure.
        This function plots the structure in the tree format, which
        relies on the `dot` of `graphviz`.
        """

        # Get the positions
        pos = graphviz_layout(graph, prog="dot")
        labels = {
            node_pool[node_id].identity: f"Step {node_pool[node_id].step_idx}"
            for node_id in graph.nodes
        }
        for node_id in graph.nodes:
            node = node_pool[node_id]
            pos_type = node.position
            node_id = node.identity
            # Draw all nodes
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node_id],
                ax=ax,
                **node_config[pos_type],
            )
            # Draw edges of Root
            edges = [(node_id, n) for n in graph.neighbors(node_id)]
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=edges,
                ax=ax,
                **edge_config[pos_type],
            )
            # A root node is a node with no predecessors
            if graph.in_degree(node_id) == 0:
                labels[node_id] = "Start"

        nx.draw_networkx_labels(graph, pos, labels, ax=ax, **labels_config)

        return ax

    def visualize(
        self, graph: nx.Graph, node_pool: List[BasicNode], save_name: str = None
    ):
        """Plot the thought structure."""
        whole_fig, ax = plt.subplots()
        ax = self.draw_graph(ax, graph=graph, node_pool=node_pool)
        ax.axis("off")
        self.num_visual += 1
        plt.show(block=False)
        plt.pause(2)

        self.save_fig(fig=whole_fig, save_name=save_name)

        plt.close()

    def save_fig(self, fig: plt.figure, save_name: str):
        """Save the figure."""

        fig.tight_layout()

        # Save the figure
        save_name = (
            save_name if save_name is not None else f"visual_{str(self.num_visual)}"
        )
        fig.savefig(f"{self.save_path}/{save_name}.png")
        fig.savefig(f"{self.save_path}/{save_name}.pdf")
