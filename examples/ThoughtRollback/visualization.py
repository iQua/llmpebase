"""
Visualization of the thought structure of the ThoughtRollback framework.
"""
from typing import List

import matplotlib.pyplot as plt
import networkx as nx

from llmpebase.model.thought_structure.visualization import (
    BasicStructureVisualizer,
    edge_config,
    node_config,
)
from llmpebase.model.thought_structure.structure_generic import BasicNode

rollback_config = {"arc_rad": 0.25}

rollback_edge_config = {
    "edge_color": "red",
    "width": 1.0,
    "style": "dashed",
    "arrowsize": 10,
}
rollback_edge_labels_config = {
    "font_size": 10,
    "font_family": "sans-serif",
    "font_color": "red",
    "font_weight": "normal",
}


def draw_rollback_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


class TRVisualizer(BasicStructureVisualizer):
    def draw_node_edges(
        self,
        ax: plt.axes,
        graph: nx.Graph,
        pos: dict,
        node: BasicNode,
    ):
        """Draw the edges of one node of the graph.

        There are two types of edges:
         - reasoning edge, i.e., the straight line
         - rollback edge, i.e., the curve dashed line
        """

        pos_type = node.position
        node_id = node.identity

        # Get the straight line edge
        # Get all edges of one node
        node_edges = list(graph.edges(node_id))
        # Get the node ids that are the rollbacks of the node
        rollback_edges = [edge for edge in node_edges if int(edge[0]) >= int(edge[1])]
        reasoning_edges = [edge for edge in node_edges if int(edge[0]) < int(edge[1])]

        # Draw edges of the node
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=reasoning_edges,
            ax=ax,
            **edge_config[pos_type],
            node_size=node_config[pos_type]["node_size"],
            # node_shape=node_config[pos_type]["node_shape"],
        )
        arc_rad = rollback_config["arc_rad"]
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=rollback_edges,
            **rollback_edge_config,
            connectionstyle=f"arc3, rad = {arc_rad}",
            node_size=node_config[pos_type]["node_size"],
            # node_shape=node_config[pos_type]["node_shape"],
        )

        return ax

    def create_edge_draw_labels(self, graph: nx.Graph, node_pool: List[BasicNode]):
        """Create the labels of edges for drawing the graph."""
        # Create the labels to be plotted
        labels = {}
        for node_id in graph.nodes:
            # Get all edges of one node
            node_edges = list(graph.edges(node_id, data=True, keys=True))
            # Get and label the following edges:
            # 1. the reasoning edges whose attribute shows that it is derived from
            #   a rollback
            # Each data is a tuple containing: (src, dst, key, attr_dict)
            # Add labels for the case 1
            # Create the label as R1, R2... for the rollback edges where
            # 1, 2,..., here means the index of the rollback

            # Add labels for the case 2
            labels.update(
                {
                    edge_data[:2]: edge_data[-1]["FromRollback"]
                    for edge_data in node_edges
                    if edge_data[-1]["edge_type"] == "Reasoning"
                    and "FromRollback" in edge_data[-1]
                }
            )
        return labels

    def create_rollback_edge_draw_labels(
        self, graph: nx.Graph, node_pool: List[BasicNode]
    ):
        """Create the labels of rollback edges for drawing the graph."""
        # Create the labels to be plotted
        labels = {}
        for node_id in graph.nodes:
            # Get all edges of one node
            node_edges = list(graph.edges(node_id, data=True, keys=True))
            # Get and label the following edges:
            # 1. the rollback edges

            # Each data is a tuple containing: (src, dst, key, attr_dict)
            # Add labels for the case 1
            # Create the label as R1, R2... for the rollback edges where
            # 1, 2,..., here means the index of the rollback
            labels.update(
                {
                    edge_data[:2]: edge_data[-1]["edge_id"].split("_")[-1]
                    for edge_data in node_edges
                    if edge_data[-1]["edge_type"] == "Rollback"
                }
            )

        return labels

    def draw_edge_labels(self, graph, node_pool, pos, ax):
        """Draw the labels of the edges to the graph."""
        # Plot the labels of the reasoning edges
        labels = self.create_edge_draw_labels(graph=graph, node_pool=node_pool)
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=labels, ax=ax, **rollback_edge_labels_config
        )

        # Plot the labels of the rollback edges
        rollback_edge_labels = self.create_rollback_edge_draw_labels(graph, node_pool)

        draw_rollback_edge_labels(
            graph,
            pos,
            edge_labels=rollback_edge_labels,
            rad=rollback_config["arc_rad"],
            ax=ax,
            **rollback_edge_labels_config,
        )
        return ax
