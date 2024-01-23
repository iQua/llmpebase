"""
Plot partial graph.
"""

import visualization

import networkx as nx
from llmpebase.model.thought_structure import base


from llmpebase.config import Config


logging_config = Config.items_to_dict(Config().logging._asdict())


sample_idx = 4
structure_path = logging_config["result_path"]
structure_folder = f"thought_structure_{sample_idx}"
graph, node_pool, edge_pool = base.BaseThoughtStructure.resume_structure(
    location=f"{structure_path}/backups/{structure_folder}"
)
visualizer = visualization.TRVisualizer(
    logging_config=logging_config, visualization_foldername=f"reload_{structure_folder}"
)
print(graph)

# Only show part of the graph

included_nodes = ["0", "1", "3", "4", "5", "6", "8"]
included_edges = ["0->1", "1->3", "3->4", "4->6", "3-5", "5->8", "6->3_R1"]
splitted_edges = []
for edge in included_edges:
    edge_nodes = edge.split("->")
    if "R" in edge:
        splitted_edges.append([edge_nodes[0], edge_nodes[1].split("_")[0]])
    else:
        splitted_edges.append(edge_nodes)

G2 = nx.MultiDiGraph()

# Add the selected nodes with their data and their edges to G2
for node in included_nodes:
    # Add node with data
    G2.add_node(node, **graph.nodes[node])

for u, v, key, data in graph.edges(keys=True, data=True):
    if any([u == edge[0] and v == edge[1] for edge in splitted_edges]):
        # Add edge with data
        G2.add_edge(u, v, key=key, **data)

visualizer.visualize(graph=G2, node_pool=node_pool, save_name="reload_structure")
