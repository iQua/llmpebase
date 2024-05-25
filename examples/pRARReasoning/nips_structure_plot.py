from nips_thought_structure import NIPSPlanStructure
from plan_tree import PlanTree, PlanNode
from visualization import PRARVisualizer, node_config, edge_config

import networkx as nx
from llmpebase.model.thought_structure import base

data_path = "NIPS/plan_tree"
plot_path = "NIPS/plot"


if __name__ == "__main__":

    logging_config = {"result_path": plot_path, "visualization_path": plot_path}
    plan_tree = PlanTree(
        logging_config=logging_config,
        visualizer=PRARVisualizer(
            logging_config=logging_config,
            plot_config={"node_config": node_config, "edge_config": edge_config},
        ),
    )

    plan_tree.load_structure(location=data_path)

    # # Change the node id of the tree
    mapper = {"1": "24", "2": "56", "3": "13", "4": "67", "5": "49"}
    # print(plan_tree.node_pool)
    # Change the node pool
    tmpl_pool = plan_tree.node_pool.copy()
    for node_id in plan_tree.node_pool:
        if node_id in mapper:
            node = tmpl_pool[node_id]
            old_id = node.identity
            new_id = mapper[node.identity]
            tmpl_pool[new_id] = tmpl_pool.pop(old_id)
            # Change the data of the new node
            tmpl_pool[new_id].identity = new_id
    plan_tree.node_pool = tmpl_pool

    # Change the edage
    tmpl_pool = plan_tree.edge_pool.copy()
    for edge_id in plan_tree.edge_pool:
        edge_id_src = edge_id.split("->")[0]
        edge_id_dst = edge_id.split("->")[1]

        new_src = mapper[edge_id_src] if edge_id_src in mapper else edge_id_src
        new_dst = mapper[edge_id_dst] if edge_id_dst in mapper else edge_id_dst
        new_edge_id = f"{new_src}->{new_dst}"

        tmpl_pool[new_edge_id] = tmpl_pool.pop(edge_id)
        # Change the data of the new edge
        tmpl_pool[new_edge_id].edge_id = new_edge_id
        tmpl_pool[new_edge_id].src_node_id = new_src
        tmpl_pool[new_edge_id].dst_node_id = new_dst

    plan_tree.edge_pool = tmpl_pool

    plan_tree.graph = nx.relabel_nodes(plan_tree.graph, mapper)
    plan_tree.visualizer.visualize(
        plan_tree.graph,
        plan_tree.node_pool,
        save_name="plan_built_structure",
    )
