"""
A thought structure to support the thought rollback.
"""

import logging

import networkx as nx


from llmpebase.model.thought_structure import trees


class ThoughtRollbackStructure(trees.DFGTreeThoughtStructure):
    """
    A thought structure to perform the adaptive reasoning process by
    continuously rolling back of thoughts.
    To facilitate the reasoning, the basic structure is built upon a tree
    with depth-wise growth manner.
    """

    def __init__(
        self,
        thought_model,
        model_config,
        logging_config,
        visualizer,
    ):
        super().__init__(thought_model, model_config, logging_config, visualizer)

        # Get the configuration
        config = model_config["thought_structure"]
        # Get the maximum number of rollbacks to be performed
        # for each node
        self.num_max_rollbacks = config["num_max_rollbacks"]

        # A variable to track which node is being rolled back to
        # thus creating a new reasoning path from it.
        self.rolling_back_state = None

    def set_node_growth(self, node_id: int):
        """Set the node to be the growable one."""

        # Need to know that before setting the node growth,
        # the position of the node has been determined

        # Get all edges of one node
        node_edges = list(self.graph.edges(node_id))
        # Get the node ids that are the next steps of the reasoning step of the node
        forward_edges = [edge for edge in node_edges if edge[0] < edge[1]]
        # # Get the node ids that are the rollbacks of the node
        rollback_edges = self.get_rollback_edges(node_id)

        # Close growth of the node as this node has enough children
        # or has enough rollbacks
        if len(forward_edges) >= self.num_next_steps:
            # Set the node to be un-growable
            self.node_pool[node_id].set_growth("Un-growable")

        # For the sink nodes, when its #rollback does not reach the limit,
        # open its growth
        if self.is_node_sink(node_id):
            if len(rollback_edges) < self.num_max_rollbacks:
                # Open the growth of the node as the edge is the rollback edge
                self.node_pool[node_id].set_growth("Growable")
            else:
                # Set the node to be un-growable
                self.node_pool[node_id].set_growth("Un-growable")

    def get_grow_node(self):
        """Get the node to be grown next relying on the reasoning condition."""
        # Get the node based on depth-wise search
        node = super().get_grow_node()
        # Set the rolling back node to be None so that
        # clean the rollback state
        self.rolling_back_state = None

        if node is None:
            return node

        # Get the reasoning path from the root to the node
        nodes = self.get_node_path(
            src_node_id=self.root.identity, dst_node_id=node.identity
        )
        # When there is no thought but only the root in the structure,
        # no rollback is needed
        if len(nodes) - 1 == 0:
            return node

        (
            rollback_step_idx,
            rollback_result,
            analysis,
        ) = self.thought_model.generate_rollback(thought_chain=nodes)

        if rollback_step_idx is not None:
            # Rollback to one previous node of the bad node
            rollback_node = nodes[rollback_step_idx - 1]
            logging.info(
                "Roll back from node %s (Step %s) to node %s (Step %s)",
                node.identity,
                node.step_idx,
                rollback_node.identity,
                rollback_node.step_idx,
            )

            # Set the rollback node to be growable there by allowing to
            # generate new reasoning path from it
            self.node_pool[rollback_node.identity].set_growth("Growable")

            edge_id = self.generate_edge_id(node.identity, rollback_node.identity)
            # Create the unique for this rollback edge
            rollback_edges = self.get_rollback_edges(node.identity)
            n_rollbacks = len(rollback_edges)
            edge_id = f"{edge_id}_R{n_rollbacks+1}"

            new_edge = self.create_edge(
                edge_id=edge_id,
                edge_type="Rollback",
                src_node_id=node.identity,
                dst_node_id=rollback_node.identity,
                reasoning_prompt=self.thought_model.prompter.rollback_controller_prompt,
                evaluation_prompt=self.thought_model.prompter.rollback_analysis_prompt,
                edge_score=1.0,
                auxiliary={
                    "RollbackResult": rollback_result,
                    "RollbackAnalysis": analysis,
                    "RollbackCondition": f"Error in Step {rollback_step_idx}",
                },
            )
            # Add the edge to the pool
            self.edge_pool[edge_id] = new_edge
            # Add the edge between the node and the rollback node
            self.graph.add_edge(
                node.identity,
                rollback_node.identity,
                edge_type="Rollback",
                edge_id=edge_id,
                weight=2.0,
            )
            self.rolling_back_state = {"node": rollback_node, "rollback_edge": edge_id}

            return rollback_node

        else:
            if self.is_node_sink(node.identity):
                # Close the growth of the node as the node does not need
                # to be rolled back
                self.node_pool[node.identity].set_growth("Un-growable")

                node = self.get_grow_node()
                return node

        return node

    def add_node(
        self,
        thought: str,
        prev_node_id: str,
        thought_score: float = 1.0,
        edge_weight: float = 1.0,
        **kwargs,
    ) -> int:
        """Adding one node to the tree."""

        node_id = super().add_node(
            thought=thought,
            prev_node_id=prev_node_id,
            thought_score=thought_score,
            edge_weight=edge_weight,
            **kwargs,
        )

        # Check whether the new node is created from the node
        # that has been rolled back to
        if (
            self.rolling_back_state is not None
            and self.rolling_back_state["node"].identity == prev_node_id
        ):
            rollback_edge_id = self.rolling_back_state["rollback_edge"]
            # Add more information to the reasoning edge by allowing
            # one to know that the node is created from which rollback,
            # i.e., From Rollback
            edge_id = self.generate_edge_id(prev_node_id, node_id)
            edge = self.edge_pool[edge_id]
            edge.auxiliary["FromRollback"] = rollback_edge_id
            self.edge_pool[edge_id] = edge

            # Add the corresponding information to the edge of the graph
            # The return edge data of graph[prev_node_id][node_id] will be a tuple
            # in which key is the edge idx while the value is a dict containing the
            # attributes of the edge
            graph_edge = self.graph[prev_node_id][node_id]
            graph_edge[0]["FromRollback"] = rollback_edge_id

        return node_id

    def get_rollback_edges(self, node_id: int):
        """Check if the node has reached the rollback limit."""
        # Get all edges of one node
        node_edges = list(self.graph.edges(node_id))
        # Get the node ids that are the rollbacks of the node
        return [edge for edge in node_edges if edge[0] >= edge[1]]
