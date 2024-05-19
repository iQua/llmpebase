"""
A structure for the policy-thought of the p-RAQ.

This corresponds to the policy tree mentioned in the p-RAQ paper.
"""

import logging
from typing import List, Tuple
from dataclasses import dataclass

import networkx as nx
from transformers.utils import ModelOutput as FieldFrozenContainer

from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer


@dataclass
class BasicPolicyStep(FieldFrozenContainer):
    """
    A base policy step.
    """

    step_idx: int

    policy: str = None
    policy_name: str = None

    policy_num_visits: int = None

    # The previous thought that induces the policy
    # This corresponds the Upsilon of the paper
    thought_origins: List[str] = None
    # Rewards of the thought should hold two reward,
    # 1. n_wins and 2. v_llm and 3. n_visits
    thought_evaluations: List[List[float]] = None

    def extend_thought_origin(self, thought_origin: str):
        """Extend the thought origins."""
        if self.thought_origins is None:
            self.thought_origins = []

        if thought_origin not in self.thought_origins:
            self.thought_origins.append(thought_origin)

        return self.thought_origins.index(thought_origin)

    def extend_thought_evaluation(
        self, thought_evaluation: List[float], extend_idx: int = None
    ):
        """Extend the thought evaluations."""
        if self.thought_evaluations is None:
            self.thought_evaluations = []

        if extend_idx >= len(self.thought_evaluations):
            self.thought_evaluations.append(thought_evaluation)
        else:
            # Merge the thought evaluations to the existing one
            # Add their n_wins, v_llm and n_visits
            self.thought_evaluations[extend_idx] = [
                v1 + v2
                for v1, v2 in zip(
                    self.thought_evaluations[extend_idx], thought_evaluation
                )
            ]


@dataclass
class PolicyNode(BasicPolicyStep):
    """Node of the policy tree."""

    identity: str = None
    task_info: dict = None
    node_name: str = None
    position: str = None
    position_states: Tuple[str] = None

    # The auxiliary information for the node
    # This aims to store any additional information
    auxiliary: dict = None


@dataclass
class PolicyEdge(FieldFrozenContainer):
    """
    A basic edge used to present the information contained the edge of two
    adjacent nodes.
    """

    edge_id: str

    src_node_id: str = None
    dst_node_id: str = None
    edge_type: str = None

    auxiliary: dict = None


class PolicyTree(base.BaseStructure):
    """
    Policy tree holding the policy combinations of a task.
    """

    def __init__(
        self,
        logging_config: dict,
        visualizer: BasicStructureVisualizer = None,
    ):
        super().__init__(
            logging_config=logging_config,
            visualizer=visualizer,
        )
        self.save_foldername = "policy_tree_structure"

        # Tracker of the node id starting from 0
        # thus, root of the thought structure should be 0
        self.node_id_tracker = -1

        self.position_states = ("PolicyRoot", "PolicyIntermediate", "PolicySink")

    def create_node(
        self,
        step_idx: int,
        identity: str,
        policy: str,
        policy_num_visits: int = 0,
        task_info: dict = "",
        thought_origins: List[str] = None,
        thought_evaluations: List[List[float]] = None,
        policy_name: str = "Policy Step",
        node_name: str = "IntermediatePolicy Node",
        position: str = "PolicyIntermediate",
        position_states: Tuple[str] = None,
        auxiliary: dict = None,
    ):
        """Create a node."""

        assert isinstance(identity, str)

        return PolicyNode(
            step_idx=step_idx,
            identity=identity,
            task_info=task_info,
            policy=policy,
            policy_num_visits=policy_num_visits,
            thought_origins=thought_origins if thought_origins is not None else None,
            thought_evaluations=(
                thought_evaluations if thought_evaluations is not None else None
            ),
            policy_name=policy_name,
            node_name=node_name,
            position=position,
            position_states=(
                self.position_states if position_states is None else position_states
            ),
            auxiliary=auxiliary,
        )

    def create_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        edge_type="Policy Forwarding",
        edge_id=None,
        auxiliary: dict = None,
    ):
        """Create an edge."""
        assert isinstance(src_node_id, str) and isinstance(dst_node_id, str)

        return PolicyEdge(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            edge_type=edge_type,
            edge_id=edge_id,
            auxiliary=auxiliary,
        )

    def construct_root(
        self,
        task_info: dict,
        category_name: str = None,
        **kwargs,
    ):
        """
        Set the root of the structure.
        """

        identity = self.generate_node_id()

        self.root = self.create_node(
            step_idx=0,
            identity=identity,
            task_info=task_info,
            policy=category_name,
            policy_num_visits=0,
            thought_origins=None,
            policy_name="Root Empty Policy",
            node_name="Root Policy Node",
            position="PolicyRoot",
            auxiliary={},
        )

        self.graph = nx.DiGraph()
        self.node_pool = {identity: self.root}
        self.edge_pool = {}
        # Add the root node to the graph
        self.graph.add_node(identity)

        logging.info("Created the root node %s for the policy tree", identity)

    def generate_node_id(self):
        """Generate a node id."""

        new_id = self.node_id_tracker + 1

        # Avoid the duplication of the node id
        if self.node_pool is not None and str(new_id) in self.node_pool:
            node_ids = list(self.node_pool.keys())
            max_node_id = max([int(node_id) for node_id in node_ids])
            self.node_id_tracker = max_node_id
            new_id = self.node_id_tracker + 1

        self.node_id_tracker += 1
        new_id = str(new_id)

        return new_id

    def add_node(
        self,
        policy: str,
        policy_num_visits: int,
        prev_node_id: str,
        thought_origins: List[str],
        thought_evaluations: List[List[float]],
        **kwargs,
    ) -> int:
        """Adding one node to the thought structure."""

        assert isinstance(prev_node_id, str)

        node_id = self.generate_node_id()
        edge_id = self.generate_edge_id(prev_node_id, node_id)

        step_idx = self.node_pool[prev_node_id].step_idx + 1
        # Create the node
        new_node = self.create_node(
            identity=node_id,
            step_idx=step_idx,
            task_info=None,
            policy=policy,
            policy_num_visits=policy_num_visits,
            thought_origins=thought_origins,
            thought_evaluations=thought_evaluations,
            policy_name=f"Intermediate Policy {node_id}",
            node_name=f"Intermediate Policy Node {step_idx}",
            position="PolicyIntermediate",
            auxiliary={},
        )
        # Create a edge create_edge
        new_edge = self.create_edge(
            edge_id=edge_id,
            edge_type="Policy Forwarding",
            src_node_id=prev_node_id,
            dst_node_id=node_id,
            auxiliary={},
        )
        # Add node to the graph
        self.node_pool[node_id] = new_node
        self.edge_pool[edge_id] = new_edge
        self.graph.add_node(node_id)
        # Connect the node to the previous node
        self.graph.add_edge(
            prev_node_id,
            node_id,
            edge_id=edge_id,
            edge_type="Policy Forwarding",
        )

        logging.info(
            "  Created new %s policy node %s grown from the policy node %s",
            self.node_pool[node_id].position,
            node_id,
            prev_node_id,
        )

        return node_id

    def extend_node(
        self,
        node_id: str,
        thought_origin: str,
        thought_evaluation: Tuple[int, float],
    ):
        """
        Update the node by including the new thought origins and evaluations.
        """
        node = self.node_pool[node_id]
        # Added new thought origins and evaluations
        extend_idx = node.extend_thought_origin(thought_origin)
        node.extend_thought_evaluation(thought_evaluation, extend_idx)
        node.policy_num_visits += 1
        self.node_pool[node_id] = node

    # def set_node_sink(self, node_id: str, max_length: int = 3):
    #     """Set the node to be the stop node."""
    #     # Set the node to be the stop node
    #     self.node_pool[node_id].set_position("Sink")
