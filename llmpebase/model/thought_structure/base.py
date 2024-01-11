"""
An implementation of the base thought step and base thought structure.
Based on existing literature, the thought is defined as a coherent language 
sequence that serves as an intermediate step toward problem solving. Thus,
we call it 'thoughtstep' to show that it is a thought that serves as a step
in the reasoning chain where a chain is presented as a path of the structure.
"""
import os
import json
import logging
from typing import List, Union, Tuple, Dict

import torch
import networkx as nx

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer
from llmpebase.model.thought_structure.structure_generic import BasicNode, BasicEdge


class BaseThoughtStructure:
    """
    A base thought structure performing as the fundamental framework
    in which each node is a thought step.
    """

    def __init__(
        self,
        thought_model: torch.nn.Module,
        model_config: dict,
        logging_config: dict,
        visualizer: BasicStructureVisualizer = None,
    ):
        # The thought model is the necessary part for the thought
        # structure building
        self.thought_model = thought_model
        assert hasattr(self.thought_model, "generate_thoughts")
        assert hasattr(self.thought_model, "evaluate_thoughts")
        assert hasattr(self.thought_model, "measure_thought_similarity")
        # The visualizer to visualize the thought structure
        self.visualizer = visualizer
        # Tracker of the node id starting from 0
        # thus, root of the tree should be 0
        self.node_id_tracker = -1
        # Root node of the structure
        # The root node contains the most common thought behaving
        # as the basis of the structure
        # Therefore, all subsequent paths/branches/sub-components of the
        # structure should be derived from the root node
        # For example, the root node contains the basic prompt
        # for this structure. Thus, with this root node, any subsequent
        # node can access the prompt by including this root node as the
        # starting point of the path/branch
        self.root: BasicNode = None
        # We utilize the directed multi-graph to represent the structure
        # as this is the most common architecture from which other
        # structures can derive
        # For example, chain and tree are special cases of graph
        self.graph: nx.Graph = None
        # The node pool to store all customized nodes in the structure
        self.node_pool: Dict[str, BasicNode] = None
        # The edge pool to store all customized edges in the structure
        self.edge_pool: Dict[str, BasicEdge] = None

        self.position_types = ("Root", "Intermediate", "Sink")
        self.growth_types = ("Growable", "Un-growable")

        # Get the configuration
        config = model_config["thought_structure"]

        # Number of next reasoning steps
        self.num_next_steps = (
            1 if "num_next_steps" not in config else config["num_next_steps"]
        )
        # Max #length of the reasoning chain, i.e., path the structure
        self.max_length = 3 if "max_length" not in config else config["max_length"]
        # Max #solution existed in the thought structure
        self.max_stops = 1 if "max_stops" not in config else config["max_stops"]

        # Threshold setup between thoughts

        # The min similarity between two thoughts that
        # they are regarded as identical
        # this is generally the value generated by
        # LLMs.
        self.min_thought_sim = (
            0.8
            if "min_thought_similarity" not in config
            else config["min_thought_similarity"]
        )
        # The max score difference between
        # two thoughts that they are regarded as identical
        self.max_score_diff = (
            0.1
            if "max_score_difference" not in config
            else config["max_score_difference"]
        )

        # Threshold setup for stopping the structure growing
        self.min_stop_score = (
            0.3 if "min_stop_score" not in config else config["min_stop_score"]
        )
        # We set the max score to be high as most
        # Llm is confident on its thought
        self.max_stop_score = (
            0.9 if "max_stop_score" not in config else config["max_stop_score"]
        )
        self.save_path = logging_config["result_path"]
        self.save_foldername = "thought_structure"

    def create_node(
        self,
        identity: str,
        step_idx: int,
        thought: Union[str, BasicSamplePrompt],
        thought_score: float = 1.0,
        step_name: str = "Reasoning Step",
        node_name: str = "Intermediate Node",
        position: str = "Intermediate",
        growth: bool = "Growable",
        **kwargs,
    ):
        """Create a node."""

        return BasicNode(
            identity=identity,
            step_idx=step_idx,
            thought=thought,
            thought_score=thought_score,
            step_name=step_name,
            node_name=node_name,
            position=position,
            growth=growth,
            position_types=self.position_types,
            growth_types=self.growth_types,
            similar_thoughts=[],
            similar_thought_scores=[],
            similar_thought_similarity=[],
            thought_similarity_prompt=[],
        )

    def create_edge(
        self,
        src_node_id: int,
        dst_node_id: int,
        reasoning_prompt="",
        evaluation_prompt="",
        edge_score: float = 1.0,
        edge_id=None,
    ):
        """Create an edge."""

        return BasicEdge(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            reasoning_prompt=reasoning_prompt,
            evaluation_prompt=evaluation_prompt,
            edge_score=edge_score,
            edge_id=edge_id,
        )

    def construct_root(
        self,
        thought: BasicSamplePrompt = None,
        thought_score: float = None,
        **kwargs,
    ):
        """Set the root of the structure.

        Only by creating the structure's root node can it be grown until it is built.
        Generally, the task prompt and the request containing the question and
        """

        identity = self.generate_node_id()

        self.root = self.create_node(
            identity=identity,
            step_idx=0,
            thought=thought,
            thought_score=thought_score,
            step_name="Initial Step",
            node_name="Root Node",
            position="Root",
            growth="Growable",
        )

        self.graph = nx.MultiDiGraph()
        self.node_pool = {identity: self.root}
        self.edge_pool = {}
        # Add the root node to the graph
        self.graph.add_node(identity)

        logging.info("Created the root node %s", identity)

    def generate_node_id(self):
        """Generate a node id."""
        new_id = self.node_id_tracker + 1
        self.node_id_tracker += 1
        new_id = str(new_id)
        return new_id

    def generate_edge_id(self, src_node_id: int, dst_node_id: int):
        """Generate an edge id."""
        return f"{src_node_id}->{dst_node_id}"

    def is_duplicated_path(
        self,
        node1_id: int,
        node2_id: int,
    ):
        """
        Two nodes are regarded as the same path when they are derived
        from a same path, which is the shortest path
        between the node and the root.

        We should note that this judgement constrain a very strong and
        strict as it requires the node id along the path of these two nodes
        are the same, making them actually grown from the same parent node.

        One may need to enhance or adjust this function to make it more
        flexible.
        """
        return nx.shortest_path(
            self.graph, self.root.identity, node1_id
        ) == nx.shortest_path(self.graph, self.root.identity, node2_id)

    def compute_thought_similarity(
        self,
        grow_node: BasicNode,
        thoughts: List[str],
    ):
        """
        Measure the similarity between the given thoughts and thoughts in the structure.

        :param grow_node: The node the thoughts should be added to.
        :param thoughts: The thoughts to be added.
        """
        similarities = [{}] * len(thoughts)
        prompts = [{}] * len(thoughts)
        grow_path = self.get_node_path(
            src_node_id=self.root.identity, dst_node_id=grow_node.identity
        )
        for idx, thought in enumerate(thoughts):
            for node_id in self.graph.nodes:
                score = 0.0
                prompt = ""
                node = self.root
                if node_id != self.root.identity:
                    node = self.node_pool[node_id]
                    score, prompt = self.thought_model.measure_thought_similarity(
                        thought,
                        node.thought,
                        thought_chain=grow_path,
                    )
                similarities[idx][node.identity] = score
                prompts[idx][node.identity] = prompt
        return similarities, prompts

    def search_identical_thought(
        self,
        thought: str,
        prev_node_id: int,
        thought_score: float,
        thought_similarities: dict,
    ) -> List[int]:
        """
        Search the graph to obtain the identical thought.

        :param thought: The thought to be searched.
        :param prev_node_id: The id of the node containing previous thought of
         the input thought.
        :param thought_score: The evaluation score of the thought.
         Default to be 1.0 to indicate that the thought is trusted.
        :param thought_similarity: Thought similarity between the thought and
         all existing thoughts in nodes.
        """

        identical_nodes = []
        for node_id in self.graph.nodes:
            node = self.node_pool[node_id]
            # Skip the root node
            if node.identity == self.root.identity:
                continue
            node_thought = node.thought
            # Get the similarity score between the visited node and the
            # input thought
            similarity_score = (
                thought_similarities[node.identity]
                if node.identity in thought_similarities
                else 0.0
            )
            # Get the difference between the evaluation score of the visited node
            # and the input thought score
            evaluation_diff = abs(node.thought_score - thought_score)

            if (
                thought != node_thought
                and similarity_score >= self.min_thought_sim
                and evaluation_diff <= self.max_score_diff
                and self.is_duplicated_path(node.identity, prev_node_id)
            ):
                identical_nodes.append(node.identity)

        return identical_nodes

    def add_node(
        self,
        thought: str,
        prev_node_id: str,
        thought_score: float = 1.0,
        edge_weight: float = 1.0,
        **kwargs,
    ) -> int:
        """Adding one node to the tree."""

        node_id = self.generate_node_id()
        edge_id = self.generate_edge_id(prev_node_id, node_id)

        step_idx = self.node_pool[prev_node_id].step_idx + 1
        # Create the node
        new_node = self.create_node(
            identity=node_id,
            step_idx=step_idx,
            thought=thought,
            thought_score=thought_score,
            node_name=f"Intermediate Node {node_id}",
            step_name=f"Reasoning step {step_idx}",
            position="Intermediate",
            growth="Growable",
        )
        # Create a edge create_edge
        new_edge = self.create_edge(
            edge_id=edge_id,
            src_node_id=prev_node_id,
            dst_node_id=node_id,
            reasoning_prompt=kwargs["reasoning_prompt"],
            evaluation_prompt=kwargs["evaluation_prompt"],
            edge_score=edge_weight,
        )
        # Add node to the graph
        self.node_pool[node_id] = new_node
        self.edge_pool[edge_id] = new_edge
        self.graph.add_node(node_id)
        # Connect the node to the previous node
        self.graph.add_edge(prev_node_id, node_id, weight=edge_weight)

        # Set the node status
        self.set_node_status(new_node.identity)

        logging.info(
            "  Created new %s node (%s) %s grown from the node %s",
            self.node_pool[node_id].position,
            self.node_pool[node_id].growth,
            node_id,
            prev_node_id,
        )

        return node_id

    def extend_node(
        self,
        thought: str,
        thought_score: float,
        node_ids: List[int],
        thought_similarity: Dict[str, float],
        similarity_prompts: Dict[str, str],
        **kwargs,
    ):
        """Extend the node by adding similar thoughts to it."""
        node_id = node_ids[0]
        sim_score = thought_similarity[node_id]
        sim_prompt = similarity_prompts[node_id]

        self.node_pool[node_id].backup_though(
            thought, thought_score, similarity_score=sim_score, prompt=sim_prompt
        )

        logging.info(
            "  Backed it up to an existing node %s",
            node_id,
        )
        return node_id

    def add_thought(
        self,
        thought: str,
        thought_score: float,
        prev_node_id: int,
        to_node_ids: List[int],
        similarities: Dict[str, float],
        similarity_prompt: Dict[str, str],
        **kwargs,
    ):
        """Add the thought to the structure."""
        logging.info("Adding the thought derived from the node %s", prev_node_id)
        # If there is no identical thought, add the thought to the structure
        # as a new node
        if len(to_node_ids) == 0:
            node_id = self.add_node(thought, prev_node_id, thought_score, **kwargs)
        else:
            node_id = self.extend_node(
                thought,
                thought_score,
                node_ids=to_node_ids,
                thought_similarity=similarities,
                similarity_prompts=similarity_prompt,
                prev_node_id=prev_node_id,
            )

        return node_id

    def grow_structure(
        self,
        prev_node_id: int,
        thoughts: List[str],
        thought_scores: List[float],
        thought_similarities: List[dict],
        **kwargs,
    ):
        """Grow the structure by adding new thoughts.

        :param prev_node_id: The node id, which produces the thoughts as the next
         steps.
        :param thoughts: The thoughts to be added.
        :param thought_scores: The evaluation scores of thoughts.
        :param thought_similarities: The similarity scores in which each item is a list
         containing the similarity scores between the corresponding thought and all existing
         thoughts in nodes of the structure.
        """
        similarity_prompts = (
            kwargs["similarity_prompts"]
            if "similarity_prompts" in kwargs
            else [{}] * len(thoughts)
        )
        for idx, (thought, score, similarities) in enumerate(
            zip(thoughts, thought_scores, thought_similarities)
        ):
            # Judge whether the prev_node is full
            # if true, there is no need to add more thoughts either
            # as new node (impossible) or as similar thoughts (unnecessary)
            if not self.is_node_growable(prev_node_id):
                break
            # Find which nodes contain the similar thought with this
            # to be added thought
            similar_node_ids = []
            # Only search similar thought when the current thought
            # is not the solution
            if not self.root.thought.solution_flag in str(thought):
                similar_node_ids = self.search_identical_thought(
                    thought,
                    prev_node_id=prev_node_id,
                    thought_score=score,
                    thought_similarities=similarities,
                )
            # Add the thought to the structure either as a new node
            # or being added to 'to_nodes' based on the similarity
            # measurement
            self.add_thought(
                thought=thought,
                thought_score=score,
                prev_node_id=prev_node_id,
                to_node_ids=similar_node_ids,
                similarities=similarities,
                similarity_prompt=similarity_prompts[idx],
                **kwargs,
            )

            self.set_node_status(prev_node_id)

        # Set the status of nodes in the graph
        for node_id in self.graph.nodes:
            self.set_node_status(node_id)

        # Set the status of edges in the graph
        self.save_structure()

    def build_structure(
        self,
        **kwargs,
    ):
        """Grow the structure by adding new thoughts.

        :param thought_model: A defined thought model used to build the structure.
        """

        while not self.stop_growth():
            # Get the node to be grown
            grow_node = self.get_grow_node()
            # Get the thought path of the node to be grown
            thought_path = self.get_node_path(self.root.identity, grow_node.identity)
            # Generate and then evaluate the next thoughts
            thoughts, gen_prompt = self.thought_model.generate_thoughts(
                thought_chain=thought_path, num_thoughts=self.num_next_steps
            )

            scores = [None] * len(thoughts)
            eval_prompt = None
            scores, eval_prompt = self.thought_model.evaluate_thoughts(
                thoughts, thought_chain=thought_path
            )

            # Measure the similarity between new thoughts and existing thoughts in the
            # structure
            similarities = [{}] * len(thoughts)
            sim_prompts = [{}] * len(thoughts)
            similarities, sim_prompts = self.compute_thought_similarity(
                grow_node, thoughts
            )

            # Grow the structure by adding the thoughts
            self.grow_structure(
                prev_node_id=grow_node.identity,
                thoughts=thoughts,
                thought_scores=scores,
                thought_similarities=similarities,
                reasoning_prompt=gen_prompt,
                evaluation_prompt=eval_prompt,
                similarity_prompts=sim_prompts,
            )

            # Draw the graph and save to the disk
            # as each node added to the graph means a new step
            if self.visualizer is not None:
                self.visualizer.visualize(
                    self.graph,
                    self.node_pool,
                    save_name=f"Step_{grow_node.step_idx + 1}",
                )

        # Draw the whole graph after building
        if self.visualizer is not None:
            self.visualizer.visualize(
                self.graph, self.node_pool, save_name="built_structure"
            )

    def get_node_path(
        self, src_node_id: int, dst_node_id: int = None
    ) -> List[BasicNode]:
        """Organize the thoughts towards target node."""

        node_ids = nx.shortest_path(self.graph, src_node_id, dst_node_id)
        return [self.node_pool[node_id] for node_id in node_ids]

    def get_path_scores(self, path: List[BasicNode]) -> List[Tuple[float, None]]:
        """Get the scores of the path."""
        # Skip the thought score of the root node as it will be None
        return [node.thought_score for node in path[1:]]

    def get_grow_node(self) -> BasicNode:
        """Get which node to be grown."""
        node = None

        for node_id in self.graph.nodes:
            if self.node_pool[node_id].growth == "Growable":
                node = self.node_pool[node_id]
                break
        return node

    def get_stop_nodes(self) -> List[BasicNode]:
        """Get the stop nodes of the thought structure.

        From the perspective of reasoning, the stop node means
        the node containing the thought that can be regarded as the
        solution to the reasoning.
        """
        return [
            self.node_pool[node_id]
            for node_id in self.graph.nodes
            if self.node_pool[node_id].position == "Sink"
        ]

    def set_node_stop(self, node_id: int):
        """Set the node to be the step node."""
        length = len(self.get_node_path(self.root.identity, node_id))
        # Set the stop node when its length is larger than the max length
        if length >= self.max_length:
            # Set the node to be the stop one
            # Thus change its position to be 'Stop' and set the
            # it un-growable
            self.node_pool[node_id].set_position("Sink")

        # Set the node to be stop when the solution flag is detected
        solution_flag = self.root.thought.solution_flag
        thought = self.node_pool[node_id].thought
        if solution_flag in str(thought) and node_id != self.root.identity:
            self.node_pool[node_id].set_position("Sink")

    def set_node_growth(self, node_id: int):
        """Set the node to be the growable one."""
        # Determine whether to stop the growth of the node as
        # this node has enough children
        if len(list(self.graph.successors(node_id))) >= self.num_next_steps:
            # Set the node to be the stop one
            # Thus change its position to be 'Stop' and set the
            # it un-growable
            self.node_pool[node_id].set_growth("Un-growable")

    def set_node_status(self, node_id: int):
        """Set the node status."""
        self.set_node_stop(node_id)
        self.set_node_growth(node_id)

    def stop_growth(self):
        """Whether to stop the growth of the structure."""
        # Stop when all nodes in the queue are stop nodes
        if self.get_grow_node() is None:
            return True
        return False

    def is_node_growable(self, node_id: int):
        """Check whether the node is growable."""
        return self.node_pool[node_id].growth == "Growable"

    def reset_structure(self):
        """Reset the tee."""
        self.root: BasicNode = None
        self.node_pool: Dict[str, BasicNode] = None
        self.edge_pool: Dict[str, BasicEdge] = None
        self.graph.clear()
        self.node_id_tracker = -1

    def create_save_folder(self, foldername: str = None, location: str = None) -> str:
        """Create the save path for the thought structure."""
        foldername = self.save_foldername if foldername is None else foldername
        location = self.save_path if location is None else location
        path = f"{location}/{foldername}"
        os.makedirs(path, exist_ok=True)
        return path

    def save_thought_path(
        self,
        thought_path: List[BasicNode],
        filename: str = None,
        foldername: str = None,
        location: str = None,
    ):
        """Save the branch of the structure."""
        filename = "thought_chain" if filename is None else filename
        save_path = self.create_save_folder(foldername, location)

        # Save the information of the thought path
        file_path = f"{save_path}/{filename}.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(thought_path, file)

    def save_structure(self, foldername: str = None, location: str = None):
        """Save the structure to the path."""
        save_path = self.create_save_folder(foldername, location)
        # Save the graph structure
        nx.write_gexf(self.graph, f"{save_path}/main_structure.gexf")
        nx.write_gml(self.graph, f"{save_path}/main_structure.gml")
        # Save the information of each node
        for node_id in self.graph.nodes:
            node = self.node_pool[node_id]
            filename = f"node_{node_id}" if node_id != self.root.identity else "root"
            file_path = f"{save_path}/{filename}.json"
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(node, file)

        for edge_id in self.edge_pool:
            edge_data = self.edge_pool[edge_id]
            filename = f"edge_{edge_id}"
            file_path = f"{save_path}/{filename}.json"
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(edge_data, file)
