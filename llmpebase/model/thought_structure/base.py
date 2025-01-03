"""
An implementation of the base thought step and base thought structure.
Based on existing literature, the thought is defined as a coherent language 
sequence that serves as an intermediate step toward problem solving. Thus,
we call it 'thoughtstep' to show that it is a thought that serves as a step
in the reasoning chain where a chain is presented as a path of the structure.
"""

import os
import json
import glob
import logging
from typing import List, Union, Dict

import networkx as nx

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer
from llmpebase.model.thought_structure.structure_generic import (
    BasicNode,
    BasicEdge,
    BasicEvaluation,
    BasicSimilarity,
    BasicPromptAndResponse,
)
from llmpebase.model.thought_structure.thought_model import LlmThoughtModel


class BaseStructure:
    """
    A base structure to present the graph and related information.
    """

    def __init__(
        self, logging_config: dict, visualizer: BasicStructureVisualizer = None
    ):
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

        # The visualizer to visualize the thought structure
        self.visualizer = visualizer

        self.logging_config = logging_config
        self.save_path = logging_config["result_path"]
        self.base_save_foldername = "thought_structure"
        self.save_foldername = "thought_structure"

    def generate_edge_id(self, src_node_id: str, dst_node_id: str):
        """Generate an edge id."""
        return f"{src_node_id}->{dst_node_id}"

    def create_node(
        self,
        *args,
    ):
        """Create a new node."""
        raise NotImplementedError

    def create_edge(self, *args):
        """Create a new edge."""
        raise NotImplementedError

    def is_duplicated_path(
        self,
        node1_id: str,
        node2_id: str,
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

    def get_node_path(
        self, src_node_id: str, dst_node_id: str = None
    ) -> List[BasicNode]:
        """Get the path containing nodes and edges between two nodes."""

        node_ids = nx.shortest_path(self.graph, src_node_id, dst_node_id)
        return [self.node_pool[node_id] for node_id in node_ids]

    def get_successor_nodes(self, node_id: str) -> List[BasicNode]:
        """Get the successor nodes of the given node."""
        return [self.node_pool[node_id] for node_id in self.graph.successors(node_id)]

    def get_path_edges(self, path: List[BasicNode]) -> List[BasicEdge]:
        """Get the edges in the path."""
        return [
            self.edge_pool[
                self.generate_edge_id(path[idx].identity, path[idx + 1].identity)
            ]
            for idx in range(len(path) - 1)
        ]

    def get_sink_nodes(self) -> List[BasicNode]:
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

    def set_node_sink(self, node_id: str, max_length: int = 3):
        """Set the node to be the stop node."""
        length = len(self.get_node_path(self.root.identity, node_id))
        # Set the sink node when its length is larger than the max length
        if length >= max_length:
            # Set the node to be the stop sink
            # Thus change its position to be 'Stop' and set the
            # it un-growable (default setup)
            self.node_pool[node_id].set_position("Sink")

        # Set the node to be stop when the solution flag is detected
        solution_flag = self.root.thought.solution_flag
        thought = self.node_pool[node_id].thought
        if (
            solution_flag.lower() in str(thought).lower()
            and node_id != self.root.identity
        ):
            self.node_pool[node_id].set_position("Sink")

    def set_node_growth(self, node_id: str, num_next_nodes: int = 1):
        """Set the node to be the growable one."""
        # Close growth of the node as this node has enough children
        # Here we use the successors of the node as the flag to determine
        # A better way is to use the edges, i.e., self.graph.edges(node_id)
        if len(list(self.graph.successors(node_id))) >= num_next_nodes:
            # Set the node to be un-growable
            self.node_pool[node_id].set_growth("Un-growable")

    def is_node_growable(self, node_id: str):
        """Check whether the node is growable."""
        return self.node_pool[node_id].growth == "Growable"

    def is_node_sink(self, node_id: str):
        """Check whether the node is growable."""
        return self.node_pool[node_id].position == "Sink"

    def reset_structure(self):
        """Reset the tee."""
        self.root: BasicNode = None
        self.node_pool: Dict[str, BasicNode] = None
        self.edge_pool: Dict[str, BasicEdge] = None
        self.graph.clear()

    def set_save_foldername(self, foldername: str):
        """Set the foldername for the visualization."""
        self.save_foldername = foldername

    def create_save_folder(self, foldername: str = None, location: str = None) -> str:
        """Create the save path for the thought structure."""
        foldername = self.save_foldername if foldername is None else foldername
        location = self.save_path if location is None else location
        path = f"{location}/{foldername}"
        os.makedirs(path, exist_ok=True)
        return path

    def save_node_path(
        self,
        node_path: List[BasicNode],
        filename: str = None,
        foldername: str = None,
        location: str = None,
    ):
        """Save the given path of the structure."""
        filename = "thought_chain" if filename is None else filename
        save_path = self.create_save_folder(foldername, location)

        # Save the information of the thought path
        file_path = f"{save_path}/{filename}.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(node_path, file)

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

    def load_structure(self, location: str = None):
        """Load the structure data from files."""
        # Load the core graph structure as the networkx
        self.graph = nx.read_gml(f"{location}/main_structure.gml")
        # Load the nodes
        self.node_pool: Dict[str, BasicNode] = {}
        # First load the root
        root_data = json.load(open(f"{location}/root.json", "r", encoding="utf-8"))
        self.root = self.create_node(**root_data)
        self.node_pool[root_data["identity"]] = self.root
        node_files = glob.glob(f"{location}/node_*.json")
        for file_path in node_files:
            node = json.load(open(file_path, "r", encoding="utf-8"))
            self.node_pool[node["identity"]] = self.create_node(**node)

        # Load the edges
        self.edge_pool: Dict[str, BasicEdge] = {}
        edge_files = glob.glob(f"{location}/edge_*.json")
        for file_path in edge_files:
            edge = json.load(open(file_path, "r", encoding="utf-8"))
            self.edge_pool[edge["edge_id"]] = self.create_edge(**edge)


class BaseThoughtStructure(BaseStructure):
    """
    A base thought structure performing as the fundamental framework
    in which each node is a thought step.
    """

    def __init__(
        self,
        thought_model: LlmThoughtModel,
        model_config: dict,
        logging_config: dict,
        visualizer: BasicStructureVisualizer = None,
    ):
        super().__init__(logging_config, visualizer)

        # The thought model is the necessary part for the thought
        # structure building
        self.thought_model = thought_model
        assert hasattr(self.thought_model, "generate_thoughts")
        assert hasattr(self.thought_model, "evaluate_thoughts")
        assert hasattr(self.thought_model, "measure_thought_similarity")

        # The growth type of the thought structure
        # The default growth type is the chain-generic
        # where chain means the structure is grown in a chain manner
        # while the dfg means the structure is grown in Depth-First Growth
        self.growth_type = "chain"

        # Tracker of the node id starting from 0
        # thus, root of the thought structure should be 0
        self.node_id_tracker = -1

        self.position_states = ("Root", "Intermediate", "Sink")
        self.growth_states = ("Growable", "Un-growable")

        # Get the configuration
        config = model_config["thought_structure"]

        # Number of next reasoning steps
        self.num_next_steps = (
            1 if "num_next_steps" not in config else config["num_next_steps"]
        )
        # Max #length of the reasoning chain, i.e., path the structure
        self.max_length = 3 if "max_length" not in config else config["max_length"]
        # Max #solution existed in the thought structure
        self.max_stops = 5 if "max_stops" not in config else config["max_stops"]

        # Threshold setup between thoughts
        # The min similarity between two thoughts that
        # they are regarded as identical
        # this is generally the value generated by
        # LLMs.
        self.min_thought_sim = (
            0.9
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

    def create_node(
        self,
        identity: str,
        step_idx: int,
        thought: Union[str, BasicSamplePrompt],
        evaluation: BasicEvaluation = None,
        step_name: str = "Reasoning Step",
        node_name: str = "Intermediate Node",
        position: str = "Intermediate",
        growth: bool = "Growable",
        auxiliary: dict = None,
    ):
        """Create a node."""
        # pylint: disable=arguments-differ
        assert isinstance(identity, str)

        return BasicNode(
            identity=identity,
            step_idx=step_idx,
            thought=thought,
            evaluation_content=(
                evaluation.content() if evaluation is not None else None
            ),
            evaluation_score=(evaluation.score() if evaluation is not None else None),
            step_name=step_name,
            node_name=node_name,
            position=position,
            growth=growth,
            position_states=self.position_states,
            growth_states=self.growth_states,
            auxiliary=auxiliary,
        )

    def create_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        edge_type="Reasoning",
        reasoning=BasicPromptAndResponse,
        evaluation=BasicEvaluation,
        edge_score: float = 1.0,
        edge_id=None,
        auxiliary: dict = None,
    ):
        """Create an edge."""
        # pylint: disable=arguments-differ
        assert isinstance(src_node_id, str) and isinstance(dst_node_id, str)

        return BasicEdge(
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            reasoning=reasoning,
            evaluation=evaluation,
            edge_type=edge_type,
            edge_score=edge_score,
            edge_id=edge_id,
            auxiliary=auxiliary,
        )

    def construct_root(
        self,
        thought: BasicSamplePrompt = None,
        evaluation: BasicEvaluation = None,
        **kwargs,
    ):
        """
        Set the root of the structure.

        Only by creating the structure's root node can it be grown until it is built.
        Generally, the task prompt and the request containing the question and
        """

        identity = self.generate_node_id()

        self.root = self.create_node(
            identity=identity,
            step_idx=0,
            thought=thought,
            evaluation=evaluation,
            step_name="Initial Step",
            node_name="Root Node",
            position="Root",
            growth="Growable",
            auxiliary={},
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

        # Avoid the duplication of the node id
        if self.node_pool is not None and str(new_id) in self.node_pool:
            node_ids = list(self.node_pool.keys())
            max_node_id = max([int(node_id) for node_id in node_ids])
            self.node_id_tracker = max_node_id
            new_id = self.node_id_tracker + 1

        self.node_id_tracker += 1
        new_id = str(new_id)
        return new_id

    def compute_thought_similarity(
        self,
        grow_node: BasicNode,
        thoughts: List[str],
    ) -> Union[List[None], List[Dict[str, BasicSimilarity]]]:
        """
        Measure the similarity between the given thoughts and thoughts in the structure.

        :param grow_node: The node the thoughts should be added to.
        :param thoughts: The thoughts to be added.
        """
        similarities = [None] * len(thoughts)
        # Once the min_thought_sim is set to None, we do not need to
        # compute them
        if self.min_thought_sim is None:
            return similarities
        similarities = [{}] * len(thoughts)
        grow_path = self.get_node_path(
            src_node_id=self.root.identity, dst_node_id=grow_node.identity
        )
        for idx, thought in enumerate(thoughts):
            for node_id in self.graph.nodes:
                node = self.root
                if node_id != self.root.identity:
                    node = self.node_pool[node_id]
                    similarity = self.thought_model.measure_thought_similarity(
                        thought,
                        node.thought,
                        thought_chain=grow_path,
                    )
                    similarities[idx][node.identity] = similarity

        return similarities

    def search_identical_thought(
        self,
        thought: str,
        prev_node_id: str,
        thought_evaluation: BasicEvaluation,
        thought_similarities: Dict[str, BasicSimilarity] = None,
    ) -> List[str]:
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
            similarity = (
                thought_similarities[node.identity]
                if node.identity in thought_similarities
                else 0.0
            )
            # Get the difference between the evaluation score of the visited node
            # and the input thought score
            thought_score = thought_evaluation.score()
            evaluation_diff = abs(node.evaluation_score - thought_score)

            if (
                thought != node_thought
                and similarity.score() >= self.min_thought_sim
                and evaluation_diff <= self.max_score_diff
                and self.is_duplicated_path(node.identity, prev_node_id)
            ):
                identical_nodes.append(node.identity)

        return identical_nodes

    def add_node(
        self,
        thought: str,
        prev_node_id: str,
        thought_evaluation: BasicEvaluation,
        thought_inference: BasicPromptAndResponse,
        edge_weight: float = 1.0,
        **kwargs,
    ) -> int:
        """Adding one node to the thought structure."""

        assert isinstance(prev_node_id, str)

        node_id = self.generate_node_id()
        edge_id = self.generate_edge_id(prev_node_id, node_id)

        if "step_idx" in kwargs:
            step_idx = kwargs["step_idx"]
        else:
            step_idx = self.node_pool[prev_node_id].step_idx + 1
        # Create the node
        new_node = self.create_node(
            identity=node_id,
            step_idx=step_idx,
            thought=thought,
            evaluation=thought_evaluation,
            node_name=(
                f"Intermediate Node {node_id}"
                if "node_name" not in kwargs
                else kwargs["node_name"]
            ),
            step_name=(
                f"Reasoning step {step_idx}"
                if "step_name" not in kwargs
                else kwargs["step_name"]
            ),
            position="Intermediate" if "position" not in kwargs else kwargs["position"],
            growth="Growable" if "growth" not in kwargs else kwargs["growth"],
            auxiliary={},
        )
        # Create a edge create_edge
        new_edge = self.create_edge(
            edge_id=edge_id,
            edge_type="Reasoning" if "edge_type" not in kwargs else kwargs["edge_type"],
            src_node_id=prev_node_id,
            dst_node_id=node_id,
            reasoning=thought_inference,
            evaluation=thought_evaluation,
            edge_score=edge_weight,
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
            weight=edge_weight,
            edge_id=edge_id,
            edge_type="Reasoning" if "edge_type" not in kwargs else kwargs["edge_type"],
        )

        # Set the node status
        self.set_node_status(new_node.identity)

        logging.info(
            "  Created new %s node (%s) (%s) %s grown from the node %s, thus edge %s (%s)",
            self.node_pool[node_id].position,
            self.node_pool[node_id].node_name,
            self.node_pool[node_id].growth,
            node_id,
            prev_node_id,
            edge_id,
            self.edge_pool[edge_id].edge_type,
        )

        return node_id

    def extend_node(
        self,
        thought: str,
        thought_score: float,
        node_ids: List[str],
        thought_similarities: Dict[str, BasicSimilarity],
        **kwargs,
    ):
        """Extend the node by adding similar thoughts to it."""
        node_id = node_ids[0]

        self.node_pool[node_id].backup_though(
            thought, thought_score, thought_similarity=thought_similarities[node_id]
        )

        logging.info(
            "  Backed it up to an existing node %s",
            node_id,
        )
        return node_id

    def add_thought(
        self,
        thought: str,
        thought_evaluation: BasicEvaluation,
        thought_inference: BasicPromptAndResponse,
        prev_node_id: str,
        to_node_ids: List[str],
        thought_similarities: Dict[str, BasicSimilarity],
        **kwargs,
    ):
        """Add the thought to the structure."""
        assert isinstance(prev_node_id, str)
        assert all([isinstance(node_id, str) for node_id in to_node_ids])

        logging.info("Adding the thought derived from the node %s", prev_node_id)
        # If there is no identical thought, add the thought to the structure
        # as a new node
        if len(to_node_ids) == 0:
            node_id = self.add_node(
                thought, prev_node_id, thought_evaluation, thought_inference, **kwargs
            )
        else:
            thought_score = thought_evaluation.score()
            node_id = self.extend_node(
                thought,
                thought_score,
                node_ids=to_node_ids,
                thought_similarities=thought_similarities,
                prev_node_id=prev_node_id,
            )

        return node_id

    def grow_structure(
        self,
        prev_node_id: str,
        thoughts: List[str],
        thought_evaluations: List[BasicEvaluation],
        thought_inferences: List[BasicPromptAndResponse],
        thought_all_similarities: List[Dict[str, BasicSimilarity]],
        **kwargs,
    ):
        """Grow the structure by adding new thoughts.

        :param prev_node_id: The node id, which produces the thoughts as the next
         steps.
        :param thoughts: The thoughts to be added.
        :param thought_evaluations: The evaluations of thoughts.
        :param thought_similarities: The similarity scores in which each item is a list
         containing the similarity between the corresponding thought and all existing
         thoughts in nodes of the structure.
        """

        for idx, (thought, evaluation, inference, similarities) in enumerate(
            zip(
                thoughts,
                thought_evaluations,
                thought_inferences,
                thought_all_similarities,
            )
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
            if (
                not self.root.thought.solution_flag in str(thought)
                and similarities is not None
            ):
                similar_node_ids = self.search_identical_thought(
                    thought,
                    prev_node_id=prev_node_id,
                    thought_evaluation=evaluation,
                    thought_similarities=similarities,
                )
            # Add the thought to the structure either as a new node
            # or being added to 'to_nodes' based on the similarity
            # measurement
            self.add_thought(
                thought=thought,
                thought_evaluation=evaluation,
                thought_inference=inference,
                prev_node_id=prev_node_id,
                to_node_ids=similar_node_ids,
                thought_similarities=similarities,
                **kwargs,
            )

            # Change the status of the previous node
            # as one of its children has been grown
            self.set_node_status(prev_node_id)

        # Set the status of nodes in the graph
        for node_id in self.graph.nodes:
            self.set_node_status(node_id)

        # Save the structure after each growth
        self.save_structure()
        self.save_state()

    def generate_next_thoughts(self, thought_path: List[BasicNode]):
        """Generate the next thoughts for the node_id."""
        # Generate and then evaluate the next thoughts
        return self.thought_model.generate_thoughts(
            thought_chain=thought_path, num_thoughts=self.num_next_steps
        )

    def evaluate_thoughts(
        self, thought_path: List[BasicNode], thoughts: List[str]
    ) -> List[BasicEvaluation]:
        """Evaluate the thoughts for the node_id."""
        evaluations = [None] * len(thoughts)
        # Whether the evaluate of thoughts needs to be compared
        if self.max_score_diff is not None:
            evaluations = self.thought_model.evaluate_thoughts(
                thoughts, thought_chain=thought_path
            )

        return evaluations

    def build_structure(
        self,
        **kwargs,
    ):
        """Build the thought structure by adding thoughts till terminal."""
        # Set the growth index
        growth_idx = 1

        while True:
            # Record the current node pool
            num_nodes = len(self.node_pool)
            # Get the node to be grown
            grow_node = self.get_grow_node()
            if grow_node is None:
                break
            # Get the thought path of the node to be grown
            thought_path = self.get_node_path(self.root.identity, grow_node.identity)

            # Generate the next thoughts
            thoughts, thought_inferences = self.generate_next_thoughts(thought_path)
            # Evaluate the thoughts
            evaluations = self.evaluate_thoughts(thought_path, thoughts)
            # Measure the similarity between new thoughts and existing thoughts in the
            # structure
            all_similarities = self.compute_thought_similarity(grow_node, thoughts)

            # Grow the structure by adding the thoughts
            self.grow_structure(
                prev_node_id=grow_node.identity,
                thoughts=thoughts,
                thought_evaluations=evaluations,
                thought_inferences=thought_inferences,
                thought_all_similarities=all_similarities,
            )

            # Draw the graph and save to the disk
            # as each node added to the graph means a new step
            if num_nodes < len(self.node_pool):
                if self.visualizer is not None:
                    self.visualizer.visualize(
                        self.graph,
                        self.node_pool,
                        save_name=f"Growth_{growth_idx}__Step_{grow_node.step_idx + 1}",
                    )
                growth_idx += 1

        # Draw the whole graph after building
        if self.visualizer is not None:
            self.visualizer.visualize(
                self.graph, self.node_pool, save_name="built_structure"
            )

    def get_path_scores(self, path: List[BasicNode]) -> List[float]:
        """Get the scores of nodes in the path."""
        # Skip the thought score of the root node as it will be None
        return [node.evaluation_score for node in path[1:]]

    def get_grow_node(self) -> BasicNode:
        """Get which node to be grown."""
        node = None

        # When the current graph has enough solutions, stop growing
        if self.early_stop():
            return node

        for node_id in self.graph.nodes:
            if self.node_pool[node_id].growth == "Growable":
                node = self.node_pool[node_id]
                break
        return node

    def set_node_status(self, node_id: str):
        """Set the node status."""
        self.set_node_sink(node_id, max_length=self.max_length)
        self.set_node_growth(node_id, num_next_nodes=self.num_next_steps)

    def early_stop(self):
        """Stop the growth of the structure."""
        # Stop the growth when the structure has enough solutions
        if len(self.get_sink_nodes()) >= self.max_stops:
            return True
        return False

    def reset_structure(self):
        """Reset the tee."""
        super().reset_structure()
        self.node_id_tracker = -1

    def save_state(self, foldername: str = None, location: str = None):
        """Save the state of the structure."""
        save_path = self.create_save_folder(foldername, location)

        # Get how many nodes
        n_nodes = len(self.graph.nodes)
        # Get how many paths
        n_paths = len(self.get_sink_nodes())
        # Get the config of the llm used
        llm_config = self.thought_model.llm_model.generation_config

        filename = "structure_state"
        file_path = f"{save_path}/{filename}.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "growth_type": self.growth_type,
                    "n_nodes": n_nodes,
                    "n_paths": n_paths,
                    "llm_config": llm_config,
                },
                file,
            )

        return file_path
