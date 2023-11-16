"""
Implementation of Basic Tree Thought Structures.

Note that the root of the tree will not be counted as one level 
or height or depth.
"""

import os
from queue import PriorityQueue, Queue
from typing import Dict

import logging

from anytree import NodeMixin, RenderTree
from anytree.search import findall as anytree_findall
from anytree.exporter import JsonExporter, DotExporter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ThoughtNode(NodeMixin):
    """A base node for the tree structure."""

    def __init__(
        self,
        name: str,
        thought: str,
        thought_score: float,
        parent=None,
        children=None,
        edge_weight=None,
    ):
        super(ThoughtNode, self).__init__()

        # unique name of the node
        self.name = name
        # thought: a coherent language sequence that serves as
        # an intermediate step toward problem solving
        self.thought = thought
        # the evaluation score of this thought
        self.thought_score = thought_score

        # backup
        self.backup_thoughts = []
        self.backup_thoughts_scores = []

        self.parent = parent
        # Add weight to edges and make use of them.
        # See https://anytree.readthedocs.io/en/latest/tricks/weightededges.html
        # for details.
        self.edge_weight = edge_weight if parent is not None else None

        if children:
            self.children = children

        # whether this node will be set as the leaf node
        # and no child will be added
        self.is_leaf_terminal = False

    def backup_similar_though(self, thought: str, thought_score: int):
        """Inserting a similar though to the backup."""
        self.backup_thoughts.append(thought)
        self.backup_thoughts_scores.append(thought_score)

    def remove_children(self):
        """Removing the children of one node."""
        self.children = []

    def open_leaf_terminal(self):
        """Opening the flag to denote that no child will be added to this node."""
        self.is_leaf_terminal = True


class ResidualThoughtNode(ThoughtNode):
    """A base node for the tree structure."""

    def __init__(
        self,
        name: str,
        thought: str,
        thought_score: float,
        parent=None,
        children=None,
        edge_weight=None,
        residual_thought=None,
    ):
        super(ResidualThoughtNode, self).__init__(
            name, thought, thought_score, parent, children, edge_weight
        )
        # the though that showing the residual obtained
        # from the previous reasoning
        self.residual_thought = residual_thought


class ResidualThoughtTree:
    """A base class for the thought tree with residual (RTT)."""

    def __init__(
        self,
        model,
        n_child_nodes: int = 2,
        model_config: dict = None,
    ):
        # the model used to build the tree
        self.model = model

        # tree structure
        # after using rendertree, each node will contain
        # the full chain from root to this node,
        # See document of `anytree` for details
        self.root: ResidualThoughtNode = None

        # collection of nodes with
        # id: ThoughtNode()
        self.nodes = {}
        self.node_id_tracker = -1

        self.n_child_nodes = n_child_nodes

        self.tree_config = model_config["tree_settings"]

        self.set_thresholds(self.tree_config)

        self.best_thought = None
        self.best_value = float("-inf")

        self.history = []  # added line initalize history

    def set_thresholds(self, tree_config):
        """Setting the threshold of the tree."""

        self.min_thoughts_similarity: float = (
            0.8
            if "min_thoughts_similarity" not in tree_config
            else tree_config["min_thoughts_similarity"]
        )
        self.max_thoughts_score_diff: float = (
            0.1
            if "max_thoughts_score_diff" not in tree_config
            else tree_config["max_thoughts_score_diff"]
        )

        self.max_depth: int = (
            3 if "max_depth" not in tree_config else tree_config["max_depth"]
        )
        # max_leaves <= 2**max_depth - 1
        self.max_leaves: int = (
            2**self.max_depth - 1
            if "max_leaves" not in tree_config
            else tree_config["max_leaves"]
        )

        # if you use the leaf-wise first, max_steps should be set
        self.max_steps: int = (
            2**self.max_depth
            if "max_steps" not in tree_config
            else tree_config["max_steps"]
        )

        self.min_leaf_thought_score: float = (
            0.3
            if "min_leaf_thought_score" not in tree_config
            else tree_config["min_leaf_thought_score"]
        )

        self.max_leaf_thought_score: float = (
            0.8
            if "max_leaf_thought_score" not in tree_config
            else tree_config["max_leaf_thought_score"]
        )

    def construct_tree_root(
        self,
        thought: str = None,
        residual_thought: str = None,
        thought_score: float = None,
    ):
        """Building the root node and prompt of the tree."""
        # the root prompt of this tree should be the thought of the root node
        identity_int = self.assign_node_id()

        self.root = ResidualThoughtNode(
            name=str(identity_int),
            thought=thought,
            thought_score=thought_score,
            parent=None,
            children=None,
            edge_weight=None,
            residual_thought=residual_thought,
        )
        self.nodes[str(identity_int)] = self.root

    def assign_node_id(self):
        """Assigning id to each node."""
        new_id = self.node_id_tracker + 1
        self.node_id_tracker += 1
        return new_id

    def is_duplicated_though(
        self,
        node: ThoughtNode,
        thought: str,
        thought_score: float,
        parent_node: ThoughtNode,
    ):
        """Whether the new thought is similar to the one in
        existing node.

        1. the thought similarity between this thought and node's thought
        is high.
        2. the scores difference between this thought and node's thought
        is small.
        3. the thought is generated from the same chain in exisitng node.
        """
        # skip root node
        if node.thought_score is None:
            return False

        similarity_score = self.model.measure_similarity(node.thought, thought)
        score_diff = abs(node.thought_score - thought_score)

        return (
            similarity_score >= self.min_thoughts_similarity
            and score_diff <= self.max_thoughts_score_diff
            and self.is_duplicated_thought_chain(node.parent, parent_node)
        )

    def is_duplicated_thought_chain(
        self,
        node_src: ThoughtNode,
        node_tgt: ThoughtNode,
    ):
        """Whether this though derives from a existing chain."""
        return node_src.path == node_tgt.path

    def is_trigger_leaf_node(self, node: ThoughtNode):
        """Whether to trigger the node to become a leaf node.

        Generally, we set a large 'max_leaf_thought_score: 1.0' as the generation model
        is very confidence about its evaluation score on the thought.

        TODO One prehaps utilize a generation model that is different from the one used
        for thought generation to perform the evaluation.
        """

        return (
            node.depth >= self.max_depth
            or len(self.root.leaves) >= self.max_leaves
            or node.thought_score <= self.min_leaf_thought_score
            or node.thought_score >= self.max_leaf_thought_score
        )

    def make_node_leaf(self, node: ThoughtNode = None, node_id: str = None):
        """Making a node to be leaf node."""
        if node is not None:
            node.remove_children()
            node.open_leaf_terminal()
        if node_id is not None:
            self.nodes[node_id].remove_children()
            self.nodes[node_id].open_leaf_terminal()

    def add_node(self, parent_node: ThoughtNode, thought: str, evaluation: float):
        """Adding one node to the tree."""
        assert isinstance(thought, str)

        new_node = None
        # find which node contain the same thought and has same parent
        searched_nodes = anytree_findall(
            self.root,
            filter_=lambda node: self.is_duplicated_though(
                node, thought, evaluation, parent_node
            ),
        )

        if not searched_nodes:
            identity_number = self.assign_node_id()
            new_node = ThoughtNode(
                name=str(identity_number),
                thought=thought,
                thought_score=evaluation,
                parent=parent_node,
            )
            self.nodes[str(identity_number)] = new_node
            if self.is_trigger_leaf_node(new_node):
                self.make_node_leaf(new_node)
                logging.info(
                    "Made the added node with id %s to be leaf node.",
                    str(identity_number),
                )
        else:
            # Backup though for the existing node
            matched_node = searched_nodes[0]
            matched_node.backup_similar_though(thought, evaluation)
            logging.info(
                "Extended the thought for an existing node %s",
                matched_node.name,
            )

        return new_node

    def get_reasoning_chain(self, node: ThoughtNode = None, node_id: str = None):
        """Organizing the thoughts towards target node."""
        node_path = node.path if node is not None else self.nodes[node_id].path
        return [i_node for i_node in node_path]

    def get_best_thought_chain(self):
        """Getting the best though chain in the tree."""
        best_value = 0
        best_leaf_node = None
        for _, node in self.nodes.items():
            if node.is_leaf and node.thought_score > best_value:
                best_leaf_node = node
                best_value = node.thought_score
        return self.get_reasoning_chain(node=best_leaf_node), best_value

    def save_tree_to_json(self, file_name, save_dir):
        """Save the tree to json file."""
        exporter = JsonExporter(indent=2, sort_keys=True)
        file_path = os.path.join(save_dir, file_name)

        os.makedirs(save_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as json_file:
            exporter.write(self.root, json_file)

    def save_tree_to_picture(self, file_name, save_dir):
        """Saving the tree structure to"""
        file_path = os.path.join(save_dir, file_name)

        os.makedirs(save_dir, exist_ok=True)
        DotExporter(self.root).to_picture(file_path)

    def print_tree_structure(self):
        """Showing the structure of the tree."""
        for pre, fill, node in RenderTree(self.root):
            treestr = f"{pre}{node.name}"
            node_content = f"{node.thought}. Evaluation Score: {node.thought_score}"
            print(treestr.ljust(8), node_content)

    def perform_thought_reasoning(self, node: ThoughtNode, **wargs: dict):
        """Performing the generation of thoughts with their evaluation scores."""

        reasoning_chain = self.get_reasoning_chain(node)

        new_thoughts = self.model.generate_thoughts(
            reasoning_chain,
            num_thoughts=self.n_child_nodes,
        )
        evaluated_thoughts = {
            thought: self.model.evaluate_reasoning_thought(reasoning_chain, thought)
            for thought in new_thoughts
        }

        return evaluated_thoughts

    def perform_tree_growth(
        self, node: ThoughtNode, evaluated_thoughts: Dict[str, float], **kwargs: dict
    ):
        """Performing one step of tree growth. For BFS, one step means one level
        of growth."""
        created_nodes = []

        for thought, value in evaluated_thoughts.items():
            added_node = self.add_node(node, thought, evaluation=value)
            if added_node is not None:
                logging.info("Added the node with id %s", added_node.name)

                created_nodes.append(added_node)

        return created_nodes

    def build_thought_tree(self):
        """Building the tree."""


class RTTLevelWise(ResidualThoughtTree):
    """RTT with BFS (Breadth First Search or Level Order Traversal or level-first or level-wise).
    This the level-wise growth strategy used by XGBoost."""

    def build_thought_tree(self):
        """Building the tree with the level-first/Level-wise growth strategy."""

        visited_nodes_id = set()
        node_queue = Queue(maxsize=self.max_leaves)

        node_queue.put(self.root)
        while not node_queue.empty():
            node = node_queue.get()
            if node.name in visited_nodes_id:
                continue

            visited_nodes_id.add(node.name)

            evaluated_thoughts = self.perform_thought_reasoning(node)

            created_nodes = self.perform_tree_growth(node, evaluated_thoughts)

            # judge whether these nodes are leaves and only
            # grow on the normal nodes
            for node in created_nodes:
                if not node.is_leaf_terminal:
                    node_queue.put(node)


class RTTLevelWiseBest(RTTLevelWise):
    """RTT with BFS but extending the `RTTLevelWise` to be the best
    node first - in each level, the node with highest thought score will be
    growth at first."""

    def build_thought_tree(self):
        """Building the tree with the level-first/Level-wise growth strategy."""

        visited_nodes_id = set()
        node_queue = PriorityQueue(maxsize=self.max_leaves)

        node_queue.put((0, int(self.root.name), self.root))

        while not node_queue.empty():
            left_score, _, node = node_queue.get()
            thought_score = 1 - left_score

            if node.name in visited_nodes_id:
                continue

            visited_nodes_id.add(node.name)

            evaluated_thoughts = self.perform_thought_reasoning(node)

            created_nodes = self.perform_tree_growth(node, evaluated_thoughts)

            # judge whether these nodes are leaves and only
            # grow on the normal nodes
            # note that we add the  int(node.name) as the second term of
            # the queue because once the thought_score is the same for two nodes
            # we utilize their unique identity for comparsion
            for node in created_nodes:
                if not node.is_leaf_terminal:
                    # 1 - `thought_score` to make higher thought score the first
                    left_score = 1 - node.thought_score
                    node_queue.put((left_score, int(node.name), node))


class RTTLeafWise(ResidualThoughtTree):
    """RTT with DFS (Deep First Search or best-first or leaf-wise).
    This the leaf-wise growth strategy used by XGBoost."""

    def build_thought_tree(self):
        """Building the tree with the depth-first/leaf-wise growth strategy."""

        visited_nodes_id = set()
        node_queue = Queue(maxsize=self.max_leaves)

        node_queue.put(self.root)

        while not node_queue.empty():
            node = node_queue.get()
            if node.name in visited_nodes_id:
                continue

            visited_nodes_id.add(node.name)

            evaluated_thoughts = self.perform_thought_reasoning(node)

            created_nodes = self.perform_tree_growth(node, evaluated_thoughts)

            # only the node with better score will be used to
            # growth the tree
            better_node = max(created_nodes, key=lambda node: node.thought_score)
            if not better_node.is_leaf_terminal:
                node_queue.put(better_node)
