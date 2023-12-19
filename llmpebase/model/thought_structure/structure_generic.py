"""
Generic components for thought structure.
"""
import logging
from typing import Union, Tuple, List
from dataclasses import dataclass

from llmpebase.model.prompting.base import BasicPromptSample

from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class BasicThoughtStep(FieldFrozenContainer):
    """
    A base thought step to be used as the basic component in reasoning chains.
    The thought can be defined as BasicPromptSample or a string.
    Generally, when the step is the initial step, the thought is the BasicPromptSample
    containing multiple basic information.
    """

    step_idx: int
    thought: Union[str, BasicPromptSample] = None
    thought_score: float = None
    step_name: str = None

    backup_thoughts: List[str] = None
    backup_thought_scores: List[float] = None
    backup_thought_similarity: List[float] = None

    def backup_though(
        self, thought: str, thought_score: float, similarity_score: float
    ):
        """Add a similar though to the backup."""
        if self.backup_thoughts is None:
            self.backup_thoughts = []
            self.backup_thought_scores = []
            self.backup_thought_similarity = []

        self.backup_thoughts.append(thought)
        self.backup_thought_scores.append(thought_score)
        self.backup_thought_similarity.append(similarity_score)


@dataclass
class BasicNode(BasicThoughtStep):
    """
    A basic node used to build the thought structure. And, each node represents one
    thought step in the reasoning chain.

    When the node is the root node, the thought is the BasicPromptSample.
    """

    identity: int = None
    node_name: str = None
    position: str = None
    growth: str = None
    position_types: Tuple[str] = None
    growth_types: Tuple[str] = None

    def set_position(self, position: str = "Intermediate"):
        """Set the node position."""
        assert position in self.position_types
        self.position = position
        self.node_name = f"{position} Node"
        # By default, Stop is unable to grow.
        self.growth = "Un-growable" if position == "Stop" else "Growable"
        logging.info(
            "Set the node %s to be %s and %s", self.identity, self.position, self.growth
        )

    def set_growth(self, status: str = "Growable"):
        """Set the node growth status."""
        if status != self.growth:
            logging.info("Set the node %s to be %s ", self.identity, status)
        self.growth = status


@dataclass
class BasicEdge(FieldFrozenContainer):
    """
    A basic edge used to present the information contained the edge of two
    adjacent nodes.
    """

    edge_id: str

    src_node_id: int = None
    dst_node_id: int = None
    reasoning_prompt: str = None
    evaluation_prompt: str = None
    edge_score: float = None
