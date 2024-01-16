"""
Generic components for thought structure.
"""
import logging
from typing import Union, Tuple, List
from dataclasses import dataclass

from llmpebase.model.prompting.base import BasicSamplePrompt

from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class BasicThoughtStep(FieldFrozenContainer):
    """
    A base thought step to be used as the basic component in reasoning chains.
    The thought can be defined as BasicSamplePrompt or a string.
    Generally, when the step is the initial step, the thought is the BasicSamplePrompt
    containing multiple basic information.
    """

    step_idx: int
    thought: Union[str, BasicSamplePrompt] = None
    thought_score: float = None
    step_name: str = None

    similar_thoughts: List[str] = None
    similar_thought_scores: List[float] = None
    similar_thought_similarity: List[float] = None
    thought_similarity_prompt: List[str] = None

    def backup_though(
        self, thought: str, thought_score: float, similarity_score: float, prompt: str
    ):
        """Add a similar though to the backup."""
        if self.similar_thoughts is None:
            self.similar_thoughts = []
            self.similar_thought_scores = []
            self.similar_thought_similarity = []
            self.thought_similarity_prompt = []

        self.similar_thoughts.append(thought)
        self.similar_thought_scores.append(thought_score)
        self.similar_thought_similarity.append(similarity_score)
        self.thought_similarity_prompt.append(prompt)


@dataclass
class BasicNode(BasicThoughtStep):
    """
    A basic node used to build the thought structure. And, each node represents one
    thought step in the reasoning chain.

    When the node is the root node, the thought is the BasicSamplePrompt.
    """

    identity: str = None
    node_name: str = None
    position: str = None
    growth: str = None
    position_types: Tuple[str] = None
    growth_types: Tuple[str] = None

    # The auxiliary information for the node
    # This aims to store any additional information
    # so that there is no need to create a new Node class
    auxiliary: dict = None

    def set_position(self, position: str = "Intermediate"):
        """Set the node position.

        Note that by default, the node is set to tbe Un-growable when
        the position is Sink.
        """

        assert position in self.position_types
        # Only make adjustment when the position
        # is different from the current one.
        if position != self.position:
            self.position = position
            self.node_name = f"{position} Node"
            # By default, Stop is unable to grow.
            self.growth = "Un-growable" if position == "Sink" else "Growable"

            logging.info(
                "Set the node %s to be %s and %s",
                self.identity,
                self.position,
                self.growth,
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

    src_node_id: str = None
    dst_node_id: str = None
    edge_type: str = None
    reasoning_prompt: str = None
    evaluation_prompt: str = None
    edge_score: float = None

    auxiliary: dict = None
