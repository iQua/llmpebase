"""
Generic components for thought structure.
"""

import logging
from typing import Union, Tuple, List
from dataclasses import dataclass

from llmpebase.model.prompting.base import BasicSamplePrompt

from transformers.utils import ModelOutput as FieldFrozenContainer
from llmpebase.model.prompting.prompt_generic import BasicThoughtPromptFormat


@dataclass
class BasicEvaluation(FieldFrozenContainer):
    """
    A base evaluation to be used to maintain evaluations of one thought.

    By default, we allow one thought corresponds to multiple evaluations.
    """

    evaluation_prompt: BasicThoughtPromptFormat = None

    evaluation_scores: List[float] = None
    evaluation_contents: List[str] = None
    evaluation_outputs: List[str] = None

    def score(self):
        """Get the evaluation score."""
        return max(self.evaluation_scores)

    def content(self):
        """Get the evaluation output."""
        max_score = self.score()
        return self.evaluation_contents[self.evaluation_scores.index(max_score)]

    def output(self):
        """Get the evaluation output."""
        max_score = self.score()
        return self.evaluation_outputs[self.evaluation_scores.index(max_score)]


@dataclass
class BasicSimilarity(FieldFrozenContainer):
    """
    A base similarity to be used to maintain similarity of two thought.

    By default, we allow two thought corresponds to multiple similarities from
    llm.
    """

    similarity_prompt: BasicThoughtPromptFormat = None

    similarity_scores: List[str] = None
    similarity_contents: List[str] = None
    similarity_outputs: List[str] = None

    def score(self):
        """Get the evaluation score."""
        return max(self.similarity_scores)

    def content(self):
        """Get the evaluation output."""
        max_score = self.score()
        return self.similarity_contents[self.similarity_scores.index(max_score)]

    def output(self):
        """Get the evaluation output."""
        max_score = self.score()
        return self.similarity_outputs[self.similarity_scores.index(max_score)]


@dataclass
class BasicReasoning(FieldFrozenContainer):
    """
    A base reasoning to be used to maintain inference of the thought.
    """

    reasoning_prompt: BasicThoughtPromptFormat = None

    reasoning_output: str = None


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
    evaluation_score: float = None
    evaluation_content: str = None
    step_name: str = None

    similar_thoughts: List[str] = None
    similar_thought_scores: List[float] = None
    similar_thought_similarities: List[BasicSimilarity] = None

    def backup_though(
        self, thought: str, thought_score: float, thought_similarity: BasicSimilarity
    ):
        """Add a similar though to the backup."""
        if self.similar_thoughts is None:
            self.similar_thoughts = []
            self.similar_thought_scores = []
            self.similar_thought_similarity = []

        self.similar_thoughts.append(thought)
        self.similar_thought_scores.append(thought_score)
        self.similar_thought_similarities.append(thought_similarity)


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
    reasoning: BasicReasoning = None
    evaluation: BasicEvaluation = None
    edge_score: float = None

    auxiliary: dict = None
