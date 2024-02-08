"""
A reasoner to perform the reasoning step by step in a graph structure.
"""

from llmpebase.model.thought_structure import graphs
from llmpebase.model.thought_structure.base import BaseThoughtStructure
from llmpebase.reasoner.structured_thought import StructuredThoughtReasoner


class GraphThoughtReasoner(StructuredThoughtReasoner):
    """
    A reasoner to answer the question with a reasoning process built upon
    a graph thought structure.
    """

    def define_structure(self) -> type[BaseThoughtStructure]:
        return graphs.GraphTreeThoughtStructure(
            thought_model=self.thought_model,
            model_config=self.model_config,
            logging_config=self.logging_config,
            visualizer=self.visualizer,
        )
