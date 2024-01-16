"""
A reasoner to perform the reasoning step by step in a tree structure.
"""

from llmpebase.model.thought_structure import trees
from llmpebase.reasoner.structured_thought import StructuredThoughtReasoner


class TreeThoughtReasoner(StructuredThoughtReasoner):
    """
    A reasoner to answer the question with a reasoning process built upon
    the tree of thoughts.
    """

    def define_structure(self):
        """Define the thought structure to be used."""
        structure_config = self.model_config["thought_structure"]
        return trees.get(growth_type=structure_config["growth_type"])(
            thought_model=self.thought_model,
            model_config=self.model_config,
            logging_config=self.logging_config,
            visualizer=self.visualizer,
        )
