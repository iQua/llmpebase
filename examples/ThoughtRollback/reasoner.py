"""
A reasoner to perform adaptive reasoning with the thought rollback.
"""

import TR_structure
from llmpebase.reasoner.structured_thought import StructuredThoughtReasoner


class ThoughtRollbackReasoner(StructuredThoughtReasoner):
    """
    A TR reasoner to answer the question by rolling back with the request model.
    """

    def define_structure(self):
        return TR_structure.ThoughtRollbackStructure(
            thought_model=self.thought_model,
            model_config=self.model_config,
            logging_config=self.logging_config,
            visualizer=self.visualizer,
        )
