"""
A reasoner to perform the reasoning step by step in a chain structure.
"""

from torch import nn

from llmpebase.model.thought_structure import chains
from llmpebase.model.prompting.base import BasicPromptSample


class ChainReasoner:
    """A CoT reasoner to answer the question with the request model.

    Args:
        thought_model: A defined thought model used to generate thought with
         the implemented functions `generate_thoughts`.
    """

    def __init__(self, thought_model: nn.Module, model_config: dict = None):
        self.thought_model = thought_model

        self.structure = chains.ChainThoughtStructure(model_config)

    def forward(self, prompt_sample: BasicPromptSample):
        """Forward the reasoning in the chain structure."""
        # Place the task prompt in the root so that all subsequent thought chains
        # include the task prompt
        self.structure.construct_root(thought=str(prompt_sample), thought_score=None)
        # Grow the thought structure
        self.structure.build_structure(thought_model=self.thought_model)
