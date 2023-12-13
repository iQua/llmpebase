"""
A reasoner to perform the reasoning step by step in a chain structure.
"""

from llmpebase.model import define_model
from llmpebase.model.LM.base import BaseLlmRequest
from llmpebase.model.thought_structure import chains


class ChainReasoner:
    """A CoT reasoner to answer the question with the request model.

    Args:
        thought_model: A defined thought model used to generate thought with
         the implemented functions `generate_thoughts`.
    """

    next_thought_prompt: str = "Next reasoning step:"

    def __init__(self, thought_model: BaseLlmRequest = None, model_config: dict = None):
        # Check if the 'generate_thoughts' is provided
        assert hasattr(thought_model, "generate_thoughts")

        # Define the model once no thought_model is provided
        if thought_model is None:
            thought_model = define_model(model_config)

        self.thought_model = thought_model

        self.structure = chains.ChainThoughtStructure(model_config)

    def forward(self, task_prompt: str, request_prompt: str):
        """Forward the reasoning in the chain structure."""
        # Place the task prompt in the root so that all subsequent thought chains
        # include the task prompt
        self.structure.construct_root(thought=task_prompt, thought_score=None)
        # Grow the thought structure
        while not self.structure.stop_growth():
            # Get the node to be grown
            grow_node = self.structure.get_grow_node()
            # Get the thought path of the node to be grown
            thought_chain = self.structure.get_grow_path()

            thoughts = self.thought_model.generate_thoughts(thought_chain, num_thoughts)
            scores = self.thought_model.evaluate_thoughts(prompt)
            self.structure.grow_structure(
                prev_node_id=grow_node.identity,
                thoughts=thoughts,
                thought_scores=scores,
            )
