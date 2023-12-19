"""
A reasoner to perform the reasoning step by step in a chain structure.
"""

from torch import nn

from llmpebase.model.thought_structure import chains
from llmpebase.model.prompting.base import BasicPromptSample
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer
from llmpebase.model.thought_structure.chain_extractors import ChainExtractor


class ChainReasoner:
    """A CoT reasoner to answer the question with the request model.

    Args:
        thought_model: A defined thought model used to generate thought with
         the implemented functions `generate_thoughts`.
    """

    def __init__(
        self,
        thought_model: nn.Module,
        model_config: dict = None,
        logging_config: dict = None,
    ):
        # The thought model used to generate thoughts
        # in the structure
        self.thought_model = thought_model
        # The visualizer to visualize the thought structure
        self.visualizer = BasicStructureVisualizer(logging_config=logging_config)

        self.structure = chains.ChainThoughtStructure(
            model_config, logging_config=logging_config
        )

        self.solution_extractor = ChainExtractor()

    def forward(self, prompt_sample: BasicPromptSample):
        """Forward the reasoning in the chain structure."""
        # Make changes to the prompt sample
        # Place the task prompt in the root so that all subsequent thought chains
        # include the task prompt
        self.structure.construct_root(thought=prompt_sample, thought_score=None)
        # Grow the thought structure
        self.structure.build_structure(
            thought_model=self.thought_model, visualizer=self.visualizer
        )

        # Get the chain and save it
        solution_chain = self.solution_extractor.extract_thought_chain(self.structure)
        self.structure.save_thought_path(solution_chain, path_name="solution_chain")
        # Convert the chain into a string
        solution_str = self.thought_model.prompter.organize_chain_prompt(
            chain_nodes=solution_chain
        )

        return solution_str
