"""
A reasoner to perform the reasoning step by step in a chain structure.
"""

from typing import List

from torch import nn

from llmpebase.model.thought_structure import graphs
from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer
from llmpebase.model.thought_structure.chain_extractors import SolutionExtractor


class GraphReasoner:
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

        self.structure = graphs.GraphTreeThoughtStructure(
            thought_model=self.thought_model,
            model_config=model_config,
            logging_config=logging_config,
            visualizer=self.visualizer,
        )

        self.solution_extractor = SolutionExtractor()

    def forward(
        self, prompt_sample: BasicSamplePrompt, sample_idx: int = 0
    ) -> List[str]:
        """Forward the reasoning in the chain structure."""
        # Create the visualization path
        structure_folder = f"thought_structure_{sample_idx}"
        self.visualizer.visualization_foldername = structure_folder
        self.structure.save_foldername = structure_folder

        # Place the task prompt in the root so that all subsequent thought chains
        # include the task prompt
        self.structure.construct_root(thought=prompt_sample, thought_score=None)
        # Grow the thought structure
        self.structure.build_structure()
        # Save the graph into the disk
        self.structure.save_structure()

        # Get the chain and save it
        solution_chain = self.solution_extractor.extract_solution_chain(self.structure)
        self.structure.save_thought_path(
            solution_chain,
            filename="solution_chain",
        )

        # Convert the chain into a string
        # We remove the root prompt and the evaluation score to organize
        # a prompt as the reasoning answer
        solution_str = self.thought_model.prompter.organize_chain_prompt(
            chain_nodes=solution_chain[1:],
            with_step_idx=False,
            with_flag=False,
            with_evaluation_score=False,
        )
        # Clean the structure after the reasoning
        self.structure.reset_structure()

        return [solution_str]

    def get_cost_statistics(self, **kwargs):
        """Get the cost statistics by using Llm."""
        # Get the statistics data
        data = self.thought_model.llm_model.get_cost_statistics(latest=False)
        # Reset the cost statistics for the llm model
        self.thought_model.llm_model.reset_cost_statistics()
        return data
