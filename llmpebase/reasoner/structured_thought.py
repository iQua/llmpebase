"""
A reasoner to support a reasoning process with structured thought.
"""

from typing import List, Type

from torch import nn

from llmpebase.model.thought_structure.base import BaseThoughtStructure
from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer
from llmpebase.model.thought_structure.solution_extractor import SolutionExtractor


class StructuredThoughtReasoner:
    """
    A reasoner to answer the question with the reasoning process
    built upon one specific thought structure.

    Args:
        thought_model: A defined thought model used to generate thought with
         the implemented functions `generate_thoughts`.
    """

    def __init__(
        self,
        thought_model: nn.Module,
        model_config: dict = None,
        logging_config: dict = None,
        visualizer: BasicStructureVisualizer = None,
        solution_extractor: SolutionExtractor = None,
    ):
        self.model_config = model_config
        self.logging_config = logging_config
        # The thought model used to generate thoughts
        # in the structure
        self.thought_model = thought_model
        # The visualizer to visualize the thought structure
        self.visualizer = (
            BasicStructureVisualizer(logging_config=logging_config)
            if visualizer is None
            else visualizer
        )
        # The thought structure used to perform the reasoning
        self.structure = self.define_structure()
        # The extractor to get solutions from the thought structure
        self.solution_extractor = (
            SolutionExtractor() if solution_extractor is None else solution_extractor
        )

    def define_structure(self) -> Type[BaseThoughtStructure]:
        """Define the thought structure to be used."""
        raise NotImplementedError

    def get_solution_paths(self, structure: BaseThoughtStructure = None) -> List[str]:
        """Extract the reasoning paths from the thought structure as the
        solutions."""

        # Get the chain and save it
        solution_chains = self.solution_extractor.extract_solution_chains(structure)
        for idx, chain in enumerate(solution_chains):
            structure.save_node_path(
                chain,
                filename=f"{idx}-th_solution_chain_{chain[0].identity}->{chain[-1].identity}",
            )
        # Convert the chain into a string
        # We remove the root prompt and the evaluation score to organize
        # a prompt as the reasoning answer
        solution_strs = []
        for chain in solution_chains:
            solution_str = self.thought_model.prompter.organize_chain_prompt(
                chain_nodes=chain[1:],
                with_step_idx=False,
                with_flag=False,
                with_evaluation_score=False,
            )
            solution_strs.append(solution_str)
        return solution_strs

    def forward(
        self,
        prompt_sample: BasicSamplePrompt,
        sample_name: str = "0-0",
        sample_info: dict = None,
    ) -> List[str]:
        """Forward the reasoning in the thought structure."""
        # Set the save path and folder for visualization and thought structure
        self.visualizer.set_save_foldername(
            f"{self.visualizer.base_save_foldername}-{sample_name}"
        )
        self.structure.set_save_foldername(
            foldername=f"{self.structure.save_foldername}-{sample_name}"
        )

        # Place the task prompt in the root so that all subsequent thought chains
        # include the task prompt
        self.structure.construct_root(thought=prompt_sample, thought_score=None)
        # Grow the thought structure
        self.structure.build_structure()
        # Save the graph into the disk
        self.structure.save_structure()

        # Get the solutions from the structure
        solution_strs = self.get_solution_paths(structure=self.structure)

        return solution_strs

    def reset_reasoning(self):
        """Reset the reasoning process."""
        # Reset the thought structure
        self.structure.reset_structure()

    def get_cost_statistics(self, **kwargs):
        """Get the cost statistics by using Llm."""
        # Get the statistics data
        data = self.thought_model.llm_model.get_cost_statistics(latest=False)
        # Reset the cost statistics for the llm model
        self.thought_model.llm_model.reset_cost_statistics()
        return data
