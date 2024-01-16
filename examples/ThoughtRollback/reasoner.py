"""
A reasoner to perform adaptive reasoning with the thought rollback.
"""
from typing import List

from torch import nn

import TR_structure
import visualization
import chain_extractor

from llmpebase.model.prompting.base import BasicSamplePrompt


class ThoughtRollbackReasoner:
    """
    A TR reasoner to answer the question by rolling back with the request model.

    Args:
        thought_model: A defined thought model used to generate thought
        during the growth of the chain structure. For the required functions
        of this mode, please access the thought_model.py file.
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
        self.visualizer = visualization.TRVisualizer(logging_config=logging_config)

        self.structure = TR_structure.ThoughtRollbackStructure(
            thought_model=self.thought_model,
            model_config=model_config,
            logging_config=logging_config,
            visualizer=self.visualizer,
        )

        self.solution_extractor = chain_extractor.SolutionExtractor()

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
        solution_chains = self.solution_extractor.extract_solution_chains(
            self.structure
        )
        for idx, chain in enumerate(solution_chains):
            self.structure.save_thought_path(
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
        # Clean the structure after the reasoning
        self.structure.reset_structure()

        return solution_strs

    def get_cost_statistics(self, **kwargs):
        """Get the cost statistics by using Llm."""
        # Get the statistics data
        data = self.thought_model.llm_model.get_cost_statistics(latest=False)
        # Reset the cost statistics for the llm model
        self.thought_model.llm_model.reset_cost_statistics()
        return data
