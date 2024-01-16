"""
The implementation of the Boosting of Thoughts (BoT).
"""
from typing import List
from torch import nn

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.thought_structure import trees
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer
from llmpebase.model.thought_structure.solution_extractor import SolutionExtractor


class BoTReasoner:
    """
    A BoT reasoner to perform the reasoning based on the trial-and-error experiences.

    Args:
        thought_model: A defined thought model used to generate thought
         during the growth of the chain structure. For the required functions
         of this mode, please access the thought_model.py file.
        chain_commenter: A defined comment model used to analysis the obtained
         reasoning chain to gain the experience, i.e., the error reports and advice.
    """

    def __init__(
        self,
        thought_model: nn.Module,
        chain_commenter: nn.Module,
        model_config: dict = None,
        logging_config: dict = None,
    ):
        # The thought model used to generate thoughts
        # in the structure
        self.thought_model = thought_model
        # The commenter to comment on the generated chain
        self.chain_commenter = chain_commenter
        # The visualizer to visualize the thought structure
        self.visualizer = BasicStructureVisualizer(logging_config=logging_config)

        structure_config = model_config["thought_structure"]
        self.structure = trees.get(growth_type=structure_config["growth_type"])(
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
        # Change the root prompt by adding the experience
        prompt_sample = self.thought_model.add_experience(prompt_sample)

        self.structure.construct_root(thought=prompt_sample, thought_score=None)
        # Grow the thought structure
        self.structure.build_structure()
        # Save the graph into the disk
        self.structure.save_structure()

        # Get the chain and save it
        solution_chain = self.solution_extractor.extract_solution_chains(self.structure)
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

        # Comment on the chain
        feedback = self.chain_commenter.comment_reasoning_chain(
            prompt_sample, solution_str
        )

        # Add the feedback to the memory
        self.thought_model.memory_experience(feedback)

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
