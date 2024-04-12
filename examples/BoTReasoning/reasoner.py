"""
The implementation of the Boosting of Thoughts (BoT).
"""

import random
from typing import List, Union, Tuple

import aggregator

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.thought_structure import trees
from llmpebase.reasoner.structured_thought import StructuredThoughtReasoner


class BoTReasoner(StructuredThoughtReasoner):
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
        self, thought_model, model_config, logging_config, visualizer, comment_model
    ):

        self.bot_config = model_config["bot_settings"]
        super().__init__(thought_model, model_config, logging_config, visualizer)

        # Get the basic iteration and aggregation type
        self.num_iterations = self.bot_config["n_iteration"]

        # Define the aggregator for the reasoning chains aggregation
        self.aggregator = aggregator.ReasoningChainAggregator(
            logging_config, model_config
        )

        # Get the commenter
        self.comment_model = comment_model

    def define_structure(
        self,
    ) -> List[
        Tuple[
            str,
            Union[
                Union[trees.BFGTreeThoughtStructure, trees.DFGTreeThoughtStructure],
                trees.LWGTreeThoughtStructure,
            ],
        ]
    ]:
        """Define the thought structure to be used."""
        # Build the heterogeneous thought structures
        tree_types = self.bot_config["growth_types"]
        return [
            (
                idx,
                trees.get(growth_type=random.choice(tree_types))(
                    thought_model=self.thought_model,
                    model_config=self.bot_config,
                    logging_config=self.logging_config,
                    visualizer=self.visualizer,
                ),
            )
            for idx in range(self.bot_config["num_trees"])
        ]

    def forward(
        self, prompt_sample: BasicSamplePrompt, sample_name: str = "0-0"
    ) -> List[str]:
        """Forward the reasoning in the chain structure."""

        for iter_idx in range(self.num_iterations):

            # Create the visualization path
            iteration_folder = f"iteration-{iter_idx}"

            # Place the task prompt in the root so that all subsequent thought chains include the task prompt
            # Change the root prompt by adding the experience
            experienced_prompt_sample = self.thought_model.add_experience(prompt_sample)

            # Reasoning with base tree structures with the prompt enhanced with the experience in the root
            base_tree_chains = {}
            for tree_idx, base_tree in self.structure:

                # Set the save path
                self.visualizer.set_save_foldername(
                    f"{self.visualizer.base_save_foldername}-{sample_name}/{iteration_folder}/tree-{tree_idx}"
                )

                base_tree.set_save_foldername(
                    foldername=f"{base_tree.base_save_foldername}-{sample_name}/{iteration_folder}/tree-{tree_idx}"
                )

                # Get the temperature and top_p for the base tree
                temperature = random.choice(self.bot_config["temperature_pool"])
                top_p = random.choice(self.bot_config["top_p_pool"])

                # Replace the parameters of the thought model's llm for the base tree
                base_tree.thought_model.llm_model.generation_config.update(
                    {"temperature": temperature, "top_p": top_p}
                )
                # Perform the reasoning with the base tree
                base_tree.construct_root(
                    thought=experienced_prompt_sample, thought_score=None
                )
                # Grow the thought structure
                base_tree.build_structure()

                # Save the graph into the disk
                base_tree.save_structure()

                # Get the solution chain of the tree
                base_tree_chains[tree_idx] = (
                    self.solution_extractor.extract_solution_chains(base_tree)
                )

            # Perform the aggregation of the solutions
            aggregated_chain = self.aggregator.perform_aggregation(
                structure_chains=base_tree_chains
            )
            # Save the aggregation state
            self.aggregator.save_state(
                location=f"{self.visualizer.base_save_foldername}-{sample_name}/{iteration_folder}",
                file_name="aggregator-state",
            )

            # Convert the chain to a solution str
            solution_str = self.thought_model.prompter.organize_chain_prompt(
                chain_nodes=aggregated_chain[1:],
                with_step_idx=False,
                with_flag=False,
                with_evaluation_score=False,
            )

            # Comment on the chain
            feedback = self.comment_model.comment_reasoning_chain(
                prompt_sample, solution_str
            )
            self.comment_model.save_state(
                location=f"{self.visualizer.base_save_foldername}-{sample_name}/{iteration_folder}",
                file_name="commenter-state",
            )

            # Add the solution and its feedback to the memory
            self.thought_model.memorize_experience(solution_str, feedback)

            # Clean the structure after the reasoning
            for _, base_tree in self.structure:
                base_tree.reset_structure()

        return [solution_str]

    def get_cost_statistics(self, **kwargs):
        """Get the cost statistics by using Llm."""
        # Get the statistics data
        data = self.thought_model.llm_model.get_cost_statistics(latest=False)
        # Reset the cost statistics for the llm model
        self.thought_model.llm_model.reset_cost_statistics()
        return data

    def reset_reasoning(self):
        """Reset the reasoning process."""
        # Reset the thought structure
        # Clean the structure after the reasoning
        for _, base_tree in self.structure:
            base_tree.reset_structure()
