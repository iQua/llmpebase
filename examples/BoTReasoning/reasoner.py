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
        self.aggregation_type = model_config["aggregation_type"]
        self.num_iterations = model_config["n_iteration"]

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
            prompt_sample = self.thought_model.add_experience(prompt_sample)

            # Reasoning with base tree structures with the prompt enhanced with the experience in the root
            base_tree_chains = []
            base_tree_llm_configs = {}
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
                base_tree.construct_root(thought=prompt_sample, thought_score=None)
                # Grow the thought structure
                base_tree.build_structure()

                # Save the graph into the disk
                base_tree.save_structure()

                # Get the chain and save it
                solution_chain = self.solution_extractor.extract_solution_chains(
                    base_tree
                )

                base_tree_chains.append((tree_idx, solution_chain))

                # Get the solutions from the structure
                solution_strs = self.get_solution_paths(structure=base_tree)

                # Save the configs
                base_tree_llm_configs.update(
                    {
                        tree_idx: {
                            "growth_type": base_tree.growth_type,
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                    }
                )

            # Perform the aggregation of the solutions
            aggregated_chain = (
                aggregator.ReasoningChainAggregator.best_first_aggregation(
                    dict(base_tree_chains)
                )
            )
            solution_str = aggregated_chain
            # Comment on the chain
            feedback = self.comment_model.comment_reasoning_chain(
                prompt_sample, solution_str
            )

            # Add the feedback to the memory
            self.thought_model.memory_experience(feedback)

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
