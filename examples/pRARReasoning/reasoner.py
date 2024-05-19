"""
A reasoner is a module to organize all components to perform reasoning.
"""

from typing import List

from embedder import GPTEmbedder
from mcts_thought_structure import MCTSPolicyStructure
from policy_operator import PolicyOperator

from llmpebase.reasoner import structured_thought
from llmpebase.model.prompting.base import BasicSamplePrompt


class PolicyThoughtReasoner(structured_thought.StructuredThoughtReasoner):
    """A reasoner for the p-RAR method to perform a policy-guided reasoning."""

    def __init__(
        self,
        thought_model,
        model_config,
        logging_config,
        visualizer,
        solution_extractor,
        policy_operator: PolicyOperator = None,
    ):
        super().__init__(
            thought_model=thought_model,
            model_config=model_config,
            logging_config=logging_config,
            visualizer=visualizer,
            solution_extractor=solution_extractor,
        )

        self.policy_operator = (
            policy_operator
            if policy_operator is not None
            else PolicyOperator(logging_config=self.logging_config)
        )

    def define_structure(self):
        """Define the thought structure to be used."""
        # Define the embedder for the mcts reasoning
        embedder = GPTEmbedder(model_config=self.model_config)

        return MCTSPolicyStructure(
            thought_model=self.thought_model,
            model_config=self.model_config,
            logging_config=self.logging_config,
            visualizer=self.visualizer,
            embedder=embedder,
        )

    def forward(
        self,
        prompt_sample: BasicSamplePrompt,
        sample_name: str = "0-0",
        sample_info: dict = None,
    ) -> List[str]:
        """Forward the reasoning in the thought structure."""
        # One must load the specific policy tree to obtain the policy tree before reasoning

        # Load the policy tree if it exists
        loaded_policy_tree = self.policy_operator.load_policy_tree(
            sample_info=sample_info
        )
        # Set the policy tree to the thought structure of the reasoner
        self.structure.set_policy_tree(loaded_policy_tree)

        # Set the save path and folder for visualization and thought structure
        self.visualizer.set_save_foldername(
            f"{self.visualizer.base_save_foldername}-{sample_name}"
        )
        self.structure.set_save_foldername(
            foldername=f"{self.structure.base_save_foldername}-{sample_name}"
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
