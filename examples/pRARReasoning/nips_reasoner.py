"""
Reasoner for the NIPS submission.
"""

from typing import List

from nips_thought_structure import NIPSPlanStructure
from plan_tree import PlanTree

from llmpebase.reasoner import structured_thought
from llmpebase.model.prompting.base import BasicSamplePrompt


class NIPSPlanThoughtReasoner(structured_thought.StructuredThoughtReasoner):
    """A reasoner for the p-RAR method to perform a plan-guided reasoning."""

    def define_structure(self):
        """Define the thought structure to be used."""

        return NIPSPlanStructure(
            thought_model=self.thought_model,
            model_config=self.model_config,
            logging_config=self.logging_config,
            visualizer=self.visualizer,
        )

    def forward(
        self,
        prompt_sample: BasicSamplePrompt,
        sample_name: str = "0-0",
        sample_info: dict = None,
    ) -> List[str]:
        """Forward the reasoning in the thought structure."""
        # One must load the specific plan tree to obtain the plan tree before reasoning
        plan_tree = PlanTree(logging_config=self.logging_config, visualizer=None)
        sample_field = sample_info["sample_field"]
        problem_category = sample_info["sample_problem"]

        plan_tree.construct_root(task_info=sample_field, category_name=problem_category)

        self.structure.plan_tree = plan_tree

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
