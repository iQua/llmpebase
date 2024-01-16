"""
The implementation of the thought model used to build the thought structure.
"""
import re
from typing import List

from llmpebase.model.thought_structure import thought_model
from llmpebase.model.thought_structure import base


class TRThoughtModel(thought_model.LlmThoughtModel):
    """
    A thought model built upon the LLM model to perform the rollback
    of thought.
    """

    def extract_rollback_index(self, text_str: str, solution_flag: str):
        """Extract the indexes of the rollback"""

        # Extracting all content after the solution_flag
        extracted_content = re.search(f"{solution_flag}(.*)", text_str, re.IGNORECASE)
        result = "None"
        if extracted_content:
            result = extracted_content.group(1)
        # Extract the indexes
        if "None" in result:
            return None
        else:
            pattern = r"\d+"

            # Extracting digital numbers
            indexes = [int(elem) for elem in re.findall(pattern, result)]

            return indexes

    def generate_rollback(
        self,
        thought_chain: List[base.BasicNode],
    ):
        """Generate the rollback condition based on the thought chain."""
        # First, analyze the reasoning process
        analysis_prompt = self.prompter.organize_reasoning_analysis_prompt(
            chain_nodes=thought_chain
        )
        responses = self.llm_model.forward(
            user_prompt=str(analysis_prompt),
            per_request_responses=1,
            sys_prompt=self.prompter.reasoning_analysis_system_prompt,
        )
        reasoning_analysis = self.llm_model.read_response_contents(responses)[0]

        # Second, get the rollback index
        rollback_prompt = self.prompter.organize_prompt_controller_prompt(
            chain_nodes=thought_chain, reasoning_analysis=reasoning_analysis
        )
        responses = self.llm_model.forward(
            user_prompt=str(rollback_prompt),
            per_request_responses=1,
            sys_prompt=self.prompter.rollback_system_prompt,
        )
        rollback_result = self.llm_model.read_response_contents(responses)[0]
        # Get the step idx that rollback to
        # This index is the step of thought_chain[1:]
        rollback_step_idxes = self.extract_rollback_index(
            rollback_result, self.prompter.rollback_solution_flag
        )
        # print("*" * 30)
        # print(analysis_prompt)
        # print("+" * 30)
        # print(reasoning_analysis)
        # print("+" * 30)
        # print(rollback_prompt)
        # print("+" * 30)
        # print(rollback_result)

        return rollback_step_idxes, rollback_result, reasoning_analysis
