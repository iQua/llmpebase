"""
Implementation of Thought model for the pRAR Reasoning.
"""

from collections import namedtuple
from typing import List, Tuple

from plan_tree import PlanNode
from thought_prompter import PlanThoughtPrompter


from llmpebase.model.thought_structure.structure_generic import (
    BasicPromptAndResponse,
)
from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure import thought_model


InferenceInfo = namedtuple("InferenceInfo", {"input_prompt", "output"})


class PlanThoughtModel(thought_model.LlmThoughtModel):
    """
    A thought model to organize and utilize the plan to generate
    thoughts.
    """

    def generate_thoughts(
        self,
        thought_chain: List[base.BasicNode],
        num_thoughts: int = 1,
        plan_chain: List[PlanNode] = None,
        **kwargs
    ) -> Tuple[List[str], List[BasicPromptAndResponse]]:
        """
        Generate the thoughts normally based on the I_G^{prime} of the p-RAR paper.
        """

        generation_config = self.model_config["generation_settings"]
        self.llm_model.generation_config.update(generation_config)
        # Create the reasoning chain prompt
        prompt = self.prompter.organize_next_thought_prompt(
            chain_nodes=thought_chain, plan_chain=plan_chain
        )

        # Equip the model with the normal thought generation configuration
        system_prompt = self.prompter.generation_system_prompt
        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        inference_info = [
            BasicPromptAndResponse(
                prompt=prompt, response=thought, system=system_prompt
            )
            for thought in thoughts
        ]

        return thoughts, inference_info

    def generate_plan_next_thoughts(
        self,
        thought_chain: List[base.BasicNode],
        plan_chain: List[PlanNode],
        plan_node: PlanNode,
        num_thoughts: int = 1,
    ):
        """Generate thoughts based on the prompt I_G of the p-RAR paper."""
        generation_config = self.model_config["optimization"]["mcts"][
            "generation_settings"
        ]["plan_guided_generation"]
        self.llm_model.generation_config.update(generation_config)
        # Create the reasoning chain prompt
        prompt = self.prompter.organize_plan_guide_thought_prompt(
            chain_nodes=thought_chain,
            plan_chain=plan_chain,
            guide_plan_node=plan_node,
        )

        system_prompt = self.prompter.system_prompts.plan_guided_generation_prompt

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        inference_info = [
            BasicPromptAndResponse(
                prompt=prompt, response=thought, system=system_prompt
            )
            for thought in thoughts
        ]

        return thoughts, inference_info

    def generate_excluded_plan_thoughts(
        self,
        thought_chain: List[base.BasicNode],
        num_thoughts: int = 1,
        plan_chain: List[PlanNode] = None,
        plan_exclusion_candidates: List[PlanNode] = None,
    ):
        """Generate thoughts based on the prompt I_G of the p-RAR paper."""
        generation_config = self.model_config["optimization"]["mcts"][
            "generation_settings"
        ]["plan_exclusion_generation"]
        self.llm_model.generation_config.update(generation_config)
        # Create the reasoning chain prompt
        prompt = self.prompter.organize_plan_exclusive_thought_prompt(
            chain_nodes=thought_chain,
            plan_chain=plan_chain,
            plan_exclusion_candidates=plan_exclusion_candidates,
        )

        system_prompt = self.prompter.system_prompts.plan_exclusion_generation_prompt

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        inference_info = [
            BasicPromptAndResponse(
                prompt=prompt, response=thought, system=system_prompt
            )
            for thought in thoughts
        ]

        return thoughts, inference_info

    def summarize_plan(
        self,
        thought_chain: List[base.BasicNode],
        plan_chain: List[PlanNode] = None,
        thought_plan_node: base.BasicNode = None,
        num_thoughts: int = 1,
    ):
        """Summarize the plan from a thought."""
        generation_config = self.model_config["optimization"]["mcts"][
            "generation_settings"
        ]["plan_summarization"]
        self.llm_model.generation_config.update(generation_config)
        prompt = self.prompter.organize_plan_summary_prompt(
            chain_nodes=thought_chain,
            plan_chain=plan_chain,
            thought_node=thought_plan_node,
        )

        system_prompt = self.prompter.system_prompts.plan_summarization_prompt

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        inference_info = [
            BasicPromptAndResponse(
                prompt=prompt, response=thought, system=system_prompt
            )
            for thought in thoughts
        ]

        return thoughts, inference_info

    def compare_plan(
        self,
        target_thought_plan_node: base.BasicNode,
        plan_node_pool: List[PlanNode],
        num_thoughts: int = 1,
    ):
        """Compare whether the target plan exists in the plan pool."""
        generation_config = self.model_config["optimization"]["mcts"][
            "generation_settings"
        ]["plan_comparison"]
        self.llm_model.generation_config.update(generation_config)
        prompt = self.prompter.organize_plan_compare_prompt(
            plan_nodes=plan_node_pool,
            target_thought_plan_node=target_thought_plan_node,
        )

        system_prompt = self.prompter.system_prompts.plan_comparison_prompt

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        inference_info = [
            BasicPromptAndResponse(
                prompt=prompt, response=thought, system=system_prompt
            )
            for thought in thoughts
        ]

        return thoughts, inference_info

    def assess_thought_plan(
        self,
        thought_chain: List[base.BasicNode],
        thought_node: base.BasicNode,
        thought_plan_node: base.BasicNode = None,
        plan_node: PlanNode = None,
        num_thoughts: int = 1,
    ):
        """
        Assess the plan of a thought based on the I_A of the p-RAR paper.

        This function handles two cases:
        1. The plan is provided as a thought node of the thought structure
        2. The plan is directly from the plan tree as a plan node

        """

        generation_config = self.model_config["optimization"]["mcts"][
            "generation_settings"
        ]["thought_plan_assessment"]
        self.llm_model.generation_config.update(generation_config)

        prompt = self.prompter.organize_plan_assessment_prompt(
            chain_nodes=thought_chain,
            thought_node=thought_node,
            thought_plan_node=thought_plan_node,
            plan_node=plan_node,
        )

        system_prompt = self.prompter.system_prompts.thought_plan_assessment_prompt

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        inference_info = [
            BasicPromptAndResponse(
                prompt=prompt, response=thought, system=system_prompt
            )
            for thought in thoughts
        ]

        return thoughts, inference_info
