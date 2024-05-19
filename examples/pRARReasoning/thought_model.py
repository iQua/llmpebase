"""
Implementation of Thought model for the pRAR Reasoning.
"""

from collections import namedtuple
from typing import List, Tuple

from policy_tree import PolicyNode
from thought_prompter import PolicyThoughtPrompter


from llmpebase.model.thought_structure.structure_generic import (
    BasicPromptAndResponse,
)
from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure import thought_model


InferenceInfo = namedtuple("InferenceInfo", {"input_prompt", "output"})


class PolicyThoughtModel(thought_model.LlmThoughtModel):
    """
    A thought model to organize and utilize the policy to generate
    thoughts.
    """

    def generate_thoughts(
        self,
        thought_chain: List[base.BasicNode],
        num_thoughts: int = 1,
        policy_chain: List[PolicyNode] = None,
        **kwargs
    ) -> Tuple[List[str], List[BasicPromptAndResponse]]:
        """
        Generate the thoughts normally based on the I_G^{prime} of the p-RAR paper.
        """

        # Create the reasoning chain prompt
        prompt = self.prompter.organize_next_thought_prompt(
            chain_nodes=thought_chain, policy_chain=policy_chain
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

    def generate_policy_next_thoughts(
        self,
        thought_chain: List[base.BasicNode],
        policy_chain: List[PolicyNode],
        policy_node: PolicyNode,
        num_thoughts: int = 1,
    ):
        """Generate thoughts based on the prompt I_G of the p-RAR paper."""

        # Create the reasoning chain prompt
        prompt = self.prompter.organize_policy_guide_thought_prompt(
            chain_nodes=thought_chain,
            policy_chain=policy_chain,
            guide_policy_node=policy_node,
        )

        system_prompt = self.prompter.system_prompts.policy_guided_generation_prompt

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

    def generate_excluded_policy_thoughts(
        self,
        thought_chain: List[base.BasicNode],
        num_thoughts: int = 1,
        policy_chain: List[PolicyNode] = None,
        policy_exclusion_candidates: List[PolicyNode] = None,
    ):
        """Generate thoughts based on the prompt I_G of the p-RAR paper."""

        # Create the reasoning chain prompt
        prompt = self.prompter.organize_policy_exclusive_thought_prompt(
            chain_nodes=thought_chain,
            policy_chain=policy_chain,
            policy_exclusion_candidates=policy_exclusion_candidates,
        )

        system_prompt = self.prompter.system_prompts.policy_exclusion_generation_prompt

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

    def summarize_policy(
        self,
        thought_chain: List[base.BasicNode],
        policy_chain: List[PolicyNode] = None,
        policy_thought_node: base.BasicNode = None,
        num_thoughts: int = 1,
    ):
        """Summarize the policy from a thought."""

        prompt = self.prompter.organize_policy_summary_prompt(
            chain_nodes=thought_chain,
            policy_chain=policy_chain,
            policy_thought_node=policy_thought_node,
        )

        system_prompt = self.prompter.system_prompts.policy_summarization_prompt

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

    def compare_policy(
        self,
        target_policy_thought_node: base.BasicNode,
        policy_node_pool: List[PolicyNode],
        num_thoughts: int = 1,
    ):
        """Compare whether the target policy exists in the policy pool."""

        prompt = self.prompter.organize_policy_compare_prompt(
            policy_nodes=policy_node_pool,
            target_policy_thought_node=target_policy_thought_node,
        )

        system_prompt = self.prompter.system_prompts.policy_comparison_prompt

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

    def assess_thought_policy(
        self,
        thought_chain: List[base.BasicNode],
        thought_node: base.BasicNode,
        policy_thought_node: base.BasicNode = None,
        policy_node: PolicyNode = None,
        num_thoughts: int = 1,
    ):
        """
        Assess the policy of a thought based on the I_A of the p-RAR paper.

        This function handles two cases:
        1. The policy is provided as a thought node of the thought structure
        2. The policy is directly from the policy tree as a policy node

        """
        prompt = self.prompter.organize_policy_assessment_prompt(
            chain_nodes=thought_chain,
            thought_node=thought_node,
            policy_thought_node=policy_thought_node,
            policy_node=policy_node,
        )

        system_prompt = self.prompter.system_prompts.thought_policy_assessment_prompt

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
