"""
The thought prompter for the pRAR.
"""

import copy
from typing import List

import pRAR_prompts
import pRAR_system_prompts
from policy_tree import PolicyNode

from llmpebase.model.thought_structure import base
from llmpebase.prompt.generic import BasicThoughtPromptFormat
from llmpebase.model.prompting.thought_prompter import ThoughtStructurePrompter
from llmpebase.prompt import format_prompt


class PolicyThoughtPrompter(ThoughtStructurePrompter):
    """
    A thought prompter to organize the thought prompts with policy.
    """

    # We present the init function here show what system and thought prompts
    # are required.
    def __init__(
        self,
        system_prompts: pRAR_system_prompts.PolicySystemPrompts,
        thought_prompts: pRAR_prompts.BasePolicyThoughtPrompts,
        policy_prompts: pRAR_prompts.PolicyPrompts,
    ):
        super().__init__(system_prompts, thought_prompts)

        self.policy_prompts = policy_prompts

        self.policy_head = self.policy_prompts.policy_head
        self.policy_start_flag = self.policy_prompts.policy_start_flag
        self.policy_end_flag = self.policy_prompts.policy_end_flag

    def organize_next_thought_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        **kwargs,
    ) -> BasicThoughtPromptFormat:
        """
        Organizing the I_G^{prime} of the p-RAR paper.
        """
        policy_nodes: List[PolicyNode] = kwargs["policy_chain"]

        root_prompt = str(chain_nodes[0].thought)

        # When the chain only contain the root
        # There is no no policy
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)

            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_step_idx=True,
            with_evaluation_score=False,
        )

        policy_chain_prompt = self.organize_node_block_prompt(
            nodes=policy_nodes[1:],
            content_attr="policy",
            head_format=self.policy_prompts.policy_head,
            start_flag=self.policy_prompts.policy_chain_start_flag,
            end_flag=self.policy_prompts.policy_chain_end_flag,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt  # , policy_chain_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            # self.policy_prompts.policy_chain_start_flag,
            len(chain_nodes),
        )

        return generation_prompt

    def organize_policy_guide_thought_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        policy_chain: List[PolicyNode],
        guide_policy_node: PolicyNode,
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt I_G of the p-RAR paper."""

        root_prompt = str(chain_nodes[0].thought)

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_step_idx=True,
            with_evaluation_score=False,
        )
        # Create the prompt for the policy chain
        policy_prompt = self.organize_node_block_prompt(
            nodes=policy_chain[1:],
            content_attr="policy",
            head_format=self.policy_prompts.policy_head,
            start_flag=self.policy_prompts.policy_chain_start_flag,
            end_flag=self.policy_prompts.policy_chain_end_flag,
        )
        # Create the prompt for a guide prompt
        policy_guide_prompt = self.organize_node_block_prompt(
            nodes=[guide_policy_node],
            content_attr="policy",
            head_format=self.policy_prompts.policy_head,
            start_flag=self.policy_prompts.policy_start_flag,
            end_flag=self.policy_prompts.policy_start_flag,
        )

        # The chain only contain the first step
        if len(chain_nodes) == 1:

            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.policy_guide_first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                policy_guide_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                self.policy_start_flag
            )
            return generation_prompt

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.policy_guide_next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt, policy_prompt, policy_guide_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            self.policy_prompts.policy_chain_start_flag,
            guide_policy_node.step_idx,
            self.policy_prompts.policy_start_flag,
            len(chain_nodes),
        )

        return generation_prompt

    def organize_policy_exclusive_thought_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        policy_chain: List[PolicyNode],
        policy_exclusion_candidates: List[PolicyNode],
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt I_E of the p-RAR paper."""
        root_prompt = str(chain_nodes[0].thought)

        # Create the prompt for the policy chain
        policy_chain_prompt = ""
        if len(policy_chain) > 1:
            policy_chain_prompt = self.organize_node_block_prompt(
                nodes=policy_chain[1:],
                content_attr="policy",
                head_format=self.policy_prompts.policy_head,
                start_flag=self.policy_prompts.policy_chain_start_flag,
                end_flag=self.policy_prompts.policy_chain_end_flag,
            )

        policy_exclusion_prompt = self.organize_node_block_prompt(
            nodes=policy_exclusion_candidates,
            content_attr="policy",
            head_format=self.policy_prompts.policy_head,
            start_flag=self.policy_prompts.policy_exclusion_start_flag,
            end_flag=self.policy_prompts.policy_exclusion_end_flag,
        )

        # The chain only contain the first step
        if len(chain_nodes) == 1:
            # Generate as usual when nothing to be excluded
            if len(policy_exclusion_candidates) == 0:
                generation_prompt = BasicThoughtPromptFormat(
                    **self.generation_prompts.first_step_prompt
                )
                generation_prompt.head = generation_prompt.head.format(root_prompt)
                return generation_prompt

            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.exclusive_policy_first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                policy_exclusion_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                self.policy_prompts.policy_exclusion_start_flag
            )
            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_evaluation_score=False,
            with_step_idx=True,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.exclusive_policy_next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt, policy_chain_prompt, policy_exclusion_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            self.policy_prompts.policy_chain_start_flag,
            policy_exclusion_candidates[0].step_idx,
            self.policy_prompts.policy_exclusion_start_flag,
            len(chain_nodes),
        )

        return generation_prompt

    def organize_policy_summary_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        policy_chain: List[PolicyNode],
        policy_thought_node: base.BasicNode,
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt for the policy summary."""
        root_prompt = copy.deepcopy(chain_nodes[0].thought)
        root_prompt.head = ""
        root_prompt.notice = ""
        root_prompt.solution_flag = ""
        root_prompt.answer = ""
        root_prompt = str(root_prompt)

        # Create the prompt for the policy chain
        policy_chain_prompt = ""
        if len(policy_chain) > 1:
            policy_chain_prompt = self.organize_node_block_prompt(
                nodes=policy_chain[1:],
                content_attr="policy",
                head_format=self.policy_prompts.policy_head,
                start_flag=self.policy_prompts.policy_chain_start_flag,
                end_flag=self.policy_prompts.policy_chain_end_flag,
            )

        # Create the policy thought prompt
        policy_thought_prompt = self.organize_node_block_prompt(
            nodes=[policy_thought_node],
            content_attr="thought",
            head_format=self.step_head,
            start_flag=self.policy_prompts.step_start_flag,
            end_flag=self.policy_prompts.step_end_flag,
        )

        # The chain only contain the first step
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.policy_prompts.first_policy_summarization_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                policy_thought_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                self.policy_prompts.step_start_flag
            )
            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_evaluation_score=False,
            with_step_idx=True,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.policy_prompts.policy_summarization_prompt
        )
        generation_prompt.head = generation_prompt.head.format(
            root_prompt, policy_thought_node.step_idx
        )
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt, policy_chain_prompt, policy_thought_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            self.policy_prompts.policy_chain_start_flag,
            policy_thought_node.step_idx,
            self.thought_chain_start_flag,
            policy_thought_node.step_idx,
        )

        return generation_prompt

    def organize_policy_compare_prompt(
        self, policy_nodes: List[PolicyNode], target_policy_thought_node: base.BasicNode
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt for comparing the policy."""
        # Create the prompt for the policy chain

        policy_pool_prompt = self.organize_node_block_prompt(
            nodes=policy_nodes,
            content_attr="policy",
            with_index=True,
            head_format=self.policy_prompts.policy_head,
            start_flag=self.policy_prompts.policy_comparison_start_flag,
            end_flag=self.policy_prompts.policy_comparison_end_flag,
        )
        # Create the prompt for the target policy
        target_policy_prompt = self.organize_node_block_prompt(
            nodes=[target_policy_thought_node],
            content_attr="thought",
            head_format=self.policy_prompts.policy_head,
            start_flag=self.policy_prompts.policy_start_flag,
            end_flag=self.policy_prompts.policy_start_flag,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.policy_prompts.compare_policy_prompt
        )

        generation_prompt.content = generation_prompt.content.format(
            policy_pool_prompt, target_policy_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            target_policy_thought_node.step_idx,
            self.policy_prompts.policy_start_flag,
            self.policy_prompts.policy_comparison_start_flag,
        )

        return generation_prompt

    def organize_policy_assessment_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        thought_node: base.BasicNode,
        policy_thought_node: base.BasicNode = None,
        policy_node: PolicyNode = None,
    ) -> BasicThoughtPromptFormat:
        """
        Organizing the prompt for assessing the policy.
        """
        root_prompt = copy.deepcopy(chain_nodes[0].thought)
        root_prompt.head = ""
        root_prompt.notice = ""
        root_prompt.solution_flag = ""
        root_prompt.answer = ""
        root_prompt = str(root_prompt)

        # Create the prompt for the policy thought
        assess_policy_prompt = None
        assess_policy_step_idx = None
        if policy_thought_node is not None:
            assess_policy_prompt = self.organize_node_block_prompt(
                nodes=[policy_thought_node],
                content_attr="thought",
                head_format=self.policy_prompts.policy_head,
                start_flag=self.policy_prompts.policy_assessment_start_flag,
                end_flag=self.policy_prompts.policy_assessment_end_flag,
            )
            assess_policy_step_idx = policy_thought_node.step_idx
        if policy_node is not None:
            assess_policy_prompt = self.organize_node_block_prompt(
                nodes=[policy_node],
                content_attr="policy",
                head_format=self.policy_prompts.policy_head,
                start_flag=self.policy_prompts.policy_assessment_start_flag,
                end_flag=self.policy_prompts.policy_assessment_end_flag,
            )
            assess_policy_step_idx = policy_node.step_idx
        assert assess_policy_prompt is not None

        # Create the prompt for the thought
        thought_prompt = self.organize_node_block_prompt(
            nodes=[thought_node],
            content_attr="thought",
            head_format=self.step_head,
            start_flag=self.policy_prompts.step_start_flag,
            end_flag=self.policy_prompts.step_end_flag,
        )

        # Create the prompt for the policy assessment of the first step
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.policy_prompts.assess_first_policy_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                thought_prompt, assess_policy_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                assess_policy_step_idx,
                self.policy_prompts.policy_assessment_start_flag,
                self.policy_prompts.step_start_flag,
                self.policy_prompts.policy_assessment_start_flag,
            )
            return generation_prompt

        # When there are multiple existing reasoning steps, create the prompt of the policy assessment of the next reasoning step

        thought_chain_prompt = self.organize_node_block_prompt(
            nodes=chain_nodes[1:],
            content_attr="thought",
            head_format=self.step_head,
            start_flag=self.thought_chain_start_flag,
            end_flag=self.thought_chain_end_flag,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.policy_prompts.assess_next_policy_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            thought_chain_prompt, thought_prompt, assess_policy_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            thought_node.step_idx,
            self.policy_prompts.step_start_flag,
            assess_policy_step_idx,
            self.policy_prompts.policy_assessment_start_flag,
            assess_policy_step_idx,
        )
        return generation_prompt
