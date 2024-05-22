"""
The thought prompter for the pRAR.
"""

import copy
from typing import List

import pRAR_prompts
import pRAR_system_prompts
from plan_tree import PlanNode

from llmpebase.model.thought_structure import base
from llmpebase.prompt.generic import BasicThoughtPromptFormat
from llmpebase.model.prompting.thought_prompter import ThoughtStructurePrompter
from llmpebase.prompt import format_prompt


class PlanThoughtPrompter(ThoughtStructurePrompter):
    """
    A thought prompter to organize the thought prompts with plan.
    """

    # We present the init function here show what system and thought prompts
    # are required.
    def __init__(
        self,
        system_prompts: pRAR_system_prompts.PlanSystemPrompts,
        thought_prompts: pRAR_prompts.BasePlanThoughtPrompts,
        plan_prompts: pRAR_prompts.PlanPrompts,
    ):
        super().__init__(system_prompts, thought_prompts)

        self.plan_prompts = plan_prompts

        self.plan_head = self.plan_prompts.plan_head
        self.plan_start_flag = self.plan_prompts.plan_start_flag
        self.plan_end_flag = self.plan_prompts.plan_end_flag

    def organize_next_thought_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        **kwargs,
    ) -> BasicThoughtPromptFormat:
        """
        Organizing the I_G^{prime} of the p-RAR paper.
        """
        plan_nodes: List[PlanNode] = kwargs["plan_chain"]

        root_prompt = str(chain_nodes[0].thought)

        # When the chain only contain the root
        # There is no no plan
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

        plan_chain_prompt = self.organize_node_block_prompt(
            nodes=plan_nodes[1:],
            content_attr="plan",
            head_format=self.plan_prompts.plan_head,
            start_flag=self.plan_prompts.plan_chain_start_flag,
            end_flag=self.plan_prompts.plan_chain_end_flag,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt  # , plan_chain_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            # self.plan_prompts.plan_chain_start_flag,
            len(chain_nodes),
        )

        return generation_prompt

    def organize_plan_guide_thought_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        plan_chain: List[PlanNode],
        guide_plan_node: PlanNode,
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt I_G of the p-RAR paper."""

        root_prompt = str(chain_nodes[0].thought)

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_step_idx=True,
            with_evaluation_score=False,
        )
        # Create the prompt for the plan chain
        plan_prompt = self.organize_node_block_prompt(
            nodes=plan_chain[1:],
            content_attr="plan",
            head_format=self.plan_prompts.plan_head,
            start_flag=self.plan_prompts.plan_chain_start_flag,
            end_flag=self.plan_prompts.plan_chain_end_flag,
        )
        # Create the prompt for a guide prompt
        plan_guide_prompt = self.organize_node_block_prompt(
            nodes=[guide_plan_node],
            content_attr="plan",
            head_format=self.plan_prompts.plan_head,
            start_flag=self.plan_prompts.plan_start_flag,
            end_flag=self.plan_prompts.plan_start_flag,
        )

        # The chain only contain the first step
        if len(chain_nodes) == 1:

            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.plan_guide_first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                plan_guide_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                self.plan_start_flag
            )
            return generation_prompt

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.plan_guide_next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt, plan_prompt, plan_guide_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            self.plan_prompts.plan_chain_start_flag,
            guide_plan_node.step_idx,
            self.plan_prompts.plan_start_flag,
            len(chain_nodes),
        )

        return generation_prompt

    def organize_plan_exclusive_thought_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        plan_chain: List[PlanNode],
        plan_exclusion_candidates: List[PlanNode],
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt I_E of the p-RAR paper."""
        root_prompt = str(chain_nodes[0].thought)

        # Create the prompt for the plan chain
        plan_chain_prompt = ""
        if len(plan_chain) > 1:
            plan_chain_prompt = self.organize_node_block_prompt(
                nodes=plan_chain[1:],
                content_attr="plan",
                head_format=self.plan_prompts.plan_head,
                start_flag=self.plan_prompts.plan_chain_start_flag,
                end_flag=self.plan_prompts.plan_chain_end_flag,
            )

        plan_exclusion_prompt = self.organize_node_block_prompt(
            nodes=plan_exclusion_candidates,
            content_attr="plan",
            head_format=self.plan_prompts.plan_head,
            start_flag=self.plan_prompts.plan_exclusion_start_flag,
            end_flag=self.plan_prompts.plan_exclusion_end_flag,
        )

        # The chain only contain the first step
        if len(chain_nodes) == 1:
            # Generate as usual when nothing to be excluded
            if len(plan_exclusion_candidates) == 0:
                generation_prompt = BasicThoughtPromptFormat(
                    **self.generation_prompts.first_step_prompt
                )
                generation_prompt.head = generation_prompt.head.format(root_prompt)
                return generation_prompt

            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.exclusive_plan_first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                plan_exclusion_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                self.plan_prompts.plan_exclusion_start_flag
            )
            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_evaluation_score=False,
            with_step_idx=True,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.exclusive_plan_next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt, plan_chain_prompt, plan_exclusion_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            self.plan_prompts.plan_chain_start_flag,
            plan_exclusion_candidates[0].step_idx,
            self.plan_prompts.plan_exclusion_start_flag,
            len(chain_nodes),
        )

        return generation_prompt

    def organize_plan_summary_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        plan_chain: List[PlanNode],
        thought_node: base.BasicNode,
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt for the plan summary."""
        root_prompt = copy.deepcopy(chain_nodes[0].thought)
        root_prompt.head = ""
        root_prompt.notice = ""
        root_prompt.solution_flag = ""
        root_prompt.answer = ""
        root_prompt = str(root_prompt)

        # Create the prompt for the plan chain
        plan_chain_prompt = ""
        if len(plan_chain) > 1:
            plan_chain_prompt = self.organize_node_block_prompt(
                nodes=plan_chain[1:],
                content_attr="plan",
                head_format=self.plan_prompts.plan_head,
                start_flag=self.plan_prompts.plan_chain_start_flag,
                end_flag=self.plan_prompts.plan_chain_end_flag,
            )

        # Create the plan thought prompt
        plan_thought_prompt = self.organize_node_block_prompt(
            nodes=[thought_node],
            content_attr="thought",
            head_format=self.step_head,
            start_flag=self.plan_prompts.step_start_flag,
            end_flag=self.plan_prompts.step_end_flag,
        )

        # The chain only contain the first step
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.plan_prompts.first_plan_summarization_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                plan_thought_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                self.plan_prompts.step_start_flag
            )
            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_flag=True,
            with_evaluation_score=False,
            with_step_idx=True,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.plan_prompts.plan_summarization_prompt
        )
        generation_prompt.head = generation_prompt.head.format(
            root_prompt, thought_node.step_idx
        )
        generation_prompt.content = generation_prompt.content.format(
            chain_prompt, plan_chain_prompt, plan_thought_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            self.plan_prompts.plan_chain_start_flag,
            thought_node.step_idx,
            self.thought_chain_start_flag,
            thought_node.step_idx,
        )

        return generation_prompt

    def organize_plan_compare_prompt(
        self, plan_nodes: List[PlanNode], target_thought_plan_node: base.BasicNode
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt for comparing the plan."""
        # Create the prompt for the plan chain

        plan_pool_prompt = self.organize_node_block_prompt(
            nodes=plan_nodes,
            content_attr="plan",
            with_index=True,
            head_format=self.plan_prompts.plan_head,
            start_flag=self.plan_prompts.plan_comparison_start_flag,
            end_flag=self.plan_prompts.plan_comparison_end_flag,
        )
        # Create the prompt for the target plan
        target_plan_prompt = self.organize_node_block_prompt(
            nodes=[target_thought_plan_node],
            content_attr="thought",
            head_format=self.plan_prompts.plan_head,
            start_flag=self.plan_prompts.plan_start_flag,
            end_flag=self.plan_prompts.plan_start_flag,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.plan_prompts.compare_plan_prompt
        )

        generation_prompt.content = generation_prompt.content.format(
            plan_pool_prompt, target_plan_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            target_thought_plan_node.step_idx,
            self.plan_prompts.plan_start_flag,
            self.plan_prompts.plan_comparison_start_flag,
        )

        return generation_prompt

    def organize_plan_assessment_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        thought_node: base.BasicNode,
        thought_plan_node: base.BasicNode = None,
        plan_node: PlanNode = None,
    ) -> BasicThoughtPromptFormat:
        """
        Organizing the prompt for assessing the plan.
        :param thought_plan_node: The thought node that presents a plan.
        :param plan_node: The plan node that presents a plan.

        We can assess either the thought_plan_node or plan_node
        """
        root_prompt = copy.deepcopy(chain_nodes[0].thought)
        root_prompt.head = ""
        root_prompt.notice = ""
        root_prompt.solution_flag = ""
        root_prompt.answer = ""
        root_prompt = str(root_prompt)

        # Create the prompt for the plan thought
        assess_plan_prompt = None
        assess_plan_step_idx = None
        if thought_plan_node is not None:
            assess_plan_prompt = self.organize_node_block_prompt(
                nodes=[thought_plan_node],
                content_attr="thought",
                head_format=self.plan_prompts.plan_head,
                start_flag=self.plan_prompts.plan_assessment_start_flag,
                end_flag=self.plan_prompts.plan_assessment_end_flag,
            )
            assess_plan_step_idx = thought_plan_node.step_idx
        if plan_node is not None:
            assess_plan_prompt = self.organize_node_block_prompt(
                nodes=[plan_node],
                content_attr="plan",
                head_format=self.plan_prompts.plan_head,
                start_flag=self.plan_prompts.plan_assessment_start_flag,
                end_flag=self.plan_prompts.plan_assessment_end_flag,
            )
            assess_plan_step_idx = plan_node.step_idx
        assert assess_plan_prompt is not None

        # Create the prompt for the thought
        thought_prompt = self.organize_node_block_prompt(
            nodes=[thought_node],
            content_attr="thought",
            head_format=self.step_head,
            start_flag=self.plan_prompts.step_start_flag,
            end_flag=self.plan_prompts.step_end_flag,
        )

        # Create the prompt for the plan assessment of the first step
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.plan_prompts.assess_first_plan_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)
            generation_prompt.content = generation_prompt.content.format(
                thought_prompt, assess_plan_prompt
            )
            generation_prompt.target = generation_prompt.target.format(
                assess_plan_step_idx,
                self.plan_prompts.plan_assessment_start_flag,
                self.plan_prompts.step_start_flag,
                self.plan_prompts.plan_assessment_start_flag,
            )
            return generation_prompt

        # When there are multiple existing reasoning steps, create the prompt of the plan assessment of the next reasoning step

        thought_chain_prompt = self.organize_node_block_prompt(
            nodes=chain_nodes[1:],
            content_attr="thought",
            head_format=self.step_head,
            start_flag=self.thought_chain_start_flag,
            end_flag=self.thought_chain_end_flag,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.plan_prompts.assess_next_plan_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(
            thought_chain_prompt, thought_prompt, assess_plan_prompt
        )
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag,
            thought_node.step_idx,
            self.plan_prompts.step_start_flag,
            assess_plan_step_idx,
            self.plan_prompts.plan_assessment_start_flag,
            assess_plan_step_idx,
        )
        return generation_prompt
