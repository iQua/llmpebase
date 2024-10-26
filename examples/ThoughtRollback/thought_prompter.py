"""
A prompter to organize and generate the prompts for the thought structure with 
the Thought Rollback.
"""

from typing import List

import tr_system_prompts
import tr_thought_prompts

from llmpebase.model.prompting import thought_prompter

from llmpebase.model.thought_structure import base
from llmpebase.prompt.generic import (
    BasicSamplePrompt,
    BasicPromptFormat,
    BasicAnswerPromptFormat,
    BasicThoughtPromptFormat,
)


class TRStructurePrompt(thought_prompter.ThoughtStructurePrompter):
    """A prompt to support the rollback in thought structure with the Thought Rollback."""

    rollback_analysis_prompt = None
    rollback_controller_prompt = None

    # We present the init function here show what system and thought prompts
    # are required.
    def __init__(
        self,
        system_prompts: tr_system_prompts.RollbackSystemPrompts,
        thought_prompts: tr_thought_prompts.BaseRollbackThoughtPrompts,
        rollback_prompts: tr_thought_prompts.RollbackPrompts,
    ):
        super().__init__(system_prompts, thought_prompts)

        self.rollback_prompts = rollback_prompts

    def organize_experience_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        thought_edges: List[base.BasicEdge] = None,
        rollback_state: dict = None,
    ):
        """Add the experience to the root as the demonstrations.
        :param root_prompt: The root prompt of the thought structure.
        :param thought_edges: A list containing the edges of the chain of thoughts.
        :param rollback_state: The rollback's state which contains
            - node: .
            - rollback_edge: A string with format "{src_node_id}->{dst_node_id}_R{n_rollbacks+1}".
        """
        cur_node = chain_nodes[-1]
        # Experience has two sources:
        # For a reasoning path, 1 -> 2 -> 3 -> 4 -> 5
        # 1. Experience may derive from the edges which maintains the experience from one rollback.
        #    For instance, for one rollback from 9->2, the experience of it will be maintained in
        #    the edge 2->3. When generating the step 6, the experience of the rollback should be included.
        # 2. Experience may direct derive from the rollback when one node rolls back to node 5, the
        #   experience should be included.
        #    For instance, for one rollback from 10->5, the generation of 5->6 should include the
        #    experience of the rollback from 10->5.
        #   the rollbacks should be added to the root prompt to facilitate the subsequent reasoning.
        # Case 1: Get experiences from existing edges
        experiences = []
        for edge in thought_edges:
            rollback_info = edge.auxiliary.get("RollbackExperience", None)
            if rollback_info is not None:
                analysis_steps = rollback_info["AnalysisSteps"]
                analysis = rollback_info["RollbackAnalysis"]

                experiences.append((analysis_steps, analysis))

        # Case 2
        # The rollback_state has values, meaning that current reasoning step generation
        # is caused by a rollback that other node rolls back to the current node.
        if rollback_state is not None:
            # Get the node and the rollback edge
            rollback_node = rollback_state["node"]
            rollback_edge = rollback_state["rollback_edge"]
            # Ensure that the node is the one that is rolled back to
            if rollback_node.identity == cur_node.identity:
                # Get the experience from the under-working rollback
                # Thus, this is the new experience
                # Get the experience when the node contains it
                # When 'do_experience_rollback' is not set in the config,
                # each node will not record the experience.
                if rollback_edge in rollback_node.auxiliary:
                    new_experience = rollback_node.auxiliary[rollback_edge]
                    experiences.append(
                        (
                            new_experience["AnalysisSteps"],
                            new_experience["RollbackAnalysis"],
                        )
                    )

        # Organize the obtained experiences as a string
        # Once there is no experience, return the original root prompt
        if len(experiences) == 0:
            return chain_nodes[0].thought

        # Otherwise, add the experience as demonstrations to the root prompt
        experience_prompt = [
            f"{self.rollback_prompts.experience_start_flag.format(idx)}\n{exp[0]}\n\nAnalysis:{exp[1]}\n{self.analysis_flag}\n"
            for idx, exp in enumerate(experiences)
        ]
        experience_prompt = "\n".join(experience_prompt)
        # Add the experience to the demonstrations of the root prompt
        experience_demos = BasicPromptFormat(**self.rollback_experience_prompt_format)
        experience_demos.content = experience_demos.content.format(experience_prompt)

        temp_root_prompt = BasicSamplePrompt(**chain_nodes[0].thought)
        # Create a new demonstration block to avoid modifying the original one
        temp_root_prompt.demonstrations = BasicPromptFormat(
            content="", head="", notice="", tail="", prompt=""
        )

        cur_content = temp_root_prompt.demonstrations.content
        temp_root_prompt.demonstrations.content = f"{cur_content}\n{experience_demos}"

        return temp_root_prompt

    def organize_next_thought_prompt(self, chain_nodes: List[base.BasicNode], **kwargs):
        """Generating the prompt for next thought."""

        chain_edges = kwargs.get("thought_edges", None)
        rollback_state = kwargs.get("rollback_state", None)

        temp_root_prompt = self.organize_experience_prompt(
            chain_nodes,
            thought_edges=chain_edges,
            rollback_state=rollback_state,
        )

        root_prompt_str = str(temp_root_prompt)
        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:], with_flag=True, with_evaluation_score=False
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.generation_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt_str)
        generation_prompt.content = generation_prompt.content.format(chain_prompt)
        generation_prompt.target = generation_prompt.target.format(self.thought_flag)

        return generation_prompt

    def organize_reasoning_analysis_prompt(self, chain_nodes: List[base.BasicNode]):
        """Organize the prompt for rollback."""

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_step_idx=True,
            with_flag=True,
            with_evaluation_score=False,
        )

        # Adjust the root prompt for the rollback analyzing
        # Get the root prompt
        root_prompt = chain_nodes[0].thought

        # Get the prompt to analyze the reasoning steps
        temp_prompt = BasicSamplePrompt(**root_prompt)
        temp_prompt.answer = BasicAnswerPromptFormat(
            content="", head="", notice="", tail="", groundtruth="", prompt=""
        )
        temp_prompt.notice = ""
        temp_prompt.head = temp_prompt.head.replace(
            "Answer", "Analyze the reasoning steps proposed for"
        )

        # Get the prompt format based on the final node type
        final_node = chain_nodes[-1]
        analysis_prompt_format = (
            self.intermediate_analysis_prompt_format
            if final_node.position == "Intermediate"
            else self.sink_analysis_prompt_format
        )

        analysis_prompt = BasicThoughtPromptFormat(**analysis_prompt_format)
        analysis_prompt.head = analysis_prompt.head.format(
            temp_prompt, len(chain_nodes) - 1
        )
        analysis_prompt.content = analysis_prompt.content.format(chain_prompt)
        if final_node.position == "Intermediate":
            analysis_prompt.target = analysis_prompt.target.format(self.thought_flag)
        else:
            question = root_prompt.question.content.split(".")[-1]
            analysis_prompt.target = analysis_prompt.target.format(
                self.thought_flag, question
            )

        # Record the rollback prompt
        self.rollback_analysis_prompt = analysis_prompt
        return analysis_prompt

    def organize_prompt_controller_prompt(
        self, chain_nodes: List[base.BasicNode], reasoning_analysis: str
    ):
        """Organize the prompt for the controller to control the rollback."""

        # Adjust the root prompt for the rollback analyzing
        # Get the root prompt
        root_prompt = chain_nodes[0].thought

        # Get the prompt to identity bad reasoning steps
        temp_prompt = BasicSamplePrompt(**root_prompt)
        temp_prompt.answer = BasicAnswerPromptFormat(
            content="", head="", notice="", tail="", groundtruth="", prompt=""
        )
        temp_prompt.notice = ""
        temp_prompt.head = temp_prompt.head.replace(
            "Answer", "Identity bad reasoning steps proposed for"
        )

        controller_prompt_format = self.rollback_controller_prompt_prompt

        controller_prompt = BasicThoughtPromptFormat(**controller_prompt_format)
        controller_prompt.head = controller_prompt.head.format(
            temp_prompt, len(chain_nodes) - 1
        )
        controller_prompt.content = controller_prompt.content.format(
            self.analysis_flag, reasoning_analysis, self.analysis_flag
        )
        controller_prompt.target = controller_prompt.target.format(
            self.analysis_flag, len(chain_nodes) - 1, self.rollback_solution_flag
        )

        # Record the rollback prompt
        self.rollback_controller_prompt = controller_prompt
        return controller_prompt
