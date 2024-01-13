"""
A prompter to organize and generate the prompts for the thought structure with 
the Thought Rollback.
"""

from typing import List

import data_prompts

from llmpebase.model.prompting import thought_prompt

from llmpebase.model.thought_structure import base
from llmpebase.model.prompting.prompt_generic import (
    BasicSamplePrompt,
    BasicAnswerPromptFormat,
    BasicThoughtPromptFormat,
)

from llmpebase.config import Config


class TRStructurePrompt(thought_prompt.ThoughtStructurePrompt):
    """A prompt to support the rollback in thought structure with the Thought Rollback."""

    # The flag to indicate the block of the experience
    # obtained before rolling back
    analysis_flag = "#" * 20

    # The system prompt for the generation of the thought rollback
    reasoning_analysis_system_prompt = """You are a mathematician specializing in checking and analyzing the reasoning process containing multiple intermediate reasoning steps proposed to address a math question. Please check the correctness of the overall reasoning logic and each reasoning step regarding mathematical logic and rationality."""

    rollback_system_prompt = """You are a mathematician specializing in identifying the unnecessary, wrong, illogical, unreasonable, or vague reasoning steps based on the given analysis of a reasoning process. You should summarize the analysis to give the index of these bad steps."""

    rollback_solution_flag = "Bad step index:"
    intermediate_analysis_prompt_format = data_prompts.rollback_prompt_formats[
        Config().data.data_name
    ]["Intermediate"]
    sink_analysis_prompt_format = data_prompts.rollback_prompt_formats[
        Config().data.data_name
    ]["Sink"]

    rollback_controller_prompt_prompt = data_prompts.rollback_controller_prompt_format

    rollback_analysis_prompt = None
    rollback_controller_prompt = None

    def add_experience_prompt(self, root_prompt: BasicSamplePrompt):
        """Add the experience to the prompt."""
        root_prompt.question

    def organize_next_thought_prompt(self, chain_nodes: List[base.BasicNode]):
        """Generating the prompt for next thought."""
        # root_prompt = self.add_experience_prompt(chain_nodes[0].thought)

        root_prompt = str(chain_nodes[0].thought)
        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:], with_flag=True, with_evaluation_score=False
        )

        generation_prompt = BasicThoughtPromptFormat(**self.generation_prompt)
        generation_prompt.head = generation_prompt.head.format(root_prompt)
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
