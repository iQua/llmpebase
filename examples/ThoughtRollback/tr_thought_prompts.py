"""
Thought prompts used by the thought rollback framework.
"""

from llmpebase.prompt.thought_prompt import BaseThoughtPrompts, ThoughtGenerationPrompts
from llmpebase.prompt.generic import (
    BasicPromptFormat,
    BasicThoughtPromptFormat,
)


class RollbackPrompts:
    """
    Prompts related to the rollback controller used by the thought rollback framework.
    """

    # A format to organize the experiences from rollbacks as the demonstrations
    # to be included in the root prompt
    experience_block_start_flag = "<Experiences>"
    experience_block_end_flag = "<\\Experiences>"

    experience_start_flag = "<{}-th Experience>"
    experience_end_flag = "<\\{}-th Experience>"

    # The flag to indicate the block of the experience
    # obtained before rolling back
    analysis_start_flag: str = "<Reasoning Analysis>"
    analysis_end_flag: str = "<\\Reasoning Analysis>"

    rollback_solution_flag: str = "Bad step index:"

    reasoning_analysis_prompt_format = BasicThoughtPromptFormat(
        head="{}Toward addressing the given question, below is a reasoning process containing {} already taken steps:\n",
        content="\n\n{}\n{}\n{}\n\n",
        target="""Please review the reasoning process within {}, then directly generate the error report and output the indexes of the identified mistaken steps after '{}'.\n""",
        notice="Output empty string when no steps are given. Do not repeat the reasoning step but use step idx as reference. Output only the error analysis.\n",
        tail="",
        prompt="",
    )


class RollbackThoughtGenerationPrompts(ThoughtGenerationPrompts):
    """
    A base class to organize the prompt with the plan for the thought generation.
    """

    rollback_experience_prompt_format = BasicPromptFormat(
        head="Experience containing previously made mistakes:\n",
        content="\n{}\n",
        notice="",
        tail="Consider the analysis in the above experience to avoid making similar mistakes during reasoning for the question.\n\n",
        prompt="",
    )


class BaseRollbackThoughtPrompts(BaseThoughtPrompts):
    """A base class to organize the plan-based thought prompts"""

    generation: RollbackThoughtGenerationPrompts = RollbackThoughtGenerationPrompts()
