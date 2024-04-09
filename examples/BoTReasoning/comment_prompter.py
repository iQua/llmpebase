"""
The prompter to organize the comment prompts for the Boosting of Thoughts (BoT).
"""

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.prompt.generic import BasicThoughtPromptFormat
from llmpebase.prompt import BaseSystemPrompts, ChainOutcomeCommentPrompts


class BoTCommentPrompter:
    """
    A thought comment prompter to organize the thought prompts for the Boosting of Thoughts (BoT).
    """

    def __init__(
        self,
        system_prompts: BaseSystemPrompts,
        comment_prompts: ChainOutcomeCommentPrompts,
    ):
        self.system_prompts = system_prompts
        self.comment_prompts = comment_prompts

    def organize_chain_feedback_prompt(
        self,
        prompt_sample: BasicSamplePrompt,
        reasoning_chain_prompt: str,
    ):
        """Organize the prompt for the chain feedback."""

        feedback_prompt = BasicThoughtPromptFormat(
            **self.comment_prompts.feedback_prompt
        )
        feedback_prompt.head = feedback_prompt.head.format(
            prompt_sample.question,
            self.comment_prompts.chain_start_flag,
            reasoning_chain_prompt,
            self.comment_prompts.chain_end_flag,
        )
        feedback_prompt.head = feedback_prompt.content.format(
            self.comment_prompts.chain_start_flag, self.comment_prompts.chain_end_flag
        )

        return feedback_prompt
