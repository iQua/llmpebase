"""
Implementation of commenter of BoT to evaluate the thought chain and provide feedback.
"""

import torch

from comment_prompter import BoTCommentPrompter

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model import define_model


class BoTCommenter:
    """A commenter to analyse a reasoning chain to produce a feedback containing
    error analysis and advice."""

    def __init__(
        self,
        llm_model: torch.nn.Module = None,
        model_config: dict = None,
        prompter: BoTCommentPrompter = None,
    ) -> None:
        if llm_model is None:
            llm_config = model_config["commenter"]
            llm_model = define_model(llm_config)

        self.llm_model = llm_model
        self.prompter = prompter

    def comment_reasoning_chain(
        self, prompt_sample: BasicSamplePrompt, reasoning_chain_prompt: str
    ):
        """Get the feedback of the thought chain from the LLMs."""

        feedback_prompt = self.prompter.organize_chain_feedback_prompt(
            prompt_sample, reasoning_chain_prompt
        )

        # Forward the generation model to get responses
        responses = self.llm_model.forward(
            user_prompt=str(feedback_prompt),
            per_request_responses=3,
            sys_prompt=self.prompter.system_prompt,
        )
        response_contents = self.llm_model.read_response_contents(responses)
        # Extract the longest response as the feedback
        feedback = max(response_contents, key=len)

        return feedback
