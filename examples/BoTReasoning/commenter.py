"""
Implementation of commenter of BoT to evaluate the thought chain and provide feedback.
"""

import json
import torch

from comment_prompter import BoTCommentPrompter

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model import define_model


class BoTCommenter:
    """
    A commenter to analyse a reasoning chain to produce a feedback containing
    error analysis and advice.
    """

    def __init__(
        self,
        llm_model: torch.nn.Module = None,
        logging_config: dict = None,
        model_config: dict = None,
        prompter: BoTCommentPrompter = None,
    ) -> None:

        self.comment_config = model_config["bot_settings"]["commenter"]

        if llm_model is None:
            llm_config = self.comment_config
            llm_model = define_model(llm_config)

        self.llm_model = llm_model
        self.prompter = prompter
        self.save_path = logging_config["result_path"]

        self.comment_state = {}

    def comment_reasoning_chain(
        self, prompt_sample: BasicSamplePrompt, reasoning_chain_prompt: str
    ):
        """Get the feedback of the thought chain from the LLMs."""

        feedback_prompt = self.prompter.organize_chain_feedback_prompt(
            prompt_sample, reasoning_chain_prompt
        )

        # Set the generation config for the llm
        self.llm_model.generation_config.update(
            self.comment_config["generation_settings"]
        )

        # Forward the generation model to get responses
        responses = self.llm_model.forward(
            user_prompt=str(feedback_prompt),
            per_request_responses=1,
            sys_prompt=self.prompter.system_prompts.comment_prompt,
        )
        response_contents = self.llm_model.read_response_contents(responses)
        # Extract the longest response as the feedback
        feedback = max(response_contents, key=len)

        self.comment_state["reasoning_chain"] = reasoning_chain_prompt
        self.comment_state["feedback_prompt"] = feedback_prompt
        self.comment_state["generation_config"] = self.llm_model.generation_config
        self.comment_state["feedback"] = feedback

        return feedback

    def save_state(
        self,
        location: str,
        file_name: str,
    ):
        """Save the state of the commenter."""
        location = f"{self.save_path}/{location}"
        with open(f"{location}/{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(self.comment_state, f)
