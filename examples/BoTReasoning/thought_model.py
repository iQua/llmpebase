"""
The thought model of the BoT to build the thought structure.    
"""

import torch

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.prompting.thought_prompter import ThoughtStructurePrompter
from llmpebase.model.thought_structure import thought_model


class BoTThoughtModel(thought_model.LlmThoughtModel):
    """
    A thought model to organize and utilize the experiences in the
    thought structure building.
    """

    def __init__(
        self,
        llm_model: torch.nn.Module = None,
        model_config: dict = None,
        prompter: ThoughtStructurePrompter = None,
    ):
        super().__init__(llm_model, model_config, prompter)
        # A container to collect experiences
        self.experience_container = []

    def memorize_experience(self, solution_str: str, feedback: str):
        """
        Collect the experience from the feedback, which contains error reports
        and detailed advice on how to revise previously generated reasoning steps.
        """
        if len(feedback.strip()) != 0:
            self.experience_container.append((solution_str, feedback))

    def add_experience(self, prompt_sample: BasicSamplePrompt):
        """Add the experience to the sample prompt."""

        return self.prompter.organize_root_prompt(
            prompt_sample=prompt_sample, experiences=self.experience_container
        )

    def clean_experience(self):
        """Clean the experience container."""
        self.experience_container = []
