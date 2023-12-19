"""
A model to perform the thought generation.
"""
import re
from typing import List

import torch

from llmpebase.model import define_model
from llmpebase.model.prompting.thought_prompt import ThoughtStructurePrompt
from llmpebase.model.thought_structure import base


class LlmThoughtModel:
    """A thought model built upon the LLM model."""

    def __init__(
        self,
        llm_model: torch.nn.Module = None,
        model_config: dict = None,
        prompter: ThoughtStructurePrompt = None,
    ):
        super().__init__()
        if llm_model is None:
            llm_model = define_model(model_config)
        self.llm_model = llm_model
        self.prompter = ThoughtStructurePrompt() if prompter is None else prompter

    def generate_thoughts(
        self, thought_chain: List[base.BasicNode], num_thoughts: int = 1
    ):
        """Generate the thoughts based on the thought chain."""
        prompt = self.prompter.organize_next_thought_prompt(chain_nodes=thought_chain)

        responses = self.llm_model.forward(
            user_prompt=prompt,
            per_request_responses=num_thoughts,
            sys_prompt=self.prompter.generation_system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        return thoughts, prompt

    def evaluate_thoughts(
        self,
        thoughts: List[str],
        thought_chain: List[base.BasicNode],
    ):
        """Evaluate the thoughts based on the thought chain."""
        evaluations = []
        for thought in thoughts:
            prompt = self.prompter.organize_evaluation_prompt(
                thought=thought, chain_nodes=thought_chain
            )

            responses = self.llm_model.forward(
                user_prompt=prompt,
                per_request_responses=1,
                sys_prompt=self.prompter.generation_system_prompt,
            )

            content = self.llm_model.read_response_contents(responses)[0]

            scores = re.findall(r"\b\d+(?:\.\d+)?\b", content)
            score = 1.0 if len(scores) == 0 else float(scores[0])

            evaluations.append(score)

        return evaluations, prompt

    def measure_thought_similarity(
        self,
        thought_a: str,
        thought_b: str,
        thought_chain: List[base.BasicNode],
    ):
        """Measure the similarity between two thoughts."""

        prompt = self.prompter.organize_similarity_prompt(
            thought_a=thought_a,
            thought_b=thought_b,
            chain_nodes=thought_chain,
        )

        responses = self.llm_model.forward(
            user_prompt=prompt,
            per_request_responses=1,
            sys_prompt=self.prompter.similarity_system_prompt,
        )

        content = self.llm_model.read_response_contents(responses)[0]
        scores = re.findall(r"\b\d+(?:\.\d+)?\b", content)

        return 0 if len(scores) == 0 else float(scores[0]), prompt
