"""
A model to perform the thought generation.
"""

import re
from typing import List, Tuple

import torch

from llmpebase.model import define_model
from llmpebase.model.prompting.thought_prompter import ThoughtStructurePrompter
from llmpebase.model.thought_structure.structure_generic import (
    BasicEvaluation,
    BasicSimilarity,
    BasicPromptAndResponse,
)
from llmpebase.model.thought_structure import base
from llmpebase.extractor.re_extraction import extract_solution


class LlmThoughtModel:
    """A thought model built upon the LLM model."""

    def __init__(
        self,
        prompter: ThoughtStructurePrompter,
        model_config: dict,
        llm_model: torch.nn.Module = None,
    ):
        super().__init__()
        if llm_model is None:
            llm_model = define_model(model_config)
        self.llm_model = llm_model
        self.prompter = prompter

        self.model_config = model_config

    def generate_thoughts(
        self, thought_chain: List[base.BasicNode], num_thoughts: int = 1, **kwargs
    ) -> Tuple[List[str], List[BasicPromptAndResponse]]:
        """Generate the thoughts based on the thought chain."""
        prompt = self.prompter.organize_next_thought_prompt(
            chain_nodes=thought_chain, **kwargs
        )

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=num_thoughts,
            sys_prompt=self.prompter.generation_system_prompt,
        )
        thoughts = self.llm_model.read_response_contents(responses)

        reasoning = [
            BasicPromptAndResponse(prompt=prompt, response=thought)
            for thought in thoughts
        ]

        return thoughts, reasoning

    def evaluate_thoughts(
        self,
        thoughts: List[str],
        thought_chain: List[base.BasicNode],
        n_request: int = 1,
    ) -> List[BasicEvaluation]:
        """Evaluate the thoughts based on the thought chain."""
        evaluations = []
        for thought in thoughts:
            prompt = self.prompter.organize_evaluation_prompt(
                thought=thought, chain_nodes=thought_chain
            )
            responses = self.llm_model.forward(
                user_prompt=str(prompt),
                per_request_responses=n_request,
                sys_prompt=self.prompter.generation_system_prompt,
            )

            responses = self.llm_model.read_response_contents(responses)

            eval_flag = self.prompter.evaluation_prompts.score_flag
            score_contents = [
                extract_solution(response, eval_flag) for response in responses
            ]

            scores = [
                re.findall(r"\b\d+(?:\.\d+)?\b", score_content)
                for score_content in score_contents
            ]
            scores = [0.5 if len(score) == 0 else float(score[0]) for score in scores]

            contents = [
                response.split(self.prompter.evaluation_prompts.score_flag)[0]
                for response in responses
            ]

            evaluations.append(
                BasicEvaluation(
                    evaluation_prompt=prompt,
                    evaluation_scores=scores,
                    evaluation_contents=contents,
                    evaluation_outputs=responses,
                )
            )

        return evaluations

    def measure_thought_similarity(
        self,
        thought_a: str,
        thought_b: str,
        thought_chain: List[base.BasicNode],
        n_request: int = 1,
    ) -> BasicSimilarity:
        """Measure the similarity between two thoughts."""

        prompt = self.prompter.organize_similarity_prompt(
            thought_a=thought_a,
            thought_b=thought_b,
            chain_nodes=thought_chain,
        )

        responses = self.llm_model.forward(
            user_prompt=str(prompt),
            per_request_responses=n_request,
            sys_prompt=self.prompter.similarity_system_prompt,
        )

        responses = self.llm_model.read_response_contents(responses)
        similarity_flag = self.prompter.similarity_prompts.score_flag
        score_contents = [
            extract_solution(response, similarity_flag) for response in responses
        ]
        scores = [
            re.findall(r"\b\d+(?:\.\d+)?\b", score_content)
            for score_content in score_contents
        ]
        scores = [0.5 if len(score) == 0 else float(score[0]) for score in scores]

        contents = [
            response.split(self.prompter.similarity_prompts.score_flag)[0]
            for response in responses
        ]

        return BasicSimilarity(
            similarity_prompt=prompt,
            similarity_scores=scores,
            similarity_contents=contents,
            similarity_outputs=responses,
        )
