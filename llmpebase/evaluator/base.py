""" A base evaluator to evaluate the performance of the model. """

import os
import json
from typing import List, Union
from collections import defaultdict

from llmpebase.model.LM.base import BaseLlmRequest


class BaseEvaluator:
    """A base evaluator, built upon inner functions of python, to measure the similarity between
    the result and the groundtruth."""

    def __init__(self, filename: str = None, save_path: str = None):
        self.measurements: dict = {"num_correct": 0, "matches": []}

        filename = "evaluation.json" if filename is None else filename
        save_path = os.getcwd() if save_path is None else save_path
        os.makedirs(save_path, exist_ok=True)

        self.eval_path = os.path.join(save_path, filename)

    def measure(
        self, result: Union[int, float], groundtruth: Union[int, float]
    ) -> bool:
        """Measure the similarity between the result and the groundtruth."""
        raise NotImplementedError

    def forward(self, results: List[str], groundtruths: List[str]):
        """Evaluate the result by the groundtruth."""

        for res, gt in zip(results, groundtruths):
            res = self.measure(res, gt)
            self.measurements["matches"].append(res)
            self.measurements["num_correct"] += int(res)

    def save_measures(self):
        """Save the measures to the disk."""
        with open(self.eval_path, "w", encoding="utf-8") as file:
            json.dump(self.measurements, file, indent=4)


class BaseLLMEvaluator(BaseEvaluator):
    """A base evaluator, built upon the LLM, to measure the similarity between the result
    and the groundtruth."""

    def __init__(self, llm_model: BaseLlmRequest):
        super().__init__()
        # Define the request model used as the extractor
        self.llm_model = llm_model

    def measure(
        self, result: Union[int, float], groundtruth: Union[int, float]
    ) -> bool:
        """Measure the similarity between the result and the groundtruth."""
        raise NotImplementedError
