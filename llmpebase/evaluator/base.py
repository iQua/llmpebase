""" A base evaluator to evaluate the performance of the model. """


from typing import List, Union

from llmpebase.model.LM.base import BaseLMRequest


class BaseEvaluator:
    """A base evaluator, built upon inner functions of python, to measure the similarity between
    the result and the groundtruth."""

    def __init__(self):
        # Track the matching between results and groundtruths
        self.matches = []

        # Set num correct answers
        self.num_correct = 0

    def measure(
        self, result: Union[int, float], groundtruth: Union[int, float]
    ) -> bool:
        """Measure the similarity between the result and the groundtruth."""
        raise NotImplementedError

    def forward(self, results: List[str], groundtruths: List[str]):
        """Evaluate the result by the groundtruth."""

        for res, gt in zip(results, groundtruths):
            res = self.measure(res, gt)
            self.matches.append(res)
            self.num_correct += int(res)


class BaseLLMEvaluator(BaseEvaluator):
    """A base evaluator, built upon the LLM, to measure the similarity between the result
    and the groundtruth."""

    def __init__(self, llm_model: BaseLMRequest):
        super().__init__()
        # Define the request model used as the extractor
        self.llm_model = llm_model

    def measure(
        self, result: Union[int, float], groundtruth: Union[int, float]
    ) -> bool:
        """Measure the similarity between the result and the groundtruth."""
        raise NotImplementedError
