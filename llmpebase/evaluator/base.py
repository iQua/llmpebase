""" A base evaluator to evaluate the performance of the model. """


from typing import List, Union

from llmpebase.model.LM.base import BaseLlmRequest


class BaseEvaluator:
    """A base evaluator, built upon inner functions of python, to measure the similarity between
    the result and the groundtruth."""

    def measure(
        self, result: Union[int, float], groundtruth: Union[int, float]
    ) -> bool:
        """Measure the similarity between the result and the groundtruth."""
        raise NotImplementedError

    def forward(self, results: List[str], groundtruths: List[str]):
        """Evaluate the result by the groundtruth."""
        assert isinstance(results, list)
        assert isinstance(groundtruths, list)
        measurements: dict = {"num_correct": 0, "matches": []}
        for res, gt in zip(results, groundtruths):
            res = self.measure(res, gt)
            measurements["matches"].append(res)
            measurements["num_correct"] += int(res)

        return measurements


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
