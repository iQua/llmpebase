"""
Implementations of Few-shot prompting, which is the fewshot prompt engineering.
"""

import random

from llmpebase.model.prompting import base


class ProblemFewShotPrompting(base.BasePrompting):
    """The fewshot prompt for the specific problem"""

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the BBH dataset."""

        n_shots = config["n_shots"]

        problem_name = sample["auxiliary"]["sample_problem"]
        if "problem_subfield" in sample["auxiliary"]:
            problem_name = sample["auxiliary"]["problem_subfield"]

        sample_indexes = dataset.get_problem_sample_indexes(problem_name)
        # Remove the test sample index to avoid including this test sample
        # in the prompt
        # This is generally needed when the dataset and the sample are from
        # the same set
        if "sample_idx" in sample["auxiliary"]:
            sample_idx = sample["auxiliary"]["sample_idx"]
            if sample_idx in sample_indexes:
                sample_indexes.remove(sample_idx)

        fewshot_indexes = (
            random.sample(sample_indexes, n_shots)
            if len(sample_indexes) > n_shots
            else sample_indexes
        )
        examples = [dataset[idx] for idx in fewshot_indexes]
        return (
            self.create_test_prompt(
                problem_name=problem_name,
                demonstrations=examples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )


class MMLUFewShotPrompting(ProblemFewShotPrompting):
    """The fewshot prompt of MMLU."""

    solution_flag: str = "The final choice is"


class AQUAFewShotPrompting(ProblemFewShotPrompting):
    """The fewshot prompt of AQUA."""

    solution_flag: str = "The final choice is"


class TheoremQAFewShotPrompting(ProblemFewShotPrompting):
    """The fewshot prompt of TheoremQA."""

    solution_flag: str = "The answer is"
