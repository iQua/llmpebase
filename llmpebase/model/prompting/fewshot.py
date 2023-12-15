"""
Implementations of Few-shot prompting, which is the standard prompt engineering.
"""

import random

from llmpebase.model.prompting import base


class BBHFewShotPrompting(base.BasePrompting):
    """The standard prompt of BBH."""

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the BBH dataset."""

        n_shots = config["n_shots"]

        problem_name = sample.auxiliary["sample_problem"]
        sample_idx = sample.auxiliary["sample_idx"]
        sample_indexes = dataset.get_problem_sample_indexes(problem_name)
        # Remove the test sample index to avoid including this test sample
        # in the prompt
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


class MATHFewShotPrompting(base.BasePrompting):
    """The standard prompting for MATH."""

    def create_prompt_sample(self, sample, dataset, config):
        """Create the prompt sample from the sample of MATH dataset."""

        n_shots = config["n_shots"]

        problem_name = sample.auxiliary["sample_problem"]
        sample_indexes = dataset.get_problem_sample_indexes(problem_name)
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


class MMLUFewShotPrompting(base.BasePrompting):
    """The standard prompt of MMLU."""

    solution_flag: str = "The final choice is"

    def create_prompt_sample(self, sample, dataset, config):
        """Create the prompt sample from the sample of MMLU dataset."""

        n_shots = config["n_shots"]

        problem_name = sample.auxiliary["sample_problem"]
        sample_indexes = dataset.get_problem_sample_indexes(problem_name)
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


class TheoremQAFewShotPrompting(base.BasePrompting):
    """The standard prompt of TheoremQA."""

    solution_flag: str = "The answer is"

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the TheoremQA dataset."""

        n_shots = config["n_shots"]

        problem_name = sample.auxiliary["problem_subfield"]
        sample_idx = sample.auxiliary["sample_idx"]
        sample_indexes = dataset.get_problem_sample_indexes(problem_name)
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
