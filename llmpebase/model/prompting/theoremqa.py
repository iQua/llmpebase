"""
The implementation of different prompts.
"""

import random

from llmpebase.model.prompting import base


class TheoremQAStandardPrompting(base.BasePrompting):
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
        samples = [dataset[idx] for idx in fewshot_indexes]
        return (
            self.create_test_prompt(
                problem_name=problem_name,
                template_samples=samples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )


class TheoremQACoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of TheoremQA."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The final solution is "

    def load_cot_prompt(self, problem_name: str):
        """Load the cot prompt."""
        problem_name = problem_name.replace(" ", "_")
        return self.cot_prompt[problem_name]


class TheoremQAZeroShotCoTPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of TheoremQA."""

    solution_flag: str = "The final solution is"
