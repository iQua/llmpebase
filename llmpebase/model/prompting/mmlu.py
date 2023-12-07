"""
The implementation of different prompts for the MMLU dataset.
"""
import random

from llmpebase.model.prompting import base


class MMLUStandardPrompting(base.BasePrompting):
    """The standard prompt of MMLU."""

    solution_flag: str = "The final choice is"

    question_prompt_tail: str = "\nWhich of the following choices is correct?"

    def organize_question_prompt(self, sample: dict):
        prompt = super().organize_question_prompt(sample)
        options = sample.auxiliary["option_str"]
        return f"""{prompt}{options}"""

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the MMLU dataset."""

        n_shots = config["n_shots"]

        problem_name = sample.auxiliary["sample_problem"]
        sample_indexes = dataset.get_problem_sample_indexes(problem_name)
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


class MMLUCoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of MMLU."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def organize_question_prompt(self, sample: dict):
        prompt = super().organize_question_prompt(sample)
        options = sample.auxiliary["option_str"]
        return f"""{prompt}{options}"""

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt.
        The problem name of CoT will be moral_scenarios.
        When the format problem name of llmpebase is Moral Scenarios,
        we need to make a conversion.
        """
        problem_name = problem_name.replace(" ", "_").lower()
        return self.cot_prompt[problem_name]


class MMLUZeroShotCoTPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of MMLU."""

    solution_flag: str = "The final choice is"

    question_prompt_tail: str = "\nWhich of the following choices is correct?"

    def organize_question_prompt(self, sample: dict):
        prompt = super().organize_question_prompt(sample)
        options = sample.auxiliary["option_str"]
        return f"""{prompt}{options}"""
