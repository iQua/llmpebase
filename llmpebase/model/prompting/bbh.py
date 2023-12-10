"""
The implementation of different prompts for BBH.
"""
import os
import glob
import random

from llmpebase.model.prompting import base
from llmpebase.utils import tools


class BBHStandardPrompting(base.BasePrompting):
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
        samples = [dataset[idx] for idx in fewshot_indexes]
        return (
            self.create_test_prompt(
                problem_name=problem_name,
                template_samples=samples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )


class BBHCoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of BBH."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def load_cot(self, cot_filepath: str):
        """Load the cot examples from the file."""
        cot_files = glob.glob(cot_filepath)
        self.cot_prompts = {
            tools.format_term(os.path.basename(path)): path for path in cot_files
        }

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt."""
        prompt_path = self.cot_prompts[problem_name]
        with open(prompt_path, "r", encoding="utf-8") as f:
            cot_prompt = f.read()
        return cot_prompt


class BBHZeroShotCoTPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of BBH."""
