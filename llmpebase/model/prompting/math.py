"""
The implementation of different prompts.
"""
import os
import random
import glob

from llmpebase.model.prompting import base


class MATHStandardPrompting(base.BasePrompting):
    """The standard prompting for MATH."""

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the MATH dataset."""

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


class MATHCoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of MATH."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def __init__(
        self, model_config: dict, cot_filepath: str = None, cot_filename: str = None
    ) -> None:
        super().__init__(model_config, cot_filepath)

        self.cot_filename = (
            cot_filename if cot_filename is not None else model_config["cot_filename"]
        )

    def load_cot(self, cot_filepath: str):
        """Load the cot examples from the file.

        Load the cot as a dictionary, with the structure:
         - problem_name:
            - prompt: filepath

        """
        folders = glob.glob(cot_filepath + "/*")
        for folder_path in folders:
            folder_name = os.path.basename(folder_path)
            prompt_files = glob.glob(folder_path + "/*")
            self.cot_prompt = {
                folder_name: {
                    os.path.basename(file_path): file_path for file_path in prompt_files
                }
            }

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt.
        The problem name of CoT will be moral_scenarios.
        When the format problem name of llmpebase is Moral Scenarios,
        we need to make a conversion.
        """
        problem_name = problem_name.lower()
        return self.cot_prompt[problem_name][self.cot_filename]


class MATHZeroShotCoTPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of MATH."""
