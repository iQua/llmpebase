"""
Implementations of Chain Of Thought (CoT) prompting
"""
import os
import glob

from llmpebase.model.prompting import base
from llmpebase.utils import tools


class BBHCoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of BBH."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def load_cot(self, cot_filepath: str):
        """Load the cot examples from the file."""
        cot_files = glob.glob(cot_filepath)
        self.cot_prompts = {
            tools.format_term(os.path.basename(path).split(".")[0]): path
            for path in cot_files
        }

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt."""

        prompt_path = self.cot_prompts[problem_name]
        with open(prompt_path, "r", encoding="utf-8") as f:
            cot_prompt = f.read()
        return cot_prompt


class GSM8KCoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of GSM8K."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def load_cot(self, cot_filepath: str):
        """Load the cot examples from the file."""
        with open(cot_filepath, "r", encoding="utf-8") as file:
            self.cot_prompt = file.read()

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt."""
        return self.cot_prompt


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
            folder_name = os.path.basename(folder_path).split(".")[0]
            category_name = tools.format_term(folder_name)
            prompt_files = glob.glob(folder_path + "/*")
            self.cot_prompt = {
                category_name: {
                    os.path.basename(file_path): file_path for file_path in prompt_files
                }
            }

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt.
        The problem name of CoT will be moral_scenarios.
        When the format problem name of llmpebase is Moral Scenarios,
        we need to make a conversion.
        """
        return self.cot_prompt[problem_name][self.cot_filename]


class MMLUCoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of MMLU."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def load_cot(self, cot_filepath: str):
        """Load the cot examples from the file."""
        super().load_cot(cot_filepath)
        # Convert the keys of cot_prompt to be the format one
        self.cot_prompt = {
            tools.format_term(key): value for key, value in self.cot_prompt.items()
        }

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt.
        The problem name of CoT will be moral_scenarios.
        When the format problem name of llmpebase is Moral Scenarios,
        we need to make a conversion.
        """

        return self.cot_prompt[problem_name]


class TheoremQACoTPrompting(base.BaseCoTPrompting):
    """The CoT prompt of TheoremQA."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The final solution is "

    def load_cot_prompt(self, problem_name: str):
        """Load the cot prompt."""
        problem_name = problem_name.replace(" ", "_")
        return self.cot_prompt[problem_name]
