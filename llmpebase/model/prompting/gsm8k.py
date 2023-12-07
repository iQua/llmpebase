"""
The implementation of different prompts.
"""

from llmpebase.model.prompting import base


class GSM8KStandardPrompting(base.BasePrompting):
    """The standard prompt of GSM8K."""


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


class GSM8KZeroShotCoTPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of GSM8K."""
