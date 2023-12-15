"""
Implementations of Zero-shot prompting
"""
from typing import List, Union

from llmpebase.model.prompting import base


class TheoremQAZeroShotPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of TheoremQA."""

    solution_flag: str = "The final solution is"


class MMLUZeroShotPrompting(base.BaseZeroShotPrompting):
    """The zeroshot CoT prompt of MMLU."""

    solution_flag: str = "The final choice is"


class GameOf24ZeroShotPrompting(base.BaseZeroShotPrompting):
    """The standard prompt of GameOf24."""

    solution_flag: str = "The solution equation is:"

    conclusion: str = "When no remaining numbers, the result of the reasoning chain will be the number in the Computed new number of the analysis. Summarizing the reasoning steps into one equation with parentheses. "

    instruction: str = f" Within each step, two numbers are selected from the current number set to perform +,-,*,/ to obtain a new number. Then, these two selected numbers are deleted from the current set. After deleting, if there is no remaining number, you reach the result. Otherwise, you combine the remaining numbers and the obtained new number into a new set for the subsequent reasoning step. Therefore, the current number set of this step is the new set of the previous step. {conclusion}"

    analysis_format: str = " Step <idx>, Current set: , Selected two numbers: , Operation: , Computed new number: , Remaining numbers: , New set: "

    question_prompt_head: str = f"""In the game of 24, you are given four numbers, and each number can be used only once. The goal is to use basic arithmetic operations (+, -, *, /) to combine these numbers and obtain a result of 24.\n Rule: {instruction}\n Analysis format of each step: {analysis_format}.\n"""

    def create_test_prompt(
        self,
        problem_name: str,
        test_sample: dict,
        demonstrations: Union[str, List[dict]],
    ) -> str:
        """Create the test prompt for the sample."""
        # Create the question prompt
        prompt_sample = super().create_test_prompt(
            problem_name, test_sample, demonstrations
        )

        # Added the question prompt head
        prompt_sample.question.head = self.question_prompt_head
        prompt_sample.question.content = prompt_sample.question.content.replace(
            "Question", "Four given numbers are"
        )

        return prompt_sample
