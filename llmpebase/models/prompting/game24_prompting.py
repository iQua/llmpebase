"""
The implementation of adjusting different prompts, including
zero-shot CoT.
"""
import re
from typing import List

from llmpebase.models.prompting import base


class GameOf24StandardPrompting(base.BasePrompting):
    """The standard prompt of GameOf24."""

    answer_format_str: str = "The solution equation is:"

    instruction: str = " Within each step, two numbers are selected from the current number set to perform +,-,*,/ to obtain a new number. Then, these two selected numbers are deleted from the current set. After deleting, if there is no remaining number, you reach the result. Otherwise, you combine the remaining numbers and the obtained new number into a new set for the subsequent reasoning step. Therefore, the current number set of this step is the new set of the previous step."

    analysis_format: str = " Step <idx>, Current set: , Selected two numbers: , Operation: , Computed new number: , Remaining numbers: , New set: "

    notice: str = """ When no remaining numbers, the result of the reasoning chain will be the number in the Computed new number of the analysis. The analysis steps are correct only when this number equals 24 mathematically.  """

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]

        prompt = f"""In the game of 24, you are given four numbers, and the goal is to use basic arithmetic operations (+, -, *, /) to combine these numbers and obtain a result of 24. You can only use each number once, and parentheses can be used to change the order of operations. \n  Task rule: {self.instruction}. \n  Analysis format of each step: {self.analysis_format}. \n  Notice: 1. {self.notice}, 2. Do not add any words apart from the Analysis format. \n\nThe given four numbers are: {ques}. """
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return f"""Answer: Let's think step by step. {answ}"""

    def organize_template_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        return ""

    def get_test_prompt(
        self, task_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""{test_qa_prompt}"""
        return prompt

    @staticmethod
    def extract_groundtruth(target_answer: str):
        """Extract the target results from the obtained targets."""
        # Get the core equation such as (4+5)*6-7

        equations = target_answer.split("=")
        core_equation = max(equations, key=len)
        core_equation = re.sub(r"[^0-9+\-*/()]", "", core_equation)
        if core_equation:
            return core_equation
        else:
            return None

    @staticmethod
    def measure_answers(src_answer: str, dst_answer: int = 24):
        """Measuring whether answers are consistent with each other."""

        src_result = GameOf24StandardPrompting.extract_groundtruth(src_answer)

        if src_result is not None:
            return eval(src_result) == dst_answer

        return None

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the GameOf24 dataset."""

        for _, test_sample in enumerate(eval_set):
            request_prompt = self.get_test_prompt(
                task_name=None, template_samples=None, test_sample=test_sample
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]
