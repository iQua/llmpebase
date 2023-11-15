"""
The implementation of adjusting different prompts, including
zero-shot CoT.
"""
import re

from llmpebase.models.prompting import base


class GameOf24StandardPrompting(base.BasePrompting):
    """The standard prompt of GameOf24."""

    answer_format_str: str = "The solution equation is"

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]
        instruction = "In each step: First, two numbers are selected from the current number set to perform +,-,*,/ to obtain a new number. Second, these two selected numbers are deleted from the current set. After deleting, if there is no remaining number, you reach the result. Otherwise, you combine the remaining numbers and the obtained new number into a new set for the subsequent reasoning step."

        prompt = f"""In the game of 24, you are given four numbers, and the goal is to use basic arithmetic operations (+, -, *, /) to combine these numbers and obtain a result of 24. You can only use each number once, and parentheses can be used to change the order of operations. \n Task instruction {instruction}. \n The given four numbers are: {ques}. Write your answer below. After analysis, please place the summarzied solution equation of these four numbers after '{self.answer_format_str}'."""

        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return f"""Answer: {self.answer_format_str} {answ}. """

    @staticmethod
    def extract_target_result(target_answer: str):
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

        src_result = GameOf24StandardPrompting.extract_target_result(src_answer)

        if src_result is not None:
            return eval(src_result) == dst_answer

        return None

    def evaluater(self, train_set, eval_set, eval_config):
        """Evaluating the GameOf24 dataset."""

        for _, test_sample in enumerate(eval_set):
            request_prompt = self.organize_question_prompt(sample=test_sample)
            yield request_prompt, test_sample, test_sample["target_answer"]
