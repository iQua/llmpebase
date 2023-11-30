"""
The implementation of different prompts.
"""
import re
from typing import List

from llmpebase.models.prompting import base


class GSM8KStandardPrompting(base.BasePrompting):
    """The standard prompt of GSM8K."""

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]
        prompt = f"""Question: {ques} \n"""
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        groundtruth = sample["groundtruth"]
        answ = "" if not is_answer_included else answ
        groundtruth = "" if not is_answer_included else groundtruth
        return f"""Answer: {answ}. {self.answer_format_str} {groundtruth}"""

    @staticmethod
    def extract_groundtruth(target_answer: str):
        """Extract the target results from the obtained targets."""
        # Compare the answers
        pattern = r"\$?(\d+)(?:\$)?"

        target_answer = str(target_answer)
        result = re.findall(pattern, target_answer)

        if result:
            return float(result[0])
        else:
            return None

    @staticmethod
    def measure_answers(src_answer: str, dst_answer: str):
        """Measuring whether answers are consistent with each other."""
        src_result = GSM8KStandardPrompting.extract_groundtruth(src_answer)
        dst_result = GSM8KStandardPrompting.extract_groundtruth(dst_answer)

        if src_result is not None and dst_result is not None:
            return src_result == dst_result

        return None


class GSM8KCoTPrompting(GSM8KStandardPrompting):
    """The CoT prompt of GSM8K."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    answer_format_str: str = "The answer is "

    def __init__(self, model_config: dict, cot_filepath: str = None) -> None:
        super().__init__()
        cot_filepath = (
            cot_filepath if cot_filepath is not None else model_config["cot_filepath"]
        )
        with open(cot_filepath, "r", encoding="utf-8") as txt_file:
            self.cot_prompt = txt_file.read()

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return """Answer: Let's think step by step. """

    def organize_template_prompt(
        self,
        samples: List[dict],
        problem_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        intro_prompt = (
            "The following examples are questions with answers about algebra problems."
        )

        prompt = f"""{intro_prompt}\n\n {self.cot_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the GSM8K dataset."""

        for _, test_sample in enumerate(eval_set):
            problem_name = test_sample.auxiliary["sample_problem"]
            request_prompt = self.get_test_prompt(
                problem_name=None, test_sample=test_sample, template_samples=None
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]


class GSM8KZeroShotCoTPrompting(GSM8KStandardPrompting):
    """The zeroshot CoT prompt of GSM8K."""

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        return """Answer: Let's think step by step. \n"""

    def organize_template_prompt(
        self,
        samples: List[dict],
        problem_name: str = None,
    ):
        return ""

    def get_test_prompt(
        self, problem_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""{test_qa_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the GSM8K dataset."""

        for _, test_sample in enumerate(eval_set):
            problem_name = test_sample.auxiliary["sample_problem"]
            request_prompt = self.get_test_prompt(
                problem_name=problem_name,
                test_sample=test_sample,
                template_samples=None,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]
