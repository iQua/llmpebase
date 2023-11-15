"""
The implementation of different prompts.
"""
import re
import random
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
        target_result = sample["target_result"]
        answ = "" if not is_answer_included else answ
        target_result = "" if not is_answer_included else target_result
        return f"""Answer: {answ}. {self.answer_format_str} {target_result}"""

    def organize_template_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        intro_prompt = (
            "The following examples are questions with answers about algebra problems."
        )
        task_content = []
        for sample in samples:
            task_content.append(self.organize_qa_prompt(sample))
        fewshots = "\n\n".join(task_content)

        prompt = f"""{intro_prompt}\n\n {fewshots}"""

        return prompt

    @staticmethod
    def extract_target_result(target_answer: str):
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
        src_result = GSM8KStandardPrompting.extract_target_result(src_answer)
        dst_result = GSM8KStandardPrompting.extract_target_result(dst_answer)

        if src_result is not None and dst_result is not None:
            return src_result == dst_result

        return None

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the GSM8K dataset."""

        n_shots = config["n_shots"]

        for _, test_sample in enumerate(eval_set):
            samples = [
                train_set[random.randint(0, len(eval_set))] for _ in range(n_shots)
            ]
            request_prompt = self.get_test_prompt(
                task_name=None, template_samples=samples, test_sample=test_sample
            )
            yield request_prompt, test_sample, test_sample["target_result"]


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

    def organize_template_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
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
            request_prompt = self.get_test_prompt(
                task_name=None, test_sample=test_sample, template_samples=None
            )
            yield request_prompt, test_sample, test_sample["target_result"]


class GSM8KZeroShotCoTPrompting(GSM8KStandardPrompting):
    """The zeroshot CoT prompt of GSM8K."""

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        return """Answer: Let's think step by step. \n"""

    def organize_template_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        return ""

    def get_test_prompt(
        self, task_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""{test_qa_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the GSM8K dataset."""

        for _, test_sample in enumerate(eval_set):
            request_prompt = self.get_test_prompt(
                task_name=None, test_sample=test_sample, template_samples=None
            )
            yield request_prompt, test_sample, test_sample["target_result"]
