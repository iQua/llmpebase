"""
The implementation of different prompts.
"""
import re
import random
from typing import List
import json

from llmpebase.models.prompting import base


class MMLUStandardPrompting(base.BasePrompting):
    """The standard prompt of MMLU."""

    answer_format_str: str = "The final choice is"

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""

        ques = sample["question"]
        opts = sample.auxiliary["option_str"]
        prompt = f"""Question: {ques} \nWhich of the following choices is correct? \n{opts}"""
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return f"""Answer: {self.answer_format_str} {answ}. """

    @staticmethod
    def extract_groundtruth(target_answer: str):
        """Extract the target results from the obtained targets."""
        # Compare the answers
        pattern = r"\b\([ABCDabcd]\)\b|\b[ABCDabcd]\b|\b\d+\b|\b\d+\.\d+\b|\(\d+\)|\(\d+\.\d+\)"

        result = re.findall(pattern, target_answer)
        if result:
            return result[0]
        else:
            return None

    @staticmethod
    def measure_answers(src_answer: str, dst_answer: str):
        """Measuring whether answers are consistent with each other."""

        # Use re.findall to find all occurrences of the pattern in the text
        src_result = MMLUStandardPrompting.extract_groundtruth(src_answer)
        dst_result = MMLUStandardPrompting.extract_groundtruth(dst_answer)

        if src_result is not None and dst_result is not None:
            return src_result == dst_result

        return None

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the MMLU dataset."""

        n_shots = config["n_shots"]

        for _, test_sample in enumerate(eval_set):
            task_name = test_sample.auxiliary["sample_task"]
            sample_indexs = train_set.get_task_sample_indexs(task_name)
            fewshot_indexs = (
                random.sample(sample_indexs, n_shots)
                if len(sample_indexs) > n_shots
                else sample_indexs
            )
            samples = [train_set[idx] for idx in fewshot_indexs]
            request_prompt = self.get_test_prompt(
                task_name=task_name, template_samples=samples, test_sample=test_sample
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]


class MMLUCoTPrompting(MMLUStandardPrompting):
    """The CoT prompt of MMLU."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    answer_format_str: str = "The answer is "

    def __init__(self, model_config: dict, cot_filepath: str = None) -> None:
        super().__init__()
        cot_filepath = (
            cot_filepath if cot_filepath is not None else model_config["cot_filepath"]
        )
        with open(cot_filepath, "r", encoding="utf-8") as txt_file:
            self.cot_prompt = json.load(txt_file)

    def load_cot_prompt(self, task_name: str):
        """Load the cot prompt."""
        task_name = task_name.replace(" ", "_")
        return self.cot_prompt[task_name]

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return """Answer: Let's think step by step."""

    def organize_template_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        intro_prompt = (
            f"""The following examples are questions with answers about {task_name}."""
        )
        task_cot_prompt = self.load_cot_prompt(task_name)
        prompt = f"""{intro_prompt}\n\n {task_cot_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the MMLU dataset."""

        for _, test_sample in enumerate(eval_set):
            task_name = test_sample["auxiliary"]["task_name"]
            request_prompt = self.get_test_prompt(
                task_name=task_name,
                template_samples=None,
                test_sample=test_sample,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]


class MMLUZeroShotCoTPrompting(MMLUStandardPrompting):
    """The zeroshot CoT prompt of MMLU."""

    answer_format_str: str = "The final choice is"

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
        """Evaluating the MMLU dataset."""

        for _, test_sample in enumerate(eval_set):
            task_name = test_sample["auxiliary"]["task_name"]
            request_prompt = self.get_test_prompt(
                task_name=task_name,
                template_samples=None,
                test_sample=test_sample,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]
