"""
The implementation of adjusting different prompts, including
CoT.
"""
import re
from typing import List

from llmpebase.models.prompting import base


class MMLUStandardPrompting(base.BasePrompting):
    """The standard prompt of MMLU."""

    answer_format_str: str = "The answer is"

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]
        opts = sample["options"]
        prompt = f"""Question: {ques} \nWhich of the following choices is correct? \n{opts}"""
        return prompt

    @staticmethod
    def extract_target_result(target_answer: str):
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
        src_result = MMLUStandardPrompting.extract_target_result(src_answer)
        dst_result = MMLUStandardPrompting.extract_target_result(dst_answer)

        if src_result is not None and dst_result is not None:
            return src_result == dst_result

        return None

    def evaluater(self, train_set, eval_set, eval_config):
        """Evaluating the MMLU dataset."""

        n_shots = eval_config["n_shots"]

        for task_name in train_set.tasks_name:
            train_samples = train_set[task_name, -1]
            shots = train_samples[:n_shots]
            test_task_samples = eval_set[task_name, -1]
            for test_sample in test_task_samples:
                request_prompt = self.organize_test_fewshot_prompt(
                    task_name, shots, test_sample
                )
                yield request_prompt, test_sample, test_sample["answer"]


class MMLUCoTPrompting(MMLUStandardPrompting):
    """The CoT prompt of MMLU."""

    answer_format_str: str = "Thus, the answer is ."

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answer_prompt = super().organize_answer_prompt(sample, is_answer_included)
        return f"""Let's think step by step. \n{answer_prompt}"""

    def organize_fewshot_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        task_name = task_name.replace(" ", "_")
        fewshot_cot_prompt = self.prompt_data[task_name]
        return fewshot_cot_prompt
