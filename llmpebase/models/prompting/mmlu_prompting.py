"""
The implementation of adjusting different prompts, including
CoT, Tree of Thoughts.
"""
import re
from typing import List

from llmpebase.models.prompting import base


class MMLUStandardPrompting(base.BasePrompting):
    """The standard prompt of MMLU."""

    answer_format_str: str = "The answer is "

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]
        opts = sample["options"]
        prompt = f"""Question: {ques} \nWhich of the following choices is correct? \n{opts}"""
        return prompt

    def extract_contents_target_answer(self, contens: List[str]):
        """Extracting the target answer from the contents of responses."""

        prefix = re.escape(self.answer_format_str)
        # 1. extract the string after the answer format
        pattern = rf"{prefix}([\(]?[A-Za-z][,\)]?)"

        obtained_targets = []
        for content in contens:
            match = re.search(pattern, content, re.IGNORECASE)

            obtained_targets.append(match.group(1) if match else None)

        return obtained_targets

    @staticmethod
    def measure_answers_consistency(src_answer: str, dst_answer: str):
        """Measuring whether answers are consistent with each other."""
        # Remove non-alphanumeric characters and whitespace
        stripped1 = re.sub(r"[^a-zA-Z0-9]", "", src_answer).lower()
        stripped2 = re.sub(r"[^a-zA-Z0-9]", "", dst_answer).lower()

        return stripped1 in stripped2 or stripped2 in stripped1

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
                yield request_prompt


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
