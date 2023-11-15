"""
The implementation of adjusting different prompts, including
CoT.
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
        opts = sample["options"]
        prompt = f"""Question: {ques} \nWhich of the following choices is correct? \n{opts}"""
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return f"""Answer: {self.answer_format_str} {answ}. """

    def organize_qa_prompt(self, sample: dict, is_answer_included=True):
        """Formatting the qa sample to be chatgpt structure.

        The structure of the sample should be a dict holding three keys:
            - question
            - options
            - answer
        """
        question = sample["question"]
        thought_answer = sample["answer"]
        target_answer = sample["target_answer"]
        answer_prompt = f"""{thought_answer} {self.answer_format_str}{target_answer}"""
        answer = "" if not is_answer_included else answer_prompt
        format_str = f"""Question: {question}\nAnswer: {answer}."""

        return format_str

    def organize_fewshot_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        intro_prompt = "Follow the given examples and answer the question."
        task_content = []
        for sample in samples:
            task_content.append(self.organize_qa_prompt(sample))
        fewshots = "\n\n".join(task_content)

        prompt = f"""{intro_prompt}\n\n {fewshots}"""

        return prompt

    def organize_test_fewshot_prompt(
        self, task_name: str, few_shot_samples: List[dict], test_sample: dict
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        fewshot_prompt = self.organize_fewshot_prompt(few_shot_samples, task_name)
        prompt = f"""{fewshot_prompt} \n\n\n With above examples, please answer: \n \n{test_qa_prompt}"""
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

    def evaluater(self, train_set, eval_set, eval_config):
        """Evaluating the GSM8K dataset."""

        n_shots = eval_config["n_shots"]

        for _, test_sample in enumerate(eval_set):
            samples = [
                train_set[random.randint(0, len(eval_set))] for _ in range(n_shots)
            ]
            request_prompt = self.organize_test_fewshot_prompt(
                task_name=None, few_shot_samples=samples, test_sample=test_sample
            )
            yield request_prompt, test_sample, test_sample["target_answer"]


class GSM8KCoTPrompting(GSM8KStandardPrompting):
    """The CoT prompt of GSM8K."""

    answer_format_str: str = "The answer is "

    def __init__(self, cot_filepath: str) -> None:
        super().__init__()

        with open(cot_filepath, "r", encoding="utf-8") as txt_file:
            self.cot_prompt = txt_file.read()

    def organize_qa_prompt(self, sample: dict, is_answer_included=True):
        """Formatting the qa sample to be chatgpt structure.

        The structure of the sample should be a dict holding three keys:
            - question
            - options
            - answer
        """
        question = sample["question"]
        thought_answer = sample["answer"]
        target_answer = sample["target_answer"]
        answer_prompt = f"""{thought_answer} {self.answer_format_str}{target_answer}"""
        answer = "" if not is_answer_included else answer_prompt
        format_str = (
            f"""Question: {question}\nLet's think step by step. \nAnswer: {answer}."""
        )

        return format_str

    def organize_test_fewshot_prompt(
        self, task_name: str, few_shot_samples: List[dict], test_sample: dict
    ):
        """organizing the prompt including the few-shot ."""
        fewshot_cot_prompt = self.cot_prompt
        return fewshot_cot_prompt
