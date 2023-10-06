"""
The implementation of adjusting different prompts, including
CoT, Tree of Thoughts.
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

    def extract_contents_target_answer(self, contens: List[str]):
        """Extracting the target answer from the contents of responses."""

        prefix = re.escape(self.answer_format_str)
        # 1. extract the string after the answer format
        pattern = rf"{prefix}((?:[\-]?(\d+\.\d+|\d+)|[\+\-\*\/\(\) ])+)"

        obtained_targets = []
        for content in contens:
            match = re.search(pattern, content, re.IGNORECASE)

            obtained_targets.append(match.group(1) if match else None)

        return obtained_targets

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
            yield request_prompt

    @staticmethod
    def measure_answers_consistency(src_answer: str, dst_answer: str):
        """Measuring whether answers are consistent with each other."""

        def get_number(answer):
            """Extracting number from string as the float/int"""
            number = re.findall(r"(\d+\.\d+|\d+\s*[+\-*/]\s*\d+|\d+)", answer)[0]
            return float(number) if "." in number else int(number)

        return get_number(src_answer) == get_number(dst_answer)


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
