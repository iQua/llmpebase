"""
The implementation of adjusting different prompts, including
CoT, Tree of Thoughts.
"""
import re
import json
from typing import List


class MMLUStandardPrompt:
    """The standard prompt of MMLU."""

    answer_format_str: str = "The answer is "

    def organize_question_prompt(self, task_sample: dict):
        """Organizing the question prompt."""
        ques = task_sample["question"]
        opts = task_sample["options"]
        prompt = f"""Q: {ques} \nWhich of the following choices is correct? \n{opts}"""
        return prompt

    def organize_qa_prompt(self, task_sample: dict, is_answer_included=True):
        """Formatting the qa sample to be chatgpt structure.

        The structure of the sample should be a dict holding three keys:
            - question
            - options
            - answer
        """
        answ = task_sample["answer"]

        format_question_prompt = self.organize_question_prompt(task_sample)
        answ = "" if not is_answer_included else answ
        format_str = (
            f"""{format_question_prompt}\nAnswer: {self.answer_format_str} {answ}."""
        )

        return format_str

    def organize_fewshot_prompt(self, task_name: str, task_samples: List[dict]):
        """organizing the prompt including the few-shot ."""
        intro_prompt = f"The following examples are multiple choice questions (with answers) about {task_name}."
        task_content = []
        for sample in task_samples:
            task_content.append(self.organize_qa_prompt(sample))
        fewshots = "\n\n".join(task_content)

        prompt = f"""{intro_prompt}\n\n {fewshots}"""

        return prompt

    def organize_test_prompt(
        self, task_name: str, few_shot_samples: List[dict], test_sample: dict
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        fewshot_prompt = self.organize_fewshot_prompt(task_name, few_shot_samples)
        prompt = f"""{fewshot_prompt} \n\n\n With above examples, please answer: \n {test_qa_prompt}"""
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


class MMLUCoTPrompt(MMLUStandardPrompt):
    """The CoT prompt of MMLU."""

    answer_format_str: str = "Thus, the answer is ."

    def __init__(self, cot_filepath: str) -> None:
        super().__init__()

        with open(cot_filepath, "r", encoding="utf-8") as json_file:
            self.cot_prompt = json.load(json_file)

    def organize_qa_prompt(self, task_sample: dict, is_answer_included=True):
        """Formatting the qa sample to be chatgpt structure.

        The structure of the sample should be a dict holding three keys:
            - question
            - options
            - answer
        """
        answ = task_sample["answer"]

        format_question_prompt = self.organize_question_prompt(task_sample)
        answ = "" if not is_answer_included else answ
        format_str = (
            f"""{format_question_prompt}\nLet's think step by step. \nA: {answ}."""
        )

        return format_str

    def organize_fewshot_prompt(self, task_name: str, task_samples: List[dict]):
        """organizing the prompt including the few-shot ."""
        task_name = task_name.replace(" ", "_")
        fewshot_cot_prompt = self.cot_prompt[task_name]
        return fewshot_cot_prompt
