"""
Basic implementation of Prompting class.
"""

import re
import json
from typing import List


class BasePrompting:
    """The basic prompt.

    Note, we set this Base prompt as the QA task with options.
    """

    answer_format_str: str = "The answer is"

    def __init__(self, prompt_file_path: str = None) -> None:
        """
        :param prompt_file_path: A `Str` showing the file path of the pre-defined
         prompt. This is generally used by CoT.
        """
        self.prompt_data = None
        if prompt_file_path is not None:
            with open(prompt_file_path, "r", encoding="utf-8") as json_file:
                self.prompt_data = json.load(json_file)

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
        """Formatting the qa sample to a format structure."""

        format_question_prompt = self.organize_question_prompt(sample)
        format_answer_prompt = self.organize_answer_prompt(sample, is_answer_included)

        format_str = f"""{format_question_prompt}\n{format_answer_prompt}"""

        return format_str

    def organize_fewshot_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        task_name = "" if task_name is None else task_name

        intro_prompt = f"The following examples are multiple choice questions (with answers) about {task_name}."
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

    def evaluater(self, train_set, eval_set, eval_config) -> str:
        """Evaluating the prompting on the testset.
        This function should be implemented as a yield-based iterator.

        :return request_prompt: A `str` showing the prompting.
        """

        raise NotImplementedError("'evaluater' has not been implemented yet.")

    def extract_target_answers(self, contents: List[str]):
        """Extracting the target answer from the contents of responses."""

        # 1. extract the string after the answer format
        pattern = re.escape(self.answer_format_str) + r".*"

        obtained_targets = []
        for content in contents:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

            # Record the matched target answer
            # or the original content if no match
            extract_answer = match.group(0) if match else content
            obtained_targets.append(extract_answer)

        return obtained_targets
