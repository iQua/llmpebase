"""
Basic implementation of Prompting class.
"""

import re
from typing import List


class BasePrompting:
    """The basic prompt.

    Note, we set this Base prompt as the QA task with options.
    """

    answer_format_str: str = "The final solution is"

    def __init__(self, model_config: dict = None):
        """ """
        self.model_config = model_config

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        raise NotImplementedError(
            "'organize_question_prompt' has not been implemented yet."
        )

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        raise NotImplementedError(
            "'organize_answer_prompt' has not been implemented yet."
        )

    def organize_qa_prompt(self, sample: dict, is_answer_included=True):
        """Formatting the qa sample to a format structure."""

        format_question_prompt = self.organize_question_prompt(sample)
        format_answer_prompt = self.organize_answer_prompt(sample, is_answer_included)

        format_str = f"""{format_question_prompt}\n{format_answer_prompt}"""

        return format_str

    def organize_template_prompt(
        self,
        samples: List[dict],
        task_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        task_name = "" if task_name is None else task_name

        intro_prompt = (
            f"The following examples are questions with answers about {task_name}."
        )
        task_content = []
        for sample in samples:
            task_content.append(self.organize_qa_prompt(sample))
        fewshots = "\n\n".join(task_content)

        prompt = f"""{intro_prompt}\n\n {fewshots}"""

        return prompt

    def get_test_prompt(
        self, task_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        fewshot_prompt = self.organize_template_prompt(template_samples, task_name)
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""{fewshot_prompt} \n\n\n With above examples, please answer: \n \n{test_qa_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config) -> str:
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
