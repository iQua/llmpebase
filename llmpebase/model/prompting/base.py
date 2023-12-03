"""
Basic implementation of Prompting class.
"""

import re
import random
from typing import List


class BasePrompting:
    """The basic prompt.

    Note, we set this Base prompt as the QA problem with options.
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
        problem_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        problem_name = "" if problem_name is None else problem_name

        intro_prompt = (
            f"The following examples are questions with answers about {problem_name}."
        )
        problem_content = []
        for sample in samples:
            problem_content.append(self.organize_qa_prompt(sample))
        fewshots = "\n\n".join(problem_content)

        prompt = f"""{intro_prompt}\n\n {fewshots}"""

        return prompt

    def get_test_prompt(
        self, problem_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        fewshot_prompt = self.organize_template_prompt(template_samples, problem_name)
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""{fewshot_prompt} \n\n\n With above examples, please answer: \n \n{test_qa_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the Base dataset.

        The defualt way is to randomly sample the few-shot samples from the train set and
        thus the sampled samples are used as the demonstrations in the prompt.
        """

        n_shots = config["n_shots"]

        for _, test_sample in enumerate(eval_set):
            samples = [
                train_set[random.randint(0, len(eval_set))] for _ in range(n_shots)
            ]

            request_prompt = self.get_test_prompt(
                problem_name=test_sample.auxiliary["sample_problem"],
                template_samples=samples,
                test_sample=test_sample,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]

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
