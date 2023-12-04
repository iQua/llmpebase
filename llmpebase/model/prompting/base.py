"""
Basic implementation of Prompting class.
"""

import random
from typing import List


class BasePrompting:
    """The basic prompt.

    Note, we set this Base prompt as the QA problem with options.
    """

    solution_flag: str = "The final solution is"

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

    def create_test_prompt(
        self, problem_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        fewshot_prompt = self.organize_template_prompt(template_samples, problem_name)
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""{fewshot_prompt} \n\n\n With above examples, please answer: \n \n{test_qa_prompt}"""
        return prompt

    def create_prompt_sample(self, sample, dataset, config):
        """Create one prompt sample."""

        n_shots = config["n_shots"]

        samples = [dataset[random.randint(0, len(dataset))] for _ in range(n_shots)]

        return (
            self.create_test_prompt(
                problem_name=sample.auxiliary["sample_problem"],
                template_samples=samples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )
