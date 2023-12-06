"""
The implementation of different prompts.
"""
from typing import List

from llmpebase.model.prompting import base


class GSM8KStandardPrompting(base.BasePrompting):
    """The standard prompt of GSM8K."""

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]
        prompt = f"""Question: {ques} \n"""
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answer = sample["answer"]
        groundtruth = sample["groundtruth"]
        if is_answer_included:
            answer = "" if not is_answer_included else answer
            groundtruth = "" if not is_answer_included else groundtruth

            return f"""Answer: {answer}. {self.solution_flag} {groundtruth}"""
        return """Answer: """


class GSM8KCoTPrompting(GSM8KStandardPrompting):
    """The CoT prompt of GSM8K."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    solution_flag: str = "The answer is "

    def __init__(self, model_config: dict, cot_filepath: str = None) -> None:
        super().__init__()
        cot_filepath = (
            cot_filepath if cot_filepath is not None else model_config["cot_filepath"]
        )
        with open(cot_filepath, "r", encoding="utf-8") as txt_file:
            self.cot_prompt = txt_file.read()

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answer = sample["answer"]
        if is_answer_included:
            return f"""Answer: Let's think step by step. {answer}"""

        return """Answer: Let's think step by step. """

    def organize_template_prompt(
        self,
        samples: List[dict],
        problem_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        head = self.template_prompt_head.format(problem_name)
        prompt = f"""{head}.\n\n {self.cot_prompt}\n\n{self.template_prompt_tail}."""
        return prompt

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the GSM8K dataset."""

        problem_name = sample.auxiliary["sample_problem"]
        return (
            self.create_test_prompt(
                problem_name=problem_name,
                test_sample=sample,
                template_samples=None,
            ),
            sample["groundtruth"],
        )


class GSM8KZeroShotCoTPrompting(GSM8KStandardPrompting):
    """The zeroshot CoT prompt of GSM8K."""

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        return """Answer: Let's think step by step.\n"""

    def organize_template_prompt(
        self,
        samples: List[dict],
        problem_name: str = None,
    ):
        """No template prompt for zeroshot."""
        return ""

    def create_test_prompt(
        self, problem_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        return test_qa_prompt

    def create_prompt_sample(self, sample, dataset, config):
        """Evaluating the GSM8K dataset."""

        problem_name = sample.auxiliary["sample_problem"]
        return (
            self.create_test_prompt(
                problem_name=problem_name,
                test_sample=sample,
                template_samples=None,
            ),
            sample["groundtruth"],
        )
