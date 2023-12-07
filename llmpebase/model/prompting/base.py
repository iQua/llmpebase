"""
Basic implementation of prompting class.
"""
import json
import random
from typing import List, Union


class BasePrompting:
    """
    The basic prompting behaving as the structure to be followed by
    other customized prompts.
    """

    # For all the following variables, the punctuation should be included.
    # Thus, no punctuation is needed to be added during organizing prompts.
    solution_flag: str = "The final solution is"

    question_prompt_head: str = "Question:"
    question_prompt_tail: str = ""

    answer_prompt_head: str = "Answer:"

    template_prompt_head: str = (
        "Following examples are question-answer pairs about {}.\n\n"
    )
    template_prompt_tail: str = (
        "With above examples, please answer the given question.\n\n"
    )

    notice: str = "Place the final solution after the sentence '{}' at the end of the answer for readability."

    def __init__(self, model_config: dict = None):
        """ """
        self.model_config = model_config

    def organize_question_prompt(self, sample: dict):
        """Organize the question prompt."""
        question = sample["question"]
        prompt = (
            f"""{self.question_prompt_head} {question}. {self.question_prompt_tail}\n"""
        )
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""

        if is_answer_included:
            answer = sample["answer"]
            groundtruth = sample["groundtruth"]
            answer = "" if not is_answer_included else answer
            groundtruth = "" if not is_answer_included else groundtruth

            return f"""{self.answer_prompt_head} {answer}. {self.solution_flag} {groundtruth}."""

        return f"""{self.answer_prompt_head}"""

    def organize_qa_prompt(self, sample: dict, is_answer_included=True):
        """Formatting the qa sample to a format structure."""

        format_question_prompt = self.organize_question_prompt(sample)
        format_answer_prompt = self.organize_answer_prompt(sample, is_answer_included)

        format_str = f"""{format_question_prompt}\n{format_answer_prompt}"""

        return format_str

    def organize_template_prompt(
        self,
        samples: Union[str, List[dict]] = None,
        problem_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        problem_name = "" if problem_name is None else problem_name
        fewshot = "" if samples is None else samples
        if fewshot is not None and isinstance(fewshot, list):
            fewshot = "\n\n".join(
                [self.organize_qa_prompt(sample) for sample in samples]
            )

        head = self.template_prompt_head.format(problem_name)
        prompt = f"""{head}{fewshot}\n\n{self.template_prompt_tail}"""

        return prompt

    def create_test_prompt(
        self,
        problem_name: str,
        test_sample: dict,
        template_samples: Union[str, List[dict]],
    ):
        """Organizing the prompt for test."""
        fewshot_prompt = self.organize_template_prompt(template_samples, problem_name)
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)

        # Assign the solution flag to the notice
        notice = self.notice.format(self.solution_flag)
        prompt = f"""{fewshot_prompt}Notice: {notice}\n\n{test_qa_prompt}"""
        return prompt

    def create_prompt_sample(self, sample, dataset, config):
        """Create one prompt sample."""

        n_shots = config["n_shots"]

        samples = [dataset[random.randint(0, len(dataset))] for _ in range(n_shots)]

        return (
            self.create_test_prompt(
                problem_name=sample["auxiliary"]["sample_problem"],
                template_samples=samples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )


class BaseCoTPrompting(BasePrompting):
    """A base CoT prompting to load prompt from a file."""

    answer_prompt_head: str = "Answer: Let's think step by step. "

    def __init__(self, model_config: dict, cot_filepath: str = None) -> None:
        super().__init__(model_config)

        cot_filepath = (
            cot_filepath if cot_filepath is not None else model_config["cot_filepath"]
        )
        self.cot_prompt = None
        self.load_cot(cot_filepath)

    def load_cot(self, cot_filepath: str):
        """Load the cot examples from the file."""
        with open(cot_filepath, "r", encoding="utf-8") as file:
            self.cot_prompt = json.load(file)

    def get_cot_prompt(self, problem_name: str, **kwargs):
        """Load the cot prompt."""
        problem_name = problem_name.replace(" ", "_").lower()
        return self.cot_prompt[problem_name]

    def create_prompt_sample(self, sample, dataset, config):
        """Create one prompt sample."""
        problem_name = sample["auxiliary"]["sample_problem"]
        cot_samples = self.get_cot_prompt(problem_name)
        return (
            self.create_test_prompt(
                problem_name=problem_name,
                template_samples=cot_samples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )


class BaseZeroShotPrompting(BasePrompting):
    """A base zero-shot prompting"""

    answer_prompt_head: str = "Answer: Let's think step by step."

    template_prompt_head: str = "Answer the question about the problem {}"
    template_prompt_tail: str = ""

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        return f"""{self.answer_prompt_head}\n"""

    def create_prompt_sample(self, sample, dataset, config):
        """Create one prompt sample."""

        return (
            self.create_test_prompt(
                problem_name=sample["auxiliary"]["sample_problem"],
                template_samples=None,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )
