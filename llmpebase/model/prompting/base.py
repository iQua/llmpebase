"""
Basic implementations of standard, fewshot, and zeroshot prompting.
"""
import json
import random
from typing import List, Union
from dataclasses import asdict


from llmpebase.model.prompting.prompt_generic import (
    BasicPromptFormat,
    BasicAnswerPromptFormat,
    BasicPromptSample,
)


class BasePrompting:
    """
    The basic prompting behaving as the structure to be followed by
    other customized prompts.
    """

    # For all the following variables, the punctuation should be included.
    # Thus, no punctuation is needed to be added during organizing prompts.
    solution_flag: str = "The final solution is"

    # Set the basic format for each part of the prompt
    demonstrate_format = BasicPromptFormat(
        head="\nFollowing demonstrations are question-answer pairs about {}.\n\n",
        content="{}\n",
        notice="\n",
        tail=(
            "With the above demonstrations, please answer the subsequently question.\n\n"
        ),
        prompt="",
    )
    question_format = BasicPromptFormat(
        head="",
        content="Question: {}",
        notice=" ",
        tail="\n",
        prompt="",
    )

    answer_format = BasicAnswerPromptFormat(
        head="\n",
        content="Answer: {}",
        groundtruth=" ",
        notice="",
        tail="",
        prompt="",
    )

    def __init__(self, model_config: dict = None):
        self.model_config = model_config

    def organize_question_prompt(self, sample: dict, problem_name: str):
        """Organize the question prompt."""
        # Create the question prompt following the format
        question_prompt = BasicPromptFormat(**asdict(self.question_format))

        question = sample["question"]
        question_prompt.content = question_prompt.content.format(question)

        return question_prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        answer_prompt = BasicAnswerPromptFormat(**asdict(self.answer_format))
        answer = sample["answer"] if is_answer_included else ""

        if is_answer_included:
            groundtruth = sample["groundtruth"] if is_answer_included else ""
            answer_prompt.groundtruth = f"""{self.solution_flag} {groundtruth}"""

        answer_prompt.content = answer_prompt.content.format(answer)
        return answer_prompt

    def organize_demonstration_prompt(
        self,
        demonstrations: Union[str, List[dict]] = None,
        problem_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""

        if demonstrations is None:
            return ""

        demonstration_prompt = BasicPromptFormat(**asdict(self.demonstrate_format))
        problem_name = "" if problem_name is None else problem_name
        content = demonstrations if isinstance(demonstrations, str) else []

        if isinstance(demonstrations, list):
            for example in demonstrations:
                question_prompt = self.organize_question_prompt(example, problem_name)
                answer_prompt = self.organize_answer_prompt(example)
                content.append(f"""{question_prompt}{answer_prompt}""")

            content = "\n\n".join(content)

        demonstration_prompt.head = demonstration_prompt.head.format(problem_name)
        demonstration_prompt.content = demonstration_prompt.content.format(content)

        return demonstration_prompt

    def create_test_prompt(
        self,
        problem_name: str,
        test_sample: dict,
        demonstrations: Union[str, List[dict]],
    ):
        """Organizing the prompt for test."""
        demonstration_prompt = self.organize_demonstration_prompt(
            demonstrations, problem_name
        )
        question_prompt = self.organize_question_prompt(test_sample, problem_name)
        answer_prompt = self.organize_answer_prompt(
            test_sample, is_answer_included=False
        )
        prompt_sample = BasicPromptSample(
            notice="After getting the final solution, place it after the sentence '{}' for readability.\n",
            solution_flag=self.solution_flag,
            demonstrations=demonstration_prompt,
            question=question_prompt,
            answer=answer_prompt,
            prompt="",
        )
        prompt_sample.head = prompt_sample.head.format(problem_name)
        prompt_sample.notice = prompt_sample.notice.format(self.solution_flag)
        return prompt_sample

    def create_prompt_sample(self, sample, dataset, config: dict):
        """Create one prompt sample.

        :param sample: The `BaseQASample` instance.
        :param dataset: The `BaseDataset` instance.
        """

        n_shots = config["n_shots"]

        samples = [dataset[random.randint(0, len(dataset))] for _ in range(n_shots)]

        return (
            self.create_test_prompt(
                problem_name=sample["auxiliary"]["sample_problem"],
                demonstrations=samples,
                test_sample=sample,
            ),
            sample["groundtruth"],
        )


class BaseCoTPrompting(BasePrompting):
    """A base CoT prompting to load prompt from a file."""

    answer_content: str = "Answer: Let's think step by step. "

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
        prompt_sample = self.create_test_prompt(
            problem_name=problem_name, demonstrations=cot_samples, test_sample=sample
        )

        prompt_sample.answer.content = self.answer_content
        return prompt_sample, sample["groundtruth"]


class BaseZeroShotPrompting(BasePrompting):
    """A base zero-shot prompting."""

    answer_content: str = "Answer: Let's think step by step."

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        answer_prompt = super().organize_answer_prompt(sample, is_answer_included=False)
        answer_prompt.content = self.answer_content

        return answer_prompt

    def create_prompt_sample(self, sample, dataset, config):
        """Create one prompt sample."""
        prompt_sample = self.create_test_prompt(
            problem_name=sample["auxiliary"]["sample_problem"],
            demonstrations=None,
            test_sample=sample,
        )

        prompt_sample.answer.content = self.answer_content

        return prompt_sample, sample["groundtruth"]
