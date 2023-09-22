"""
The implementation of adjusting different prompts, including
CoT, Tree of Thoughts.
"""
import re
from typing import List


class GSM8KStandardPrompt:
    """The standard prompt of GSM8K."""

    answer_format_str: str = "Thus, the answer is "

    def organize_qa_prompt(self, task_sample: dict, is_answer_included=True):
        """Formatting the qa sample to be chatgpt structure.

        The structure of the sample should be a dict holding three keys:
            - question
            - options
            - answer
        """
        question = task_sample["question"]
        thought_answer = task_sample["answer"]
        target_answer = task_sample["target_answer"]
        answer_prompt = f"""{thought_answer} {self.answer_format_str}{target_answer}"""
        answer = "" if not is_answer_included else answer_prompt
        format_str = f"""Question: {question}\nAnswer: {answer}."""

        return format_str

    def organize_fewshot_prompt(self, task_name: str, task_samples: List[dict]):
        """organizing the prompt including the few-shot ."""
        intro_prompt = "Follow the given examples and answer the question."
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

    @staticmethod
    def measure_answers_consistency(src_answer: str, dst_answer: str):
        """Measuring whether answers are consistent with each other."""

        def get_number(answer):
            """Extracting number from string as the float/int"""
            number = re.findall(r"(\d+\.\d+|\d+\s*[+\-*/]\s*\d+|\d+)", answer)[0]
            return float(number) if "." in number else int(number)

        return get_number(src_answer) == get_number(dst_answer)


class GSM8KCoTPrompt(GSM8KStandardPrompt):
    """The CoT prompt of GSM8K."""

    answer_format_str: str = "The answer is "

    def __init__(self, cot_filepath: str) -> None:
        super().__init__()

        with open(cot_filepath, "r", encoding="utf-8") as txt_file:
            self.cot_prompt = txt_file.read()

    def organize_qa_prompt(self, task_sample: dict, is_answer_included=True):
        """Formatting the qa sample to be chatgpt structure.

        The structure of the sample should be a dict holding three keys:
            - question
            - options
            - answer
        """
        question = task_sample["question"]
        thought_answer = task_sample["answer"]
        target_answer = task_sample["target_answer"]
        answer_prompt = f"""{thought_answer} {self.answer_format_str}{target_answer}"""
        answer = "" if not is_answer_included else answer_prompt
        format_str = (
            f"""Question: {question}\nLet's think step by step. \nAnswer: {answer}."""
        )

        return format_str

    def organize_fewshot_prompt(self, task_name: str, task_samples: List[dict]):
        """organizing the prompt including the few-shot ."""
        fewshot_cot_prompt = self.cot_prompt
        return fewshot_cot_prompt
