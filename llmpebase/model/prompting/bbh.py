"""
The implementation of different prompts for BBH.
"""
import os
import re
import glob
import random
from typing import List

from llmpebase.model.prompting import base
from llmpebase.dataset.bbh import extract_problem_name


class BBHStandardPrompting(base.BasePrompting):
    """The standard prompt of BBH."""

    def organize_question_prompt(self, sample: dict):
        """Organizing the question prompt."""
        ques = sample["question"]
        prompt = f"""Question: {ques} \n"""
        return prompt

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organizing the answer prompt."""
        answ = sample["answer"]
        answ = "" if not is_answer_included else answ
        return f"""Answer: {answ}."""

    @staticmethod
    def extract_groundtruth(target_answer: str):
        """Extract the target results from the obtained targets."""
        # Compare the answers
        pattern = r"\$?(\d+)(?:\$)?"

        target_answer = str(target_answer)
        result = re.findall(pattern, target_answer)

        if result:
            return float(result[0])
        else:
            return None

    @staticmethod
    def measure_answers(src_answer: str, dst_answer: str):
        """Measuring whether answers are consistent with each other."""
        src_result = BBHStandardPrompting.extract_groundtruth(src_answer)
        dst_result = BBHStandardPrompting.extract_groundtruth(dst_answer)

        if src_result is not None and dst_result is not None:
            return src_result == dst_result

        return None

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the BBH dataset."""

        n_shots = config["n_shots"]

        for test_sample in eval_set:
            problem_name = test_sample.auxiliary["sample_problem"]
            sample_idx = test_sample.auxiliary["sample_idx"]
            sample_indexs = train_set.get_problem_sample_indexs(problem_name)
            # Remove the test sample index to avoid including this test sample
            # in the prompt
            sample_indexs.remove(sample_idx)
            fewshot_indexs = (
                random.sample(sample_indexs, n_shots)
                if len(sample_indexs) > n_shots
                else sample_indexs
            )
            samples = [train_set[idx] for idx in fewshot_indexs]
            request_prompt = self.get_test_prompt(
                problem_name=problem_name,
                template_samples=samples,
                test_sample=test_sample,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]


class BBHCoTPrompting(BBHStandardPrompting):
    """The CoT prompt of BBH."""

    # This should be the same as the answer format in the cot_filepath
    # Current CoT ones use "The answer is".
    answer_format_str: str = "The answer is "

    def __init__(self, model_config: dict, cot_filepath: str = None) -> None:
        super().__init__()
        cot_filepath = (
            cot_filepath if cot_filepath is not None else model_config["cot_filepath"]
        )
        cot_files = glob.glob(cot_filepath)
        self.cot_prompts = {
            extract_problem_name(os.path.basename(path)): path for path in cot_files
        }

    def organize_template_prompt(
        self,
        samples: List[dict],
        problem_name: str = None,
    ):
        """organizing the prompt including the few-shot ."""
        intro_prompt = (
            "The following examples are questions with answers about algebra problems."
        )
        prompt_path = self.cot_prompts[problem_name]
        with open(prompt_path, "r", encoding="utf-8") as f:
            cot_prompt = f.read()
        prompt = f"""{intro_prompt}\n\n {cot_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the BBH dataset."""

        for _, test_sample in enumerate(eval_set):
            problem_name = test_sample.auxiliary["sample_problem"]
            request_prompt = self.get_test_prompt(
                problem_name=problem_name,
                test_sample=test_sample,
                template_samples=None,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]


class BBHZeroShotCoTPrompting(BBHStandardPrompting):
    """The zeroshot CoT prompt of BBH."""

    def organize_answer_prompt(self, sample, is_answer_included=True):
        """Organize the answer prompt."""
        return """Answer: Let's think step by step. \n"""

    def organize_template_prompt(
        self,
        samples: List[dict],
        problem_name: str = None,
    ):
        return ""

    def get_test_prompt(
        self, problem_name: str, test_sample: dict, template_samples: List[dict]
    ):
        """Organizing the prompt for test."""
        test_qa_prompt = self.organize_qa_prompt(test_sample, is_answer_included=False)
        prompt = f"""This is the {problem_name} problem. Please answer the given question.\n\n{test_qa_prompt}"""
        return prompt

    def evaluater(self, train_set, eval_set, config):
        """Evaluating the BBH dataset."""

        for _, test_sample in enumerate(eval_set):
            problem_name = test_sample.auxiliary["sample_problem"]
            request_prompt = self.get_test_prompt(
                problem_name=problem_name,
                test_sample=test_sample,
                template_samples=None,
            )
            yield request_prompt, test_sample, test_sample["groundtruth"]
