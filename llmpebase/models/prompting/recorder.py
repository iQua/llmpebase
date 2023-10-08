"""
Implementation of a recorder for saving prompts, responses, and results.
"""
import os
import logging
from typing import Dict, List, Union
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
import json


class PromptLLMRecoder:
    """A base class to record the prompts, responses, and results."""

    def __init__(
        self,
        records_filename: str = "records",
        samples_filename: str = "samples",
        records_dir: str = None,
        # whether append the existing records instead of overwriting them
        is_append: bool = False,
    ) -> None:
        self.records_filename = records_filename
        self.samples_filename = samples_filename
        self.records_dir = records_dir

        self.default_path()

        os.makedirs(self.records_dir, exist_ok=True)

        self.records_path = os.path.join(
            self.records_dir, self.records_filename + ".json"
        )
        self.samples_path = os.path.join(
            self.records_dir, self.samples_filename + ".json"
        )

        # a pool recording all results
        self.records_pool: OrderedDictType[str, list] = OrderedDict()
        # a pool recording all samples
        self.samples_pool: OrderedDictType[str, list] = OrderedDict()
        # Note, the length of these two pools should be the same
        # as they are corresponding to each other

        if is_append:
            self.load_records()
            self.load_samples()

    def default_path(self):
        """Setting the default path."""
        records_dir = "llm_records" if self.records_dir is None else self.records_dir
        base_dir = os.getcwd()
        self.records_dir = os.path.join(base_dir, records_dir)

    def load_records(self):
        """Loading the records from the disk."""
        if os.path.exists(self.records_path):
            logging.info("%s exists.", self.records_path)
            logging.info("Loading the records from %s", self.records_path)
            with open(self.records_path, "r", encoding="utf-8") as file:
                self.records_pool = json.load(file)

        else:
            logging.info("%s does not exist. Will create new one", self.samples_path)

    def load_samples(self):
        """Loading the records from the disk."""
        if os.path.exists(self.samples_path):
            logging.info("%s exists.", self.samples_path)
            logging.info("Loading the records from %s", self.samples_path)
            with open(self.samples_path, "r", encoding="utf-8") as file:
                self.samples_pool = json.load(file)

        else:
            logging.info("%s does not exist. Will create new one", self.samples_path)

    def polish_string(self, input_str: str):
        """Polishing the question to reduce invalid white spaces."""
        return input_str.strip()

    def is_existed_question(self, question: str):
        """Judge whether the given question exists."""
        if question in self.samples_pool:
            return True

        return False

    def get_answer_idx(self, question: str, answer: str):
        """Judge whether this answer exists for the given question."""
        qs_answers = self.samples_pool[question]
        is_answer_exists = [answer == item["answer"] for item in qs_answers]
        if any(is_answer_exists):
            return is_answer_exists.index(True)

        return -1

    def is_record_prompt_exists(self, qs_record: dict, new_record: dict):
        """Judge whether the prompts of two records are the same."""
        request_prompt = self.polish_string(new_record["request_prompt"])

        if qs_record["request_prompt"] == request_prompt:
            return True

        return False

    def add_one_record(
        self, sample: dict, record: Dict[str, Union[str, List[Union[str, bool]]]]
    ):
        """Adding one record.

        :param record: A `dict` containing
         - request_prompt: str
         - responses: List[str]
         - extracted_answers: List[str]
         - answers_consistency: List[bool]
        """
        question = sample.pop("question", None)
        answer = sample["answer"]
        question = self.polish_string(question)
        answer = self.polish_string(answer)

        if self.is_existed_question(question):
            answer_idx = self.get_answer_idx(question, answer)
            print("answer_idx: ", answer_idx)
            # current answer exists
            if answer_idx != -1:
                # get the corresponding record of the answer
                qs_record = self.records_pool[question][answer_idx]
                print("answer exists")
                if not self.is_record_prompt_exists(qs_record, record):
                    # append the new record
                    print("append the new record")
                    self.records_pool[question].append(record)
                else:
                    print("extending the record")
                    self.records_pool[question][answer_idx]["responses"].extend(
                        record["responses"]
                    )
                    self.records_pool[question][answer_idx]["extracted_answers"].extend(
                        record["extracted_answers"]
                    )
                    self.records_pool[question][answer_idx][
                        "answers_consistency"
                    ].extend(record["answers_consistency"])

            else:
                print("answer not exists")
                self.samples_pool[question].append(sample)
                self.records_pool[question].append(record)

        else:
            # add the new question and the corresponding
            # sample and record
            self.samples_pool[question] = [sample]
            self.records_pool[question] = [record]

    def save_records(self):
        """Saving the records to the disk."""

        with open(self.records_path, "w", encoding="utf-8") as file:
            json.dump(self.records_pool, file, indent=2)

        with open(self.samples_path, "w", encoding="utf-8") as file:
            json.dump(self.samples_pool, file, indent=2)
