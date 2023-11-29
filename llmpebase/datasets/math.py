""" 
The datasource inferance for the MATH dataset.
The detaild information of it is shown in 
https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
"""
import os
import json
from typing import Tuple

import torch

from llmpebase.datasets import base
from llmpebase.datasets.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)

from llmpebase.utils import extractor


def format_category_name(name):
    """Convert a category name to be format."""
    # Replace _ to whitespace
    # Capitalize the first letter of each word
    name = name.replace("_", " ").replace("and", "&").title()
    return name


class MATHDataset(base.BaseDataset):
    """
    An interface for the MATH dataset.
    """

    def create_data_catalog(self):
        category_names = [
            folder_name
            for folder_name in os.listdir(self.phase_data_path)
            if os.path.isdir(os.path.join(self.phase_data_path, folder_name))
        ]

        collected_items = []
        n_samples = 0
        for name in category_names:
            category_path = os.path.join(self.phase_data_path, name)
            qa_files = os.listdir(category_path)
            qa_files = [file for file in qa_files if file.endswith(".json")]
            format_name = format_category_name(name)

            for filename in qa_files:
                n_samples += 1

                filepath = os.path.join(category_path, filename)

                with open(filepath, "r", encoding="utf-8") as json_file:
                    # Load the JSON data from the file
                    data = json.load(json_file)

                difficulty_level = data["level"]
                if difficulty_level in data_statistic[format_name]:
                    data_statistic[format_name][difficulty_level] += 1
                else:
                    data_statistic[format_name].update({difficulty_level: 1})

                data_statistic["total_samples"] += 1
                BaseQASampleInfo(
                    sample_id=sample_idx + 1,
                    sample_filepath=self.phase_data_path,
                )

    def get_one_sample(self, filepath: str):
        """Get one sample from the file."""
        filepath = os.path.join(category_path, filename)

        with open(filepath, "r", encoding="utf-8") as json_file:
            # Load the JSON data from the file
            data = json.load(json_file)

        difficulty_level = data["level"]
        solution = data["solution"].rstrip()

        sents = extractor.extract_sentences(solution)
        sents = [sent for sent in sents if "=" in sent or "\\boxed" in sent]
        conclusion = sents[-1]

        groundtruths = extractor.extract_target_equations(conclusion)
        result = groundtruths[-1]
        final_result = extractor.extract_equation_result(result)

        return {
            "question": data["problem"],
            "answer": solution,
            "conclusion": conclusion,
            "result_equation": result,
            "groundtruth": final_result,
            "level": difficulty_level,
            "type": data["type"],
        }

    def get_data_statistic(self):
        """Collecting the question and the correspondin answer."""
        # Filter the list to include only files ending with ".json"
        data_statistic = {"total_samples": 0}
        if os.path.exists(self.data_statistic_path):
            filepath = self.data_statistic_path
            with open(filepath, "r", encoding="utf-8") as json_file:
                # Load the JSON data from the file
                data_statistic = json.load(json_file)

            return data_statistic

        category_names = [
            folder_name
            for folder_name in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, folder_name))
        ]

        for name in category_names:
            category_path = os.path.join(self.data_path, name)
            qa_files = os.listdir(category_path)
            qa_files = [file for file in qa_files if file.endswith(".json")]
            format_name = self.format_category_name(name)
            data_statistic[format_name] = {"num_samples": 0}
            for filename in qa_files:
                data_statistic[format_name]["num_samples"] += 1
                filepath = os.path.join(category_path, filename)

                with open(filepath, "r", encoding="utf-8") as json_file:
                    # Load the JSON data from the file
                    data = json.load(json_file)

                difficulty_level = data["level"]
                if difficulty_level in data_statistic[format_name]:
                    data_statistic[format_name][difficulty_level] += 1
                else:
                    data_statistic[format_name].update({difficulty_level: 1})

                data_statistic["total_samples"] += 1

        with open(self.data_statistic_path, "r", encoding="utf-8") as json_file:
            # Save the JSON data from the file
            json.dump(json_file, data_statistic)

        return data_statistic


class DataSource(base.DataSource):
    """The MATH datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = MATHDataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="MATH",
            problem_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "MATH/train"),
                "test": os.path.join(self.data_path, "MATH/test"),
                "val": os.path.join(self.data_path, "MATH/test"),
            },
        )
