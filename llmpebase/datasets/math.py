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
from llmpebase.utils import extracter


class MATHDataset(torch.utils.data.Dataset):
    """
    An interface for the MATH dataset.
    """

    def __init__(self, splits_info, phase):
        # A path showing where the data is stored
        self.splits_info = splits_info
        self.phase = phase

        data_folder = self.splits_info[phase]["path"]
        data_phase_folder = self.splits_info[phase]["foldername"]
        self.data_path = os.path.join(data_folder, data_phase_folder)

        self.data_statistic_path = os.path.join(self.data_path, "data_statistic.json")

        self.data_statistic = {}
        self.data_statistic = self.get_data_statistic()

    def get_one_sample(self, filepath: str):
        """Get one sample from the file."""
        filepath = os.path.join(category_path, filename)

        with open(filepath, "r", encoding="utf-8") as json_file:
            # Load the JSON data from the file
            data = json.load(json_file)

        difficulty_level = data["level"]
        solution = data["solution"].rstrip()

        sents = extracter.extract_sentences(solution)
        sents = [sent for sent in sents if "=" in sent or "\\boxed" in sent]
        conclusion = sents[-1]

        groundtruths = extracter.extract_target_equations(conclusion)
        result = groundtruths[-1]
        final_result = extracter.extract_equation_result(result)

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

    def get_qas(self):
        """Getting the qas of the tasks."""
        return self.data_qas

    def __getitem__(self, idx: Tuple):
        """Get the sample for either training or testing given index."""
        return self.data_qas[idx]

    def __len__(self):
        """obtain the number of samples."""
        return len(self.data_qas)

    def format_category_name(self, name):
        """Convert a category name to be format."""
        # Replace _ to whitespace
        # Capitalize the first letter of each word
        name = name.replace("_", " ").replace("and", "&").title()
        return name


class DataSource(base.DataSource):
    """The MATH datasource."""

    def __init__(self):
        # Extract the data information from the config file
        super().__init__()

        self.splits_info = {
            "train": {"path": self.data_path, "foldername": "MATH/train"},
            "test": {"path": self.data_path, "foldername": "MATH/test"},
        }

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        # obtain the datacatalog for desired phase
        self.prepare_data(phase)
        dataset = MATHDataset(splits_info=self.splits_info, phase=phase)
        return dataset

    def get_train_set(self):
        """Obtains the training dataset."""
        phase = "train"
        return self.get_phase_dataset(phase)

    def get_test_set(self):
        """Obtains the validation dataset."""
        phase = "test"
        return self.get_phase_dataset(phase)
