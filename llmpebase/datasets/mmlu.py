""" 
The datasource inferance for the MMLU dataset.
The detaild information of it is shown in 
https://huggingface.co/datasets/cais/mmlu
"""
import os
from collections import defaultdict
import glob

import pandas as pd

from llmpebase.datasets import base
from llmpebase.datasets.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)
from llmpebase.utils.formatter import format_term


def extract_problem_name(filename: str, phase: str):
    """Extract the problem name from the filepath."""
    filename = filename.split(".csv")[0]
    phase = "dev" if phase == "train" else phase
    return format_term(filename.replace(phase, ""))


class MMLUDataset(base.BaseDataset):
    """
    An interface for the MMLU dataset.
    """

    def create_data_catalog(self):
        csv_files = []
        if isinstance(self.phase_data_path, (list, tuple)):
            for path in self.phase_data_path:
                csv_files.extend(glob.glob(path + "/*.csv"))
        else:
            csv_files = glob.glob(self.phase_data_path + "/*.csv")

        category_count = defaultdict(dict)
        category_samples = defaultdict(list)
        collected_items = []
        # Visit all files under the folder to get data information
        for idx, file_path in enumerate(csv_files):
            file_name = os.path.basename(file_path)
            problem_name = extract_problem_name(file_name, phase=self.phase)

            data_frame = pd.read_csv(file_path, header=None)
            problem_n_samples = data_frame.shape[0]
            category_count[problem_name]["num_samples"] = problem_n_samples
            # Create sample info
            #  sample_id: using iteration idx as the
            # prefix while the row index as the suffix
            collected_items.extend(
                [
                    BaseQASampleInfo(
                        sample_id=f"{idx}_{i}",
                        sample_problem=problem_name,
                        sample_filepath=file_path,
                    )
                    for i in range(problem_n_samples)
                ]
            )
            current_idx = len(collected_items)
            # Add sample indexs to the category_samples
            category_samples[problem_name].extend(
                range(current_idx - problem_n_samples, current_idx)
            )

        return DatasetCatalog(
            data_phase=self.phase,
            data_samples=collected_items,
            category_samples=category_samples,
            problem_category=list(category_count.keys()),
            data_statistics=DatasetStatistics(
                num_samples=len(collected_items), category_info=category_count
            ),
        )

    def get_sample(self, idx):
        sample_info = self.data_catalog.data_samples[idx]
        sample_id = sample_info["sample_id"]
        sample_problem = sample_info["sample_problem"]
        sample_filepath = sample_info["sample_filepath"]

        data_frame = pd.read_csv(sample_filepath, header=None)
        row_idx = int(sample_id.split("_")[-1])
        # Extract data from the loaded csv
        question = data_frame.iloc[row_idx, 0]
        options = data_frame.iloc[row_idx, 1:-1]
        choice_letters = [chr(ord("A") + num) for num in range(len(options))]
        options_str = [
            f"({letter}) {choice}" for choice, letter in zip(options, choice_letters)
        ]
        options_str = "\n".join(options_str)
        answer = data_frame.iloc[row_idx, -1]
        answer = f"{answer}"

        return BaseQASample(
            question=question,
            answer=answer,
            conclusion=answer,
            groundtruth=answer,
            auxiliary={
                "options": options,
                "choice_letters": choice_letters,
                "option_str": options_str,
                "sample_problem": sample_problem,
            },
        )


class DataSource(base.DataSource):
    """The MMLU datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = MMLUDataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="MMLU",
            task_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": [
                    os.path.join(self.data_path, "dev"),
                    os.path.join(self.data_path, "auxiliary_train"),
                ],
                "test": os.path.join(self.data_path, "test"),
                "val": os.path.join(self.data_path, "val"),
            },
        )
