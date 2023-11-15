""" 
The datasource inferance for the MMLU dataset.
The detaild information of it is shown in 
https://huggingface.co/datasets/cais/mmlu
"""
import os
from typing import Tuple
import glob

import torch
import pandas as pd

from llmpebase.datasets import base


class MMLUDataset(torch.utils.data.Dataset):
    """
    An interface for the MMLU dataset.
    """

    def __init__(self, splits_info, phase):
        # a path showing where the data is stored
        self.splits_info = splits_info
        self.phase = phase
        self.data_folder = self.splits_info[phase]["path"]
        self.tasks_name = []
        self.tasks_file_mapper = {}
        self.tasks_data = {}
        self.tasks_qas = {}

        self.collect_tasks()

    def collect_tasks(self):
        """Collecting tasks of MMLU dataset."""
        csv_files = glob.glob(self.data_folder + "/*.csv")

        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            file_name_no_extension = file_name.split(".csv")[0]
            task_name = (
                file_name_no_extension.replace(self.phase, "")
                .replace("_", " ")
                .rstrip()
            )
            self.tasks_name.append(task_name)

            self.tasks_file_mapper[task_name] = file_path
            data_frame = pd.read_csv(file_path, header=None)

            self.tasks_data[task_name] = data_frame
            self.tasks_qas[task_name] = self.collect_qas(data_frame)

    def collect_qas(self, data_frame):
        """Collecting the question and the correspondin answer."""
        n_itmes = data_frame.shape[0]
        collected_items = []
        for row_idx in range(n_itmes):
            question = data_frame.iloc[row_idx, 0]
            options = data_frame.iloc[row_idx, 1:-1]
            choices_letters = [chr(ord("A") + num) for num in range(len(options))]
            options_str = [
                f"({letter}) {choice}"
                for choice, letter in zip(options, choices_letters)
            ]
            options_str = "\n".join(options_str)
            answer = data_frame.iloc[row_idx, -1]
            answer = f"({answer})"
            target_answer = answer
            collected_items.append(
                {
                    "question": question,
                    "options": options_str,
                    "answer": answer,
                    "target_answer": target_answer,
                },
            )

        return collected_items

    def get_task_qas(self, task_name):
        """Getting the qas of the tasks."""
        return self.tasks_qas[task_name]

    def __getitem__(self, task_and_idx: Tuple):
        """Get the sample for either training or testing given index."""
        task_name, idx = task_and_idx
        task_qas = self.tasks_qas[task_name]

        # get all qas of this task
        if idx == -1:
            return task_qas
        else:
            return task_qas[idx]

    def __len__(self):
        """obtain the number of samples."""
        return len(self.tasks_name)

    def get_sample(self):
        """Getting one sample"""


class DataSource(base.DataSource):
    """The MMLU datasource."""

    def __init__(self):
        super().__init__()

        # Set the splits for MMLU dataset
        self.splits_info = {
            "dev": {"path": os.path.join(self.data_path, "dev")},
            "test": {"path": os.path.join(self.data_path, "test")},
            "val": {"path": os.path.join(self.data_path, "val")},
        }

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        # Obtain the dataset for desired phase
        self.prepare_source_data(phase)
        dataset = MMLUDataset(splits_info=self.splits_info, phase=phase)
        return dataset

    def get_train_set(self):
        """Obtains the training dataset."""
        phase = "dev"
        return self.get_phase_dataset(phase)
