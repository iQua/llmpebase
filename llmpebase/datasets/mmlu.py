""" 
The datasource inferance for the MMLU dataset.
The detaild information of it is shown in 
https://huggingface.co/datasets/cais/mmlu
"""
import os, re
from typing import Tuple, List
import glob

import torch
import pandas as pd
from vgbase.datasets.datalib import data_utils
from vgbase.datasets.vgbase_data_structure import DataSourceStructure
from vgbase.config import Config


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


class DataSource(DataSourceStructure):
    """The MMLU dataset."""

    def __init__(self):
        super().__init__()

        self.supported_modalities = ["text"]
        self.splits = ["dev", "val", "test"]
        self.source_data_types = []
        self.source_data_file_formats = []

        self.build_source_data_structure()
        self.build_splits_structure()

    def prepare_source_data(self):
        """Prepare the source data."""

        self.source_data_name = Config().data.datasource_name
        self.source_data_path = Config().data.datasource_path
        self.source_data_dir_path = os.path.join(
            self.source_data_path, self.source_data_name
        )
        source_data_download_url = Config().data.datasource_download_url
        if source_data_download_url is not None:
            data_utils.download_url_data(
                download_url_address=source_data_download_url,
                obtained_file_name=self.source_data_name,
                put_data_dir=self.source_data_path,
            )
        self.connect_source_data(self.source_data_dir_path)

    def build_splits_structure(self):
        # generate path/type information for splits
        for split_type in self.splits:
            self.splits_info[split_type]["path"] = os.path.join(
                self.source_data_dir_path, split_type
            )
            self.splits_info[split_type]["split_file"] = None

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        # obtain the datacatalog for desired phase
        dataset = MMLUDataset(splits_info=self.splits_info, phase=phase)
        return dataset

    def get_train_set(self):
        """Obtains the training dataset."""
        phase = "dev"
        return self.get_phase_dataset(phase)

    def get_test_set(self):
        """Obtains the validation dataset."""
        phase = "test"
        return self.get_phase_dataset(phase)

    def get_val_set(self):
        """Obtains the validation dataset."""
        phase = "val"
        return self.get_phase_dataset(phase)
