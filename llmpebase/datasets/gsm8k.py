""" 
The datasource inferance for the GSM8K dataset.
The detaild information of it is shown in 
https://huggingface.co/datasets/gsm8k
"""
import os
import re
from typing import Tuple

import torch
import pandas as pd

from llmpebase.datasets import base


class GSM8KDataset(torch.utils.data.Dataset):
    """
    An interface for the GSM8K dataset.
    """

    def __init__(self, splits_info, phase):
        # a path showing where the data is stored
        self.splits_info = splits_info
        self.phase = phase

        data_folder = self.splits_info[phase]["path"]
        data_file = self.splits_info[phase]["filename"]
        self.data_path = os.path.join(data_folder, data_file)
        self.data_qas = self.collect_qas()

    def collect_qas(self):
        """Collecting the question and the correspondin answer."""
        data_frame = pd.read_parquet(self.data_path, engine="pyarrow")
        n_itmes = data_frame.shape[0]
        collected_items = []
        for row_idx in range(n_itmes):
            question = data_frame.iloc[row_idx, 0]
            answer = data_frame.iloc[row_idx, -1]
            thought_answer = answer.split("####")[0]
            thought_answer = thought_answer.replace("\n", " ").rstrip()
            target_answer = self.get_target_answer(answer)
            collected_items.append(
                {
                    "question": question,
                    "answer": thought_answer,
                    "target_answer": target_answer,
                }
            )

        return collected_items

    def get_qas(self):
        """Getting the qas of the tasks."""
        return self.data_qas

    def get_target_answer(self, answer):
        """Getting the target answer."""
        # Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
        match = re.search(r"#### (\d+(\.\d+)?)", answer)
        if match:
            return (
                float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            )
        return None

    def __getitem__(self, idx: Tuple):
        """Get the sample for either training or testing given index."""
        return self.data_qas[idx]

    def __len__(self):
        """obtain the number of samples."""
        return len(self.data_qas)


class DataSource(base.DataSource):
    """The GSM8K datasource."""

    def __init__(self):
        # Extract the data information from the config file
        super().__init__()

        self.splits_info = {
            "train": {"path": self.data_path, "filename": "train.parquet"},
            "test": {"path": self.data_path, "filename": "test.parquet"},
        }

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        # obtain the datacatalog for desired phase
        self.prepare_source_data(phase)
        dataset = GSM8KDataset(splits_info=self.splits_info, phase=phase)
        return dataset

    def get_train_set(self):
        """Obtains the training dataset."""
        phase = "train"
        return self.get_phase_dataset(phase)

    def get_test_set(self):
        """Obtains the validation dataset."""
        phase = "test"
        return self.get_phase_dataset(phase)
