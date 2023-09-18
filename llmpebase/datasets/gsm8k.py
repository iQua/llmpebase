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
from vgbase.datasets.datalib import data_utils
from vgbase.datasets.vgbase_data_structure import DataSourceStructure
from vgbase.config import Config


class GSM8KDataset(torch.utils.data.Dataset):
    """
    An interface for the GSM8K dataset.
    """

    def __init__(self, splits_info, phase):
        # a path showing where the data is stored
        self.splits_info = splits_info
        self.phase = phase

        data_folder = self.splits_info[phase]["path"]
        data_file = self.splits_info[phase]["split_file"]
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


class DataSource(DataSourceStructure):
    """The GSM8K dataset."""

    def __init__(self):
        super().__init__()

        self.supported_modalities = ["text"]
        self.splits = ["train", "test"]
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
                self.source_data_dir_path
            )
            self.splits_info[split_type]["split_file"] = f"{split_type}.parquet"

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        # obtain the datacatalog for desired phase
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
