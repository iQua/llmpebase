""" 
The datasource inferance for the MATH dataset.
The detaild information of it is shown in 
https://github.com/hendrycks/math
https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
"""
import os

import torch
from vgbase.datasets.datalib import data_utils
from vgbase.datasets.vgbase_data_structure import DataSourceStructure
from vgbase.config import Config


class MATHDataset(torch.utils.data.Dataset):
    """
    An interface for the MMLU dataset.
    """


class DataSource(DataSourceStructure):
    """The Flickr30K Entities dataset."""

    def __init__(self):
        super().__init__()

        self.supported_modalities = ["text"]
        self.splits = ["dev", "val", "test"]
        self.source_data_types = [
            "Text",
        ]
        self.source_data_file_formats = ["csv"]

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
