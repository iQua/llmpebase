"""
A base datasource class.
"""

import os
import re
import logging

from torchvision.datasets.utils import download_url, extract_archive

from llmpebase.config import Config


def extract_compression_style(url):
    """Extract the style of compression from the url."""
    pattern = r"\.(zip|tar|tar\.gz)$"
    match = re.search(pattern, url)
    if match is not None:
        return match.group(1)

    return "zip"


class DataSource:
    """The Base datasource."""

    def __init__(self):
        # Extract the data information from the config file
        self.source_data_name = Config().data.datasource_name
        self.source_data_path = Config().data.datasource_path
        self.download_url = Config().data.datasource_download_url
        self.data_path = os.path.join(self.source_data_path, self.source_data_name)

        self.splits_info = {
            "train": {"path": self.data_path, "filename": ""},
            "test": {"path": self.data_path, "filename": ""},
        }

    def prepare_source_data(self, phase: str):
        """Prepare the source data."""
        split_info = self.splits_info[phase]
        phase_data_path = split_info["path"]

        if "filename" in split_info and split_info["filename"] != "":
            phase_data_path = os.path.join(phase_data_path, split_info["filename"])

        if self.download_url is not None and not os.path.exists(phase_data_path):
            compression = extract_compression_style(self.download_url)
            filename = self.source_data_name + "." + compression
            download_file_path = os.path.join(self.data_path, filename)

            if not os.path.exists(download_file_path):
                download_url(
                    url=self.download_url,
                    root=self.data_path,
                    filename=filename,
                )

            extracted_folder = extract_archive(
                from_path=download_file_path,
                to_path=self.data_path,
            )
            logging.info(
                "Downloaded the %s and extract to %s.",
                download_file_path,
                extracted_folder,
            )
        if not os.path.exists(phase_data_path):
            raise FileNotFoundError(f"{phase_data_path} does not exist.")

        logging.info("Connected to the data source in %s.", phase_data_path)

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        self.prepare_source_data(phase)
        # Obtain the datacatalog for desired phase
        raise NotImplementedError(
            "An implementation of one specific dataset is required."
        )

    def get_train_set(self):
        """Obtains the training dataset."""
        phase = "train"
        return self.get_phase_dataset(phase)

    def get_test_set(self):
        """Obtains the validation dataset."""
        phase = "test"
        return self.get_phase_dataset(phase)
