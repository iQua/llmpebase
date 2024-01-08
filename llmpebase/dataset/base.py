"""
A base datasource class.
"""

import os
import re
import json
import logging

import torch
from torchvision.datasets.utils import download_url, extract_archive

from llmpebase.dataset.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
)
from llmpebase.model import define_model
from llmpebase.extractor import get as get_extractor
from llmpebase.config import Config


def extract_compression_style(url):
    """Extract the style of compression from the url."""
    pattern = r"\.(zip|tar|tar\.gz)$"
    match = re.search(pattern, url)
    if match is not None:
        return match.group(1)

    return "zip"


class BaseDataset(torch.utils.data.Dataset):
    """The Base dataset."""

    def __init__(
        self, data_meta_catalog: DatasetMetaCatalog, phase: str, gt_extractor=None
    ):
        super().__init__()
        self.data_meta_catalog = data_meta_catalog
        self.data_name = data_meta_catalog["dataset_name"]
        self.phase = phase
        self.phase_data_path = data_meta_catalog.split_path[phase]
        catalog_filename = f"{phase}_data_catalog.json"
        self.data_catalog_path = os.path.join(
            data_meta_catalog.dataset_path, catalog_filename
        )

        self.customize_data_catalog = DatasetCatalog
        self.data_catalog: DatasetCatalog = None

        # Set the groundtruth extractor
        self.gt_extractor = gt_extractor

    def create_data_catalog(self):
        """Create the data catalog for the dataset."""
        raise NotImplementedError(
            "An implementation of create_data_catalog is required."
        )

    def configuration(self):
        """Configure the catalog of the dataset"""
        data_config = Config().data
        model_config = Config().model
        data_config = Config.items_to_dict(data_config._asdict())
        model_config = Config.items_to_dict(model_config._asdict())
        if os.path.exists(self.data_catalog_path):
            with open(self.data_catalog_path, "r", encoding="utf-8") as f:
                self.data_catalog = self.customize_data_catalog(**json.load(f))
            logging.info("Loaded data catalog from %s.", self.data_catalog_path)
        else:
            self.data_catalog = self.create_data_catalog()
            with open(self.data_catalog_path, "w", encoding="utf-8") as f:
                json.dump(self.data_catalog, f)
            logging.info("Created data catalog in %s.", self.data_catalog_path)

        # Set the extractor for the groundtruth extraction
        if self.gt_extractor is None and "extractor" in data_config:
            config = data_config["extractor"]

            extractor = get_extractor(
                data_name=data_config["data_name"],
                config=config,
            )
            if config["style"] == "llm":
                if "llm_config" not in config:
                    config["llm_config"] = model_config
                self.gt_extractor = extractor(
                    llm_model=define_model(config["llm_config"])
                )
            else:
                self.gt_extractor = extractor()

    def get_sample(self, idx):
        """Get one sample."""
        raise NotImplementedError("An implementation of get_sample is required.")

    def __getitem__(self, idx: int):
        """Get the sample from the given idx."""
        return self.get_sample(idx)

    def __len__(self):
        """Obtain the number of samples."""
        return self.data_catalog.data_statistics["num_samples"]

    def get_problem_sample_indexes(self, problem_name):
        """Get sample indexes of one problem."""

        return self.data_catalog.category_samples[problem_name]


class DataSource:
    """The Base datasource."""

    def __init__(self):
        # Extract the data information from the config file

        self.data_name = Config().data.data_name
        self.data_path = Config().data.data_path
        self.download_url = Config().data.download_url

        # Generate the data path and create one when necessary
        self.data_path = os.path.join(self.data_path, self.data_name)
        os.makedirs(self.data_path, exist_ok=True)

        self.meta_catalog_path = os.path.join(self.data_path, "meta_catalog.json")
        self.data_meta_catalog: DatasetMetaCatalog = None

        # Set the base dataset
        self.base_dataset = BaseDataset

    def create_meta_catalog(self):
        """Create the meta catalog for the dataset."""
        raise NotImplementedError(
            "An implementation of one specific dataset is required."
        )

    def configuration(self):
        """Configure the dataset."""

        if os.path.exists(self.meta_catalog_path):
            with open(self.meta_catalog_path, "r", encoding="utf-8") as f:
                self.data_meta_catalog = DatasetMetaCatalog(**json.load(f))
            logging.info("Loaded meta catalog from %s.", self.meta_catalog_path)
        else:
            self.data_meta_catalog = self.create_meta_catalog()
            with open(self.meta_catalog_path, "w", encoding="utf-8") as f:
                json.dump(self.data_meta_catalog, f)

    def download_data(self, phase):
        """Download the data for the current phase."""
        phase_data_path = self.data_meta_catalog["split_path"][phase]
        phase_data_path = (
            phase_data_path[0]
            if isinstance(phase_data_path, (list, tuple))
            else phase_data_path
        )
        if self.download_url is not None and not os.path.exists(phase_data_path):
            compression = extract_compression_style(self.download_url)
            filename = self.data_name + "." + compression
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

    def prepare_data(self, phase: str):
        """Prepare the source data."""
        # Configuration the dataset
        self.configuration()

        # Download the data when necessary
        self.download_data(phase)

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""

        self.prepare_data(phase)
        dataset = self.base_dataset(
            data_meta_catalog=self.data_meta_catalog, phase=phase
        )
        dataset.configuration()
        return dataset

    def get_train_set(self):
        """Obtains the training dataset."""
        phase = "train"
        return self.get_phase_dataset(phase)

    def get_test_set(self):
        """Obtains the validationidation dataset."""
        phase = "test"
        return self.get_phase_dataset(phase)
