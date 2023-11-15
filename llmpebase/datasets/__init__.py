"""
An interface of datasets
"""

from llmpebase.datasets.gsm8k import (
    DataSource as gsm8k_datasource,
)

from llmpebase.datasets.mmlu import DataSource as mmlu_datasource

from llmpebase.datasets.game24 import DataSource as game24_datasource

datasources = {
    "GSM8K": gsm8k_datasource,
    "MMLU": mmlu_datasource,
    "GameOf24": game24_datasource,
}


def define_dataset(data_config):
    """Define the datasets based on the config file."""
    data_name = data_config["data_name"]
    return datasources[data_name]()
