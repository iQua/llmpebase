"""
An interface of datasets
"""

from llmpebase.datasets.gsm8k import DataSource as gsm8k_datasource
from llmpebase.datasets.mmlu import DataSource as mmlu_datasource
from llmpebase.datasets.game24 import DataSource as game24_datasource
from llmpebase.datasets.math import DataSource as math_datasource
from llmpebase.datasets.bbh import DataSource as bbh_datasource
from llmpebase.datasets.theoremqa import DataSource as theoremqa_datasource

datasources = {
    "GSM8K": gsm8k_datasource,
    "MMLU": mmlu_datasource,
    "GameOf24": game24_datasource,
    "MATH": math_datasource,
    "BBH": bbh_datasource,
    "TheoremQA": theoremqa_datasource,
}


def define_dataset(data_config):
    """Define the datasets based on the config file."""
    data_name = data_config["data_name"]
    return datasources[data_name]()
