""" 
The datasource inference for the Game of 24 dataset.
"""
import os
import time
from typing import List

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

from llmpebase.dataset import base
from llmpebase.dataset.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)


class GameOf24Dataset(base.BaseDataset):
    """
    An interface for the GameOf24 dataset.
    """

    def create_data_catalog(self):
        data_frame = pd.read_csv(self.phase_data_path)
        n_items = data_frame.shape[0]

        collected_items = [
            BaseQASampleInfo(
                sample_id=data_frame["Rank"].iloc[i].item(),
                sample_problem="Algebra",
                sample_filepath=self.phase_data_path,
            )
            for i in range(n_items)
        ]
        return DatasetCatalog(
            data_phase=self.phase,
            data_samples=collected_items,
            data_statistics=DatasetStatistics(num_samples=n_items),
        )

    def get_sample(self, idx):
        """Get one sample."""
        sample_path = self.data_catalog.data_samples[idx]["sample_filepath"]
        sample_problem = self.data_catalog.data_samples[idx]["sample_problem"]
        data_frame = pd.read_csv(sample_path)
        return BaseQASample(
            question=data_frame["Puzzles"].iloc[idx],
            answer="",
            conclusion="",
            groundtruth=24,
            auxiliary={
                "solved_rate": data_frame["Solved rate"].iloc[idx],
                "AMT": data_frame["AMT (s)"].iloc[idx],
                "1_sigma_Mean": data_frame["1-sigma Mean (s)"].iloc[idx],
                "1_sigma_STD": data_frame["1-sigma STD (s)"].iloc[idx],
                "sample_problem": sample_problem,
            },
        )


class DataSource(base.DataSource):
    """The GameOf24 dataset."""

    def __init__(self):
        super().__init__()

        self.base_dataset = GameOf24Dataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="GameOf24",
            task_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "24.csv"),
                "test": os.path.join(self.data_path, "24.csv"),
                "val": os.path.join(self.data_path, "24.csv"),
            },
        )


class GameOf24Solver:
    """A base solver to solve the Game of 24 problem."""

    def __init__(self) -> None:
        # The website where the solver is hosted
        self.solver_website = "http://24solver.us-west-2.elasticbeanstalk.com/"

    def solve(self, numbers: List[int]):
        """Solve the problem by contacting with the solver website."""
        # A webdriver instance
        driver = webdriver.Chrome()
        # Open the webpage
        driver.get(self.solver_website)
        # Wait for the page to load
        time.sleep(3)
        # Fill in the form fields
        driver.find_element(By.NAME, "n1").send_keys(numbers[0])
        driver.find_element(By.NAME, "n2").send_keys(numbers[1])
        driver.find_element(By.NAME, "n3").send_keys(numbers[2])
        driver.find_element(By.NAME, "n4").send_keys(numbers[3])

        # Submit the form
        driver.find_element(By.NAME, "n1").submit()

        # Wait for the page to load results
        time.sleep(5)

        # Extract the solutions from the website
        # A list of web elements
        responses = driver.find_elements(By.CSS_SELECTOR, "ul > li > span")
        solutions = [item.text for item in responses]

        # Close the webdriver
        driver.quit()

        return solutions

    def solve_row(self, row: pd.Series):
        """Solve the problem present in each row of the data."""
        # Get numbers
        numbers = [int(item) for item in row["Puzzles"].split(" ")]
        # Solve the problem
        solutions = self.solve(numbers)
        # Convert the solutions to string
        solutions_str = ";".join(solutions)

        return solutions_str

    def get_solutions(self, data_filepath: str):
        """Get solution for the game of 24 dataset in one files.

        For example:
            solver = game24.GameOf24Solver()
            solver.get_solutions(data_filepath="./ICLR/data/GameOf24/new_24.csv")
        """
        data_frame = pd.read_csv(data_filepath)

        data_frame["Solutions"] = data_frame.apply(self.solve_row, axis=1)

        data_frame.to_csv(data_filepath, index=False)
