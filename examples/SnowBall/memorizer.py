"""
The memorizer of the SnowBall.

The hardness to obtain useful experiences from step-wise reasoning and verification performed by LLM, as the guidance instead of misleading for the LLM is attributed to the following factors:

Main: Without human annotations, we cannot know which reasoning steps are correct while which reasoning steps are incorrect. Thus:

1. When the reasoning is incorrect, some verifications may report incorrect reasoning steps. But we do not know whether these verifications are trustworthy because the real incorrect reasoning step may exist in other steps whose verifications are correct.

2. When the reasoning is incorrect, verifications all report correct reasoning steps. We know that these verifications are not trustworthy. But we do not know which reasoning steps are incorrect and thus the corresponding verifications, which should report mistakes, are unknown.

There are two styles of experiences are:
 - Reasoning
 - Verification
"""

import os
import json
from typing import List, Dict
from collections import defaultdict

import pandas as pd

from experience_db import LLMExperienceDBManager
from experience_generic import BaseExperience

from llmpebase.model.thought_structure import base
from llmpebase.dataset.data_generic import BaseQASample


class ExperienceLearner:
    """
    A base class to learn experiences from the reasoning result.

    The learner mainly aims to .
    """

    def __init__(self, model_config: dict) -> None:
        experience_config = model_config["experience_memorization"]
        self.neg_eval_threshold = experience_config["negative_evaluation_threshold"]

    def extract_reasoning_experience(
        self, reasoning_chain: List[base.BasicNode], comparison: bool
    ):
        """Extract the experience for a chain of reasoning."""
        # Get the reasoning path
        reasoning_steps = [node.thought for node in reasoning_chain[1:]]

        evaluation_scores = [
            float(node.evaluation_score) for node in reasoning_chain[1:]
        ]

        reasoning_path = "\n".join(reasoning_steps)
        mode = ""
        if comparison:
            # Collect positive reasoning.
            experience = reasoning_path
            mode = "Positive"
        else:
            # Collect the negative reasoning.
            # When the reasoning is incorrect, we assume that all verifications are
            # reasonable and thus are able to get the incorrect steps based on
            # verifications
            incorrect_steps = [
                idx + 1
                for idx, score in enumerate(evaluation_scores)
                if score <= self.neg_eval_threshold
            ]
            incorrect_steps = ",".join(incorrect_steps)
            experience = f"{reasoning_path}\n\nIncorrect steps are: {incorrect_steps}"
            mode = "Negative"

        return BaseExperience(
            experience=experience,
            style="Reasoning",
            mode=mode,
            priority_score=0,
        )

    def extract_verification_experience(
        self, reasoning_chain: List[base.BasicNode], comparison: bool
    ):
        """
        Extract the verification experience from a chain of verification.

        A positive verification is obtained when
            1. all verifications report correct reasoning and the reasoning
            is correct.
            2. some verifications report incorrect reasoning and the reasoning
            is incorrect.
        """
        # Get the reasoning path
        reasoning_steps = [node.thought for node in reasoning_chain[1:]]
        evaluation_contents = [node.evaluation_content for node in reasoning_chain[1:]]

        evaluation_scores = [
            float(node.evaluation_score) for node in reasoning_chain[1:]
        ]
        reasoning_path = "\n".join(reasoning_steps)
        # We do not add \n here as the evaluation contents are already separated by \n\n
        evaluation_contents = "".join(evaluation_contents)

        experience = f"{reasoning_path}\n\nVerifications of these reasoning steps are:\n{evaluation_contents}"

        verify_correct = all(
            [score > self.neg_eval_threshold for score in evaluation_scores]
        )

        mode = ""
        if comparison and verify_correct:
            mode = "Positive"
        elif comparison and not verify_correct:
            mode = "Negative"
        elif not comparison and not verify_correct:
            mode = "Positive"
        elif not comparison and verify_correct:
            mode = "Negative"

        return BaseExperience(
            experience=experience,
            style="Verification",
            mode=mode,
            priority_score=0,
        )

    def remove_duplicate_experience(
        self, experiences: List[BaseExperience]
    ) -> List[BaseExperience]:
        """
        Remove the duplicate experiences.

        This is the most basic way to remove the duplicate experiences. Further
        improvement can be made.
        For example, one can introduce the LLM model to measure the similarity to facilitate the removal.
        """
        # Convert the list of dictionaries into a pandas DataFrame
        exp_df = pd.DataFrame(experiences)
        # Remove duplicates based on the 'experience' column
        unique_exps = exp_df.drop_duplicates(subset=["experience"]).to_dict("records")

        return [BaseExperience(**exp) for exp in unique_exps]

    def collect_experiences(
        self, solution_chains: List[List[base.BasicNode]], comparisons: List[bool]
    ) -> Dict[str, List[BaseExperience]]:
        """
        Get the correct reasoning steps.

        :param defined_reasoner: The defined reasoner to get the reasoning experiences.
        :param comparisons: The comparison results of the solutions of reasoning steps.
        """
        # Initialize the reasoning experience
        reasoning_experiences = []
        verification_experiences = []

        # Collect experience from each chain
        for idx, chain in enumerate(solution_chains):
            chain_cmp = comparisons[idx]
            # Extract the reasoning experience
            experience = self.extract_reasoning_experience(chain, chain_cmp)
            reasoning_experiences.append(experience)

            # Extract the verification experience
            experience = self.extract_verification_experience(chain, chain_cmp)
            verification_experiences.append(experience)

        # Remove the duplicate experiences
        reasoning_experiences = self.remove_duplicate_experience(reasoning_experiences)
        verification_experiences = self.remove_duplicate_experience(
            verification_experiences
        )

        return {
            "Reasoning": reasoning_experiences,
            "Verification": verification_experiences,
        }


class LLMMemorizer:
    """
    The memorizer of the SnowBall to cluster, store and retrieve the experiences.
    """

    def __init__(
        self,
        model_config: dict,
        log_config: dict,
        db_manager: LLMExperienceDBManager = None,
    ):

        # The root path for the memory
        self.memory_root_path = os.path.join(
            log_config["result_path"], "ExperienceMemory"
        )
        os.makedirs(self.memory_root_path, exist_ok=True)

        # There should be a memory group file under the memory root path
        # to record the current groups and the corresponding number of
        # experiences in each group.
        self.memory_info_path = os.path.join(self.memory_root_path, "memory_info.json")
        # Get the group information
        self.memory_info = defaultdict(dict)

        # Define the experience learner
        self.experience_learner = ExperienceLearner(model_config)

        # The organizer of the experience database
        # Unless it is provided, the db organizer cannot be defined
        # at this stage as the experiences, groups, and clusters are not
        # defined yet.
        self.db_manager = (
            db_manager
            if db_manager is not None
            else LLMExperienceDBManager(model_config)
        )

    def create_memory_group(self, field: str, category: str):
        """
        Create the memory group by making new folder and recording
        the information to the file.
        """
        # Create the folder
        folder_path = f"{field}/{category}"
        group_path = os.path.join(self.memory_root_path, folder_path)
        os.makedirs(group_path, exist_ok=True)
        # Add the information to the memory group info
        if os.path.exists(self.memory_info_path):
            with open(self.memory_info_path, "r", encoding="utf-8") as f:
                self.memory_info = json.load(f)
        if field not in self.memory_info or category not in self.memory_info[field]:
            self.memory_info[field][category] = {
                "Reasoning": {"Positive": 0, "Negative": 0},
                "Verification": {"Positive": 0, "Negative": 0},
            }
        with open(self.memory_info_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_info, f)

        return group_path

    def compute_accuracy(self, experiences: Dict[str, List[str]]):
        """Compute the accuracy of the experiences."""
        # Compute the priority of the experiences
        n_correct_reasoning = len(experiences["Reasoning"]["Positive"])
        n_wrong_reasoning = len(experiences["Reasoning"]["Negative"])
        n_reasoning_total = n_correct_reasoning + n_wrong_reasoning
        n_correct_verify = len(experiences["Verification"]["Positive"])
        n_wrong_verify = len(experiences["Verification"]["Negative"])
        n_verify_total = n_correct_verify + n_wrong_verify
        return {
            "Reasoning": n_correct_reasoning / n_reasoning_total,
            "Verification": n_correct_verify / n_verify_total,
        }

    def memory(
        self,
        sample: BaseQASample,
        solution_chains: List[List[base.BasicNode]],
        comparisons: List[bool],
    ):
        """Forward the batch samples and comparisons to memory experiences."""
        experiences = self.experience_learner.collect_experiences(
            solution_chains=solution_chains, comparisons=comparisons
        )
        # Create the memory group once there is no one
        sample_info = sample["auxiliary"]["sample_info"]
        field = sample_info["sample_field"]
        category = sample_info["sample_problem"]
        path = self.create_memory_group(field, category)

        # # Insert questions to the vector database of txtai
        q_db_counts = self.db_manager.record_questions(location=path, samples=[sample])

        # Insert the experiences into the database
        # Get the counts of the experiences
        db_counts = self.db_manager.record_experiences(
            location=path, sample=sample, experiences=experiences
        )

        # Update the question information
        self.memory_info[field][category].update(q_db_counts)
        # Update the experience information
        self.memory_info[field][category].update(db_counts)

        # Record the memory info into file
        with open(self.memory_info_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_info, f)
