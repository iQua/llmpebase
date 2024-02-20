"""
The memorizer of the SnowBall.

The hardness to obtain useful experiences from step-wise reasoning and verification performed by LLM, as the guidance instead of misleading for the LLM is attributed to the following factors:

Main: Without human annotations, we cannot know which reasoning steps are correct while which reasoning steps are incorrect. Thus:

1. When the reasoning is incorrect, some verifications may report incorrect reasoning steps. But we do not know whether these verifications are trustworthy because the real incorrect reasoning step may exist in other steps whose verifications are correct.

2. When the reasoning is incorrect, verifications all report correct reasoning steps. We know that these verifications are not trustworthy. But we do not know which reasoning steps are incorrect and thus the corresponding verifications, which should report mistakes, are unknown.
"""

import os
import json
from typing import List, Tuple, Dict
from collections import defaultdict

import reasoner

from experience_db import LLMExperienceDBManager

from llmpebase.model.thought_structure import base
from llmpebase.dataset.data_generic import BaseQASample


class ExperienceLearner:
    """
    A base class to learn experiences from the reasoning result.

    Current reasoner is built upon the thought structure, making the learner
    enable to get reasoning steps and verification from it.
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

        if comparison:
            # Collect positive reasoning.
            return reasoning_path, "Positive"
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
            incorrect_path = (
                f"{reasoning_path}\n\nIncorrect steps are: {incorrect_steps}"
            )
            return incorrect_path, "Negative"

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
        verify_wrong = any(
            [score <= self.neg_eval_threshold for score in evaluation_scores]
        )
        if comparison:
            if verify_correct:
                # Collect positive verifications once step-wise verifications all report correct reasoning.
                return experience, "Positive"
            else:
                # Collect negative verifications once step-wise verifications report incorrect reasoning.
                return experience, "Negative"
        else:
            if verify_wrong:
                # Collect the positive verification one step-wise verification reports incorrect reasoning.
                return experience, "Positive"
            else:
                # Collect the negative verification once step-wise verifications all report correct reasoning.
                return experience, "Negative"

    def collect_experiences(
        self, defined_reasoner: reasoner.ChainThoughtReasoner, comparisons: List[bool]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Get the correct reasoning steps.

        :param defined_reasoner: The defined reasoner to get the reasoning experiences.
        :param comparisons: The comparison results of the solutions of reasoning steps.
        """
        # Initialize the reasoning experience
        reasoning_experiences = {"Positive": [], "Negative": []}
        verification_experiences = {"Positive": [], "Negative": []}
        # Get all solution chains
        # As the thought structure is a chain, we only get a single chain
        solution_chains = defined_reasoner.solution_extractor.extract_solution_chains(
            defined_reasoner.structure
        )
        # Collect experience from each chain
        for idx, chain in enumerate(solution_chains):
            # Extract the reasoning experience
            experience = self.extract_reasoning_experience(chain, comparisons[idx])
            reasoning_experiences[experience[-1]].append(experience[0])

            # Extract the verification experience
            experience = self.extract_verification_experience(chain, comparisons[idx])
            verification_experiences[experience[-1]].append(experience[0])

        return reasoning_experiences, verification_experiences


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

        # Experience holder to hold experiences
        self.experience_holder = list()

    def accumulate_experiences(
        self,
        defined_reasoner: reasoner.ChainThoughtReasoner,
        comparisons: List[bool],
    ):
        # Collect experiences for this sample
        reasoning_experiences, verification_experiences = (
            self.experience_learner.collect_experiences(defined_reasoner, comparisons)
        )
        self.experience_holder.append(
            {
                "Reasoning": reasoning_experiences,
                "Verification": verification_experiences,
            }
        )

    def compute_batch_accuracy(self, batch_comparisons: List[List[bool]]):
        """Compute the accuracy of the batch comparisons."""
        # Compute the accuracy
        n_correct = sum([sum(comparisons) for comparisons in batch_comparisons])
        n_total = sum([len(comparisons) for comparisons in batch_comparisons])
        accuracy = float(n_correct / n_total)

        return accuracy

    def create_memory_group_folder(self, field: str, category: str):
        """Create the memory group."""
        # Create the folder
        folder_path = f"{field}/{category}"
        group_path = os.path.join(self.memory_root_path, folder_path)
        os.makedirs(group_path, exist_ok=True)
        # Add the information to the memory group info
        if os.path.exists(self.memory_info_path):
            with open(self.memory_info_path, "r") as f:
                self.memory_info = json.load(f)
        if field not in self.memory_info or category not in self.memory_info[field]:
            self.memory_info[field][category] = {
                "Reasoning": {"Positive": 0, "Negative": 0},
                "Verification": {"Positive": 0, "Negative": 0},
            }

        return group_path

    def memory(
        self, batch_samples: List[BaseQASample], batch_comparisons: List[List[bool]]
    ):
        """Forward the batch samples and comparisons to memory experiences."""

        # Create the memory groups once there is no one
        group_paths = []
        for sample in batch_samples:
            sample_info = sample["auxiliary"]["sample_info"]
            field = sample_info["sample_field"]
            category = sample_info["sample_problem"]
            path = self.create_memory_group_folder(field, category)
            group_paths.append(path)

        # Insert questions to the vector database of txtai
        self.db_manager.record_questions(batch_samples)

        # Insert the experiences into the database
        self.db_manager.record_experiences(
            locations=group_paths,
            batch_samples=batch_samples,
            batch_experiences=self.experience_holder,
        )

        # Update the experience priority

        # Clean the current experience holder
        self.experience_holder = {"Reasoning": [], "Verification": []}

    def recall_experiences(
        self,
        sample_info: dict,
        question: str,
        experience_type: str = "positive",
        experience_mode: str = "reasoning",
        n_shots: int = 1,
    ):
        """Recall n_shots experiences based on the information of the sample."""
        # Get the field and category of the sample
        sample_field = sample_info["sample_field"]
        sample_problem = sample_info["sample_field"]

        # Get the database containing clusters of the group
        group_folder = self.group_creator.get_group_folder(
            root_path=self.memory_root_path,
            field=sample_field,
            category=sample_problem,
        )
        # Get the database of the corresponding experience
        experience_db = self.db_manager.search_question_cluster(
            question=question,
            group_folder=group_folder,
            experience_type=experience_type,
            experience_mode=experience_mode,
        )

        # Get the cluster that the question belongs to
