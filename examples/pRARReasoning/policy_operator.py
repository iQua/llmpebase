"""
A series of classes to operate the policy.
We should know that the content of the policy is a tree, thereby being
included in a folder.
"""

import os
import logging

import policy_tree

from llmpebase.model.thought_structure import base

from llmpebase.utils import tools


class PolicyOperator:
    """A operator to load, save, and organize the policies."""

    def __init__(self, logging_config: dict):
        self.logging_config = logging_config
        self.result_path = logging_config["result_path"]

        self.base_path = f"{self.result_path}/PolicyBase"
        self.policy_info = {}

    def get_policy_name(self, problem_category):
        """Get the policy name."""
        problem_category = tools.format_term(problem_category)
        return f"Policy-{problem_category}"

    def get_policy_path(self, problem_category: str):
        """Get the path of the policy."""
        problem_category = self.get_policy_name(problem_category)
        return os.path.join(self.base_path, problem_category)

    def load_policy_tree(self, sample_info: dict) -> policy_tree.PolicyTree:
        """Load the policy from the given category."""
        sample_field = sample_info["sample_field"]
        sample_dataset = sample_info["sample_dataset"]
        problem_category = sample_info["sample_problem"]
        category_policy_name = self.get_policy_name(problem_category)
        policy_path = os.path.join(self.base_path, category_policy_name)

        # Create an empty policy tree
        tree = policy_tree.PolicyTree(
            logging_config=self.logging_config, visualizer=None
        )

        if category_policy_name in self.policy_info or os.path.exists(policy_path):
            # Ensure that the policy information is added
            self.policy_info[category_policy_name] = policy_path
            tree.load_structure(policy_path)
            if tree.root.task_info != sample_field:
                raise ValueError(
                    f"The task info of the sample {sample_field} is not matched with the policy tree {tree.root.task_info}."
                )

            logging.info(
                "Loaded the Policy tree %s from %s.", category_policy_name, policy_path
            )

            # Add the dataset of the sample to the tree root
            if sample_dataset not in tree.root.auxiliary["VisitedDatasets"]:
                tree.root.auxiliary["VisitedDatasets"].append(sample_dataset)
        else:
            self.policy_info[category_policy_name] = policy_path
            os.makedirs(policy_path, exist_ok=True)

            tree.construct_root(task_info=sample_field, category_name=problem_category)
            # Add the dataset of the sample to the tree root
            tree.root.auxiliary["VisitedDatasets"] = [sample_dataset]

            logging.info(
                "Created a new policy tree %s under %s.",
                category_policy_name,
                policy_path,
            )

            tree.save_structure(
                foldername=category_policy_name, location=self.base_path
            )

        return tree

    def save_policy_tree(self, policy_tree: base.BaseStructure, sample_info: dict):
        """Save the policy tree."""

        problem_category = sample_info["sample_problem"]
        category_policy_name = self.get_policy_name(problem_category)

        policy_tree.save_path = self.base_path
        policy_tree.save_foldername = category_policy_name
        policy_tree.save_structure(
            location=self.base_path, foldername=category_policy_name
        )

        if category_policy_name not in self.policy_info:
            logging.info(
                "New Policy tree %s is added to %s.",
                category_policy_name,
                self.base_path,
            )
            self.policy_info[category_policy_name] = os.path.join(
                self.base_path, category_policy_name
            )
        else:
            logging.info(
                "Policy tree %s under %s has been updated.",
                category_policy_name,
                self.base_path,
            )
