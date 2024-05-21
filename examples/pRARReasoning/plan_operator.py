"""
A series of classes to operate the plan.
We should know that the content of the plan is a tree, thereby being
included in a folder.
"""

import os
import logging

import plan_tree

from llmpebase.model.thought_structure import base

from llmpebase.utils import tools


class PlanOperator:
    """A operator to load, save, and organize the policies."""

    def __init__(self, logging_config: dict):
        self.logging_config = logging_config
        self.result_path = logging_config["result_path"]

        self.base_path = f"{self.result_path}/PlanBase"
        self.plan_info = {}

    def get_plan_name(self, problem_category):
        """Get the plan name."""
        problem_category = tools.format_term(problem_category)
        return f"Plan-{problem_category}"

    def get_plan_path(self, problem_category: str):
        """Get the path of the plan."""
        problem_category = self.get_plan_name(problem_category)
        return os.path.join(self.base_path, problem_category)

    def load_plan_tree(self, sample_info: dict) -> plan_tree.PlanTree:
        """Load the plan from the given category."""
        sample_field = sample_info["sample_field"]
        sample_dataset = sample_info["sample_dataset"]
        problem_category = sample_info["sample_problem"]
        category_plan_name = self.get_plan_name(problem_category)
        plan_path = os.path.join(self.base_path, category_plan_name)

        # Create an empty plan tree
        tree = plan_tree.PlanTree(logging_config=self.logging_config, visualizer=None)

        if category_plan_name in self.plan_info or os.path.exists(plan_path):
            # Ensure that the plan information is added
            self.plan_info[category_plan_name] = plan_path
            tree.load_structure(plan_path)
            if tree.root.task_info != sample_field:
                raise ValueError(
                    f"The task info of the sample {sample_field} is not matched with the plan tree {tree.root.task_info}."
                )

            logging.info(
                "Loaded the Plan tree %s from %s.", category_plan_name, plan_path
            )

            # Add the dataset of the sample to the tree root
            if sample_dataset not in tree.root.auxiliary["VisitedDatasets"]:
                tree.root.auxiliary["VisitedDatasets"].append(sample_dataset)
        else:
            self.plan_info[category_plan_name] = plan_path
            os.makedirs(plan_path, exist_ok=True)

            tree.construct_root(task_info=sample_field, category_name=problem_category)
            # Add the dataset of the sample to the tree root
            tree.root.auxiliary["VisitedDatasets"] = [sample_dataset]

            logging.info(
                "Created a new plan tree %s under %s.",
                category_plan_name,
                plan_path,
            )

            tree.save_structure(foldername=category_plan_name, location=self.base_path)

        return tree

    def save_plan_tree(self, plan_tree: base.BaseStructure, sample_info: dict):
        """Save the plan tree."""

        problem_category = sample_info["sample_problem"]
        category_plan_name = self.get_plan_name(problem_category)

        plan_tree.save_path = self.base_path
        plan_tree.save_foldername = category_plan_name
        plan_tree.save_structure(location=self.base_path, foldername=category_plan_name)

        if category_plan_name not in self.plan_info:
            logging.info(
                "New Plan tree %s is added to %s.",
                category_plan_name,
                self.base_path,
            )
            self.plan_info[category_plan_name] = os.path.join(
                self.base_path, category_plan_name
            )
        else:
            logging.info(
                "Plan tree %s under %s has been updated.",
                category_plan_name,
                self.base_path,
            )
