"""
Get the group of experiences based on the problem category 
provided in the dataset or the question embedding.

There should be three levels of experiences:

- Main problem field: Where the problems come from. For example, MATH, EECS, Physics, etc.
    - Problem category level: The experiences are grouped based on the problem category
    such as algebra, calculus, etc.
        - Cluster level: The experiences are grouped based on the question embedding,
        such as 0, 1, 2, etc.

where the problem category is based on the dataset.
"""

import os
import json
import glob


class ExperienceGroupCreator:
    """
    An creator to create the experience group based on the category name and the
    question embedding.
    """

    def __init__(self) -> None:

        self.support_datasets = ["GSM8K", "SVAMP", "AQUA", "MATH", "TheoremQA"]

        # Set the group of the experiences
        # It should be field/category/
        self.groups = {}

    def create_groups(self, data_root: str):
        """Create groups from all datasets under data_root."""
        category_info_dict = {}
        # Pattern to match 'test_data_catalog.json' files directly under any subdirectory of 'data_root'
        pattern = os.path.join(data_root, "*", "test_data_catalog.json")
        # Use glob to find all files that match the pattern
        for json_file_path in glob.glob(pattern):
            # Extract dataset name from the path (assuming it's the directory name just before the JSON file)
            dataset_name = os.path.basename(os.path.dirname(json_file_path))
            # Open and read the JSON file
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                # Extract 'category_info' from the file
                category_info = data.get("problem_categories", {})

                # Add the extracted info to the dictionary with the dataset name as the key
                category_info_dict[dataset_name] = category_info

        # Collect the category as the groups
        for data_name in self.support_datasets:
            category_info = category_info_dict[data_name]

            for field in category_info:
                field_categories = category_info[field]
                # Create new category with an empty dict of clusters
                field_categories = (
                    field_categories
                    if isinstance(field_categories, list)
                    else list(field_categories.keys())
                )
                if field not in self.groups:
                    self.groups[field] = field_categories
                else:
                    self.groups[field].extend(field_categories)

    def create_group_folders(self, root_path: str):
        """Create folders for the groups"""
        for field, categories in self.groups.items():
            for category in categories:
                os.makedirs(os.path.join(root_path, field, category), exist_ok=True)

    def get_group_folder(self, root_path, field, category):
        """Get the database path of the target group."""
        return os.path.join(root_path, field, category)

    def is_group_exist(self, group_identity):
        """Whether the group exists."""
        parts = group_identity.split("-")
        assert len(parts) == 3
        field, category, _ = parts

        assert field in self.groups
        assert category in self.groups[field]


if __name__ == "__main__":
    creator = ExperienceGroupCreator()

    creator.create_groups(data_root="./LLMPE/data")

    from txtai.embeddings import Embeddings

    # Create embeddings instance
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2", "backend": "numpy"}
    )

    # Index data
    embeddings.index((x, text, None) for x, text in enumerate(dataset))

    embeddings.search("red sox")
