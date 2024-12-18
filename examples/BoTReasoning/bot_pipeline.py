"""
The pipeline for the BoT reasoning process.
"""

from llmpebase.pipeline import Pipeline
from torch.utils.data import DataLoader, SubsetRandomSampler


class BoTPipeline(Pipeline):
    """The pipeline for the BoT reasoning process."""

    def load_data(self):
        """Redesign the data loading."""
        super().load_data()

        sample = self.testset[0]

        sample_dataset = sample["auxiliary"]["sample_info"]["sample_dataset"]
        if "gameof24" in sample_dataset.lower():
            # Indices of the last 100 samples
            indices = list(range(900, 1000))

            # Create the SubsetRandomSampler
            sampler = SubsetRandomSampler(indices)
            # Add the sampler to the dataloader
            self.testset_loader = DataLoader(
                self.trainset,
                batch_size=1,
                shuffle=False,
                sampler=sampler,
                collate_fn=lambda batch: batch[0],
            )
