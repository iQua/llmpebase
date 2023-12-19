"""
An implementation of the chain thought structure in which thoughts of 
one reasoning process are organized as a chain.
"""

import torch

from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer


class ChainThoughtStructure(base.BaseThoughtStructure):
    """
    A chain thought structure to perform a linear reasoning process in
    which reasoning steps are sequentially interconnected.
    """

    def build_structure(
        self,
        thought_model: torch.nn.Module,
        visualizer: BasicStructureVisualizer = None,
        **kwargs,
    ):
        """Build the thought chain structure."""
        super().build_structure(
            thought_model=thought_model, visualizer=visualizer, **kwargs
        )
