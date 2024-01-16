"""
Implementations of the chain extractor utilized to extract the thought chain
from the thought structure.
"""

from typing import List

from llmpebase.model.thought_structure import base


class SolutionExtractor:
    """A base extractor to extract the thought chain from the thought structure."""

    def extract_solution_chains(
        self,
        structure: base.BaseThoughtStructure,
    ) -> List[List[base.BasicNode]]:
        """
        Extract the solution from the thought structure.

        This is a demo implementation that only returns the thought chain
        with the highest evaluation score.
        """

        root_node = structure.root

        # Get stop nodes as the stop node presents the end of a
        # reasoning chain
        sink_nodes = structure.get_sink_nodes()

        max_scores = 0
        best_chain = None
        for node in sink_nodes:
            # By default, we extract the thought chain with the
            # highest evaluation score
            thought_chain = structure.get_node_path(
                src_node_id=root_node.identity, dst_node_id=node.identity
            )
            # Get the sum score of the thought chain
            chain_score = sum(structure.get_path_scores(thought_chain))
            if chain_score > max_scores:
                max_scores = chain_score
                best_chain = thought_chain

        return [best_chain]
