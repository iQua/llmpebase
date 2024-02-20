"""
Implementations of the solution extractor utilized to extract the all
solutions from the thought structure.
"""

from typing import List

from llmpebase.model.thought_structure import base


class SolutionExtractor:
    """A base extractor to extract solutions from the thought structure."""

    def extract_solution_chains(
        self,
        structure: base.BaseThoughtStructure,
    ) -> List[List[base.BasicNode]]:
        """
        Extract all solutions from the thought structure.

        By default, a reasoning path toward a solution is the path
        from the root node to a sink node.
        """

        root_node = structure.root

        solution_chains = []
        # Get stop nodes as the stop node presents the end of a
        # reasoning chain
        sink_nodes = structure.get_sink_nodes()
        for node in sink_nodes:
            # By default, we extract the thought chain with the
            # highest evaluation score
            thought_chain = structure.get_node_path(
                src_node_id=root_node.identity, dst_node_id=node.identity
            )
            solution_chains.append(thought_chain)

        return solution_chains
