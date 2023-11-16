"""
Implementation of the global aggregation, in which users' update - branches 
are merged into a new one. There are two strategies:
- Best-First Aggregation
- Greedy Aggregation
"""

from typing import List, Dict

from llmpebase.models.prompting.tree_thoughts import ThoughtNode


class ReasoningChainAggregator:
    """A base class towards aggregating multiple different reasoning chains"""

    def __init__(self, model_config: dict) -> None:
        self.model_config = model_config
        self.aggregation_type = model_config["aggregation_type"]

        assert self.aggregation_type in ["best_first", "greedy"]

    @staticmethod
    def best_first_aggregation(chains: Dict[int, List[ThoughtNode]]):
        """Aggregate the reasoning chains by selecting the best chain."""
        # Get the id of each chain
        chain_ids = list(chains.keys())

        best_id = 0
        best_score = 0
        for chain_id in chain_ids:
            chain = chains[chain_id]
            chain_score = sum([float(node.thought_score) for node in chain])
            if chain_score > best_score:
                best_score = chain_score
                best_id = chain_id

        return chains[best_id]

    @staticmethod
    def greedy_aggregation(chains: Dict[int, List[ThoughtNode]]):
        """Aggregate the reasoning chains by visting the chains from the root to the leaf."""
        chain_ids = list(chains.keys())
        # Get the root of the chains as they start from the same
        root_node = chains[chain_ids[0]][0]
        # The aggregated chain containing multiple reaonsing steps
        aggregated_chain = [root_node]

        def greedy_search(step_idx, aggregated_chain):
            step_nodes = [
                chains[chain_id][step_idx]
                for chain_id in chain_ids
                if len(chains[chain_id]) > step_idx
            ]
            if len(step_nodes) == 0:
                return aggregated_chain

            best_node = max(step_nodes, key=lambda node: node.thought_score)
            aggregated_chain.append(best_node)
            return greedy_search(step_idx + 1, aggregated_chain)

        aggregated_chain = greedy_search(1, aggregated_chain)
        return aggregated_chain

    def perform_aggregation(self, chains: Dict[int, List[ThoughtNode]]):
        """Perform the aggregation for the update chains."""
        if self.aggregation_type == "best_first":
            return self.best_first_aggregation(chains)

        return self.greedy_aggregation(chains)
