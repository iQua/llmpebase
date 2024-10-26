"""
Implementation of the global aggregation, in which users' update - branches 
are merged into a new one. There are two strategies:
- Best-First Aggregation
- Greedy Aggregation
"""

import json
import logging
from typing import List, Dict

from llmpebase.model.thought_structure.structure_generic import BasicNode


def get_chain_path(chain: List[BasicNode]) -> str:
    """Get the chain path from the node to the root."""
    start_node = chain[0]
    end_node = chain[-1]
    return f"N-{start_node.identity} S-{start_node.step_idx} -> N-{end_node.identity} S-{end_node.step_idx}"


class ReasoningChainAggregator:
    """A base class towards aggregating multiple different reasoning chains"""

    def __init__(self, logging_config: dict, model_config: dict) -> None:

        self.model_config = model_config
        self.aggregation_type = model_config["bot_settings"]["aggregation_type"]

        assert self.aggregation_type in ["best_first", "greedy"]
        self.save_path = logging_config["result_path"]

        self.aggregation_state = {}

    @staticmethod
    def best_first_aggregation(
        structure_chains: Dict[int, List[List[BasicNode]]]
    ) -> Dict[str, any]:
        """Aggregate the reasoning chains by selecting the best chain."""

        # Get the best chain from all chains
        aggregated_results = {}
        best_score = 0
        for structure_id, chains in structure_chains.items():
            chain_scores = [
                sum([float(node.evaluation_score) for node in chain[1:]])
                for chain in chains
            ]
            chain_scores = [0] if len(chain_scores) == 0 else chain_scores
            max_value = max(chain_scores)
            if max_value > best_score:
                chain_idx = chain_scores.index(max_value)
                best_chain = chains[chain_idx]
                aggregated_results["structure_id"] = structure_id
                aggregated_results["chain_idx"] = chain_idx
                aggregated_results["aggregated_chain"] = best_chain
                aggregated_results["best_score"] = max_value

        return aggregated_results

    @staticmethod
    def greedy_aggregation(chains: Dict[int, List[List[BasicNode]]]):
        """Aggregate the reasoning chains by visiting the chains from the root to the leaf."""
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

            best_node = max(step_nodes, key=lambda node: node.evaluation_score)
            aggregated_chain.append(best_node)
            return greedy_search(step_idx + 1, aggregated_chain)

        aggregated_chain = greedy_search(1, aggregated_chain)
        return aggregated_chain

    def perform_aggregation(
        self, structure_chains: Dict[int, List[List[BasicNode]]]
    ) -> List[BasicNode]:
        """Perform the aggregation for the update chains."""

        logging.info(
            """
            Aggregating (%s) %d structures, individual containing %s chains .
            """,
            self.aggregation_type,
            len(structure_chains),
            [len(chains) for chains in structure_chains.values()],
        )

        self.aggregation_state = {}

        # Record candidate chains for structures
        self.aggregation_state["candidate_chains"] = {
            structure_id: [get_chain_path(chain) for chain in chains]
            for structure_id, chains in structure_chains.items()
        }
        if self.aggregation_type == "best_first":
            aggregated_results = self.best_first_aggregation(structure_chains)
        else:
            aggregated_results = self.greedy_aggregation(structure_chains)

        # Record the aggregation state
        self.aggregation_state.update(
            {
                "aggregation_type": self.aggregation_type,
                "aggregated_results": aggregated_results,
            }
        )
        return aggregated_results["aggregated_chain"]

    def save_state(self, location: str, file_name: str):
        """Save the state of the aggregator."""
        location = f"{self.save_path}/{location}"
        with open(f"{location}/{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(self.aggregation_state, f)
