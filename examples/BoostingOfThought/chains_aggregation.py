"""
Implementation of the global aggregation, in which users' update - branches 
are merged into a new one.

There are two options:
- leaf-depend aggregation
- level-wise aggregation
"""

from typing import List, Dict

from llmpebase.models.prompting.residual_tree_of_thoughts import ThoughtNode


def leaf_depend_aggregation(chains: Dict[int, List[ThoughtNode]]):
    """Aggregating the updates branches based on their evaluation score
    in the leaf."""
    # as these reasoning chains, i.e., branch
    users = list(chains.keys())

    best_leaf_score = 0
    best_user = users[0]
    for user_id in users:
        user_chain = chains[user_id]
        leaf_thought_node = user_chain[-1]
        if leaf_thought_node.thought_score > best_leaf_score:
            best_leaf_score = leaf_thought_node.thought_score
            best_user = user_id

    return chains[best_user]
