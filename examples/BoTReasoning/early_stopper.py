"""
The implementation of early stopping.
"""

import logging
from typing import List


from llmpebase.model.thought_structure import base


def stop_via_gameof24(
    solution_strs: List[str], solution_chains: List[List[base.BasicNode]]
):
    """Perform the early stopping for the Game of 24 dataset."""
    is_stop = False
    final_sol_idx = None
    for idx, solution_str in enumerate(solution_strs):
        if r"{24}" in solution_str or "24." in solution_str:
            final_sol_idx = idx
            is_stop = True
            break

    return (is_stop, final_sol_idx)


def stop_via_evaluation(
    solution_strs: List[str],
    solution_chains: List[List[base.BasicNode]],
    threshold_score: float = 0.9,
):
    """
    Perform the early stopping based on the evaluations.
    When the evaluation scores of chain's nodes are all larger than `threshold_score`, then stop.
    """
    is_stop = False
    final_sol_idx = None
    for sol_idx, solution_str in enumerate(solution_strs):
        is_confident = all(
            [
                node.evaluation_score > threshold_score
                for node in solution_chains[sol_idx][1:]
            ]
        )
        if is_confident and ("final" in solution_str and "solution" in solution_str):
            is_stop = True
            final_sol_idx = sol_idx
            break

    return (is_stop, final_sol_idx)


def stop_via_comment(comment_feedback: str):
    """Perform the early stopping based on the comment."""
    is_stop = False
    comment_feedback = comment_feedback.lower()
    if not (
        "invalid" in comment_feedback
        or "mistake" in comment_feedback
        or "error" in comment_feedback
        or "wrong" in comment_feedback
        or "incorrect" in comment_feedback
        or "redundant" in comment_feedback
        or "unnecessary" in comment_feedback
    ):
        is_stop = True
    return is_stop


def get(sample_info: dict):
    """Get the information of the sample."""
    sample_dataset = sample_info["sample_dataset"]
    if "GameOf24" in sample_dataset:
        logging.info("Early stop for Game of 24.")
        return stop_via_gameof24
    else:
        logging.info("Early stop via evaluation.")
        return stop_via_evaluation
