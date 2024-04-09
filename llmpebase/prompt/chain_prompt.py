"""
Implementation of prompts for the chain thought.
Generally, the category of the comment on the chain thought can be divided 
into two parts, including the outcome and process.

For example, BoT uses the outcome comment to provide feedback on the reasoning chain. TR uses the process comment to provide feedback on the reasoning process.

"""

from llmpebase.prompt.generic import BasicThoughtPromptFormat


class ChainOutcomeCommentPrompts:
    """A base class to organize the prompt of the chain thought comments."""

    chain_start_flag: str = """<Chain>"""
    chain_end_flag: str = """<\\Chain>"""

    # Format of the feedback prompt in which
    # For head:
    # first {} is the category of the question,
    # second {} is the question,
    # third {} is the chain_start_flag,
    # forth {} is the reasoning_chain,
    # fifth {} is the chain_end_flag
    # For content:
    # first {} is the chain_start_flag,
    # second {} is the chain_end_flag
    feedback_prompt = BasicThoughtPromptFormat(
        head="Comment on the given reasoning chain for addressing the question of {}.\n\n{}\nReasoning chain:\n{}\n{}\n{}\n\n\n",
        content="Please review the reasoning steps between {} and {} and thus provide the analysis for each step, especially the one with errors.\n",
        target="\n",
        notice="",
        tail="",
        prompt="",
    )


class Gameof24ChainOutcomeCommentPrompts(ChainOutcomeCommentPrompts):
    """A base class to organize the prompt of the chain thought comments."""

    feedback_prompt = BasicThoughtPromptFormat(
        head="Comment on the given reasoning chain for addressing the question of Game of 24.\n\n{}\nReasoning chain:\n{}\n{}\n{}\n\n\n",
        content="Please review the reasoning steps between {} and {} and thus provide the analysis for each step, especially the one with errors and when the New Set of Step 3 is not 24.\n",
        target="\n",
        notice="",
        tail="",
        prompt="",
    )
