"""
Implementation of prompts for the thought.
Current prompts mainly include four parts:

- Generation: prompts for generating the next thought in the chain.
- Evaluation: prompts for evaluating the thought in the chain.
- Similarity: prompts for measuring the similarity between two thoughts in the chain.

It deserves to note that prompts of evaluation is to ensure the LLM produce an
 evaluation score without any additional contents after assessing the 
 reasoning. By doing so, the evaluation process can be as simple as possible.
 This also leaves space for users who want to implement more complex evaluation processes.
"""

from llmpebase.prompt.generic import BasicThoughtPromptFormat


class ThoughtGenerationPrompts:
    """A base class to organize the prompt of the thought generation."""

    first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on carefully and directly generating the first reasoning step.\n",
        content="",
        target="Please generate a small and well-crafted first step as the start of reasoning, i.e., Step 1. ",
        notice="",
        tail="",
        prompt="",
    )

    next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on carefully and directly generating the next possible reasoning step for reasoning steps below.\n",
        content="\n{}\n\n",
        target="For reasoning steps within the tag {}, please directly generate the best next step, i.e., Step {}.",
        notice="",
        tail="",
        prompt="",
    )


class ThoughtEvaluationPrompts:
    """A base class to organize the prompt of the thought evaluation."""

    score_flag: str = "Evaluation Score: "

    first_step_prompt = BasicThoughtPromptFormat(
        head="Evaluate the first reasoning step (Step 1) for the given question:\n{}\n\n",
        content="\n{}\n\n",
        target="After assessing, please only generate a evaluation score ranging from 0 (critical error) to 1 (great) without additional contents.\n",
        notice=f"Present the score after '{score_flag}' for readability.",
        tail="",
        prompt="",
    )

    current_step_prompt = BasicThoughtPromptFormat(
        head="Evaluate the reasoning step {} for the given question:\n{}\n\n",
        content="Toward addressing the question, we have, thus far, obtained a series of reasoning steps: \n {}\n\n The reasoning Step {} is: \n{}\n\n",
        target="After assessing, please only generate the evaluation score ranging from 0 (critical error) to 1 (great) without additional contents.\n",
        notice=f"Present the score after '{score_flag}' for readability.",
        tail="",
        prompt="",
    )


class ThoughtSimilarityPrompts:
    """A base class to organize the prompt of the thought similarity."""

    score_flag: str = "Similarity Score: "

    sim_prompt = BasicThoughtPromptFormat(
        head="Evaluate the similarity between two reasoning steps generated for addressing the given question: \n{}\n\n",
        content="Below are two reasoning steps to be compared:\n\nA. {}\n\nB. {}\n\n",
        target="Score similarity by measuring their consistency in logic, words, and results. The similarity score ranges from 0 to 1, where a higher score means higher similarity.\n ",
        notice=f"Present the score after '{score_flag}' for readability.\n",
        tail="",
        prompt="",
    )


class BaseThoughtPrompts:
    """A base class to organize the prompts of the thought."""

    generation: ThoughtGenerationPrompts = ThoughtGenerationPrompts()
    evaluation: ThoughtEvaluationPrompts = ThoughtEvaluationPrompts()
    similarity: ThoughtSimilarityPrompts = ThoughtSimilarityPrompts()
