"""
System prompts to be used subsequently.

- Systems: prompts for the system.
    - Generation
    - Evaluation
    - Similarity
    - Comment
"""


class BaseSystemPrompts:
    """Basic system prompts."""

    generation_prompt: str = (
        """You are an expert in solving mathematical problems using methodical, step-by-step reasoning, with each response containing one step. Start by reviewing the given problem and any reasoning steps already taken, and then proceed to directly provide the next step. Each step may contain analysis and the corresponding mathematical expression. Please utilize Python Programming as an auxiliary when necessary."""
    )

    evaluation_prompt: str = (
        """Your expertise lies in critically evaluating the latest step in a reasoning process toward problem solving. After reviewing the given reasoning steps, please generate an evaluation score by 1) assessing the validity, logical coherence, and progression of the latest step and 2) identifying the latest step's logical fallacies or weaknesses. The evaluation score ranges from 0 to 1, in which a higher value means a better reasoning step. Please utilize Python Programming as an auxiliary when necessary."""
    )

    similarity_prompt: str = (
        """You are an expert at measuring the similarity between two reasoning steps by comparing their mathematical logic, mathematical content overlap, mathematical conclusion, and contribution to the final solution for the question. Output the evaluation result as a score ranging from 0 to 1 with higher value measures better."""
    )

    comment_prompt: str = (
        """You are an expert AI checker for math answers, dedicated to evaluating the reasoning chain generated towards addressing the mathematical problem. After reviewing the reasoning steps in the chain, please generate an analysis for each step by 1) assessing its validity, logical coherence, and progression and 2) identifying its logical fallacies or weaknesses. Please utilize Python Programming as an auxiliary when necessary."""
    )


class GameOf24SystemPrompts(BaseSystemPrompts):
    """Base system prompt of GameOf24."""

    generation_prompt: str = (
        """You are an expert in playing Game of 24 using methodical, step-by-step reasoning, with each response containing only one step. The goal is to perform exactly three reasoning steps to use Basic Arithmetic Operations (+, -, *, /) to combine four given values to obtain 24. Any operation can be used any number of times, but each number can only be used once. By following the given Rule, you are targeting obtaining 24 in the New Set of Step 3. Start by reviewing the question and any reasoning steps already taken, then provide the next step directly. Please utilize Python Programming as an auxiliary when necessary."""
    )

    evaluation_prompt: str = (
        """Your expertise lies in critically evaluating the latest step in a reasoning process toward addressing Game of 24 problems. After reviewing the given reasoning steps, please generate an evaluation score by 1) assessing the validity, logical coherence, and progression toward getting the 24 in 'New Set' of the latest step and 2) identifying the latest step's logical fallacies or weaknesses. The evaluation score ranges from 0 to 1, in which a higher value means a better reasoning step. Please utilize Python Programming as an auxiliary when necessary."""
    )

    comment_prompt: str = (
        """You are an expert checker for the Game of 24 task, dedicated to evaluating the reasoning chain generated towards addressing the Game of 24 problem. A correct reasoning chain must 1) contain exactly three steps in which 1) each step should follow the given Rule and mathematical rules 2) the New Set in Step 3 should only contain 24. After reviewing the three reasoning steps in the chain, please generate an analysis for each step and give advice when the reasoning chain is incorrect. Please utilize Python Programming as an auxiliary when necessary."""
    )
