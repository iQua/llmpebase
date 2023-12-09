"""
The implementation of adjusting different prompts, including
zero-shot CoT.
"""

from llmpebase.model.prompting import base


class GameOf24ZeroShotPrompting(base.BaseZeroShotPrompting):
    """The standard prompt of GameOf24."""

    solution_flag: str = "The solution equation is:"

    template_prompt_head: str = "Answer the question about the problem {}"
    template_prompt_tail: str = ""

    instruction: str = " Within each step, two numbers are selected from the current number set to perform +,-,*,/ to obtain a new number. Then, these two selected numbers are deleted from the current set. After deleting, if there is no remaining number, you reach the result. Otherwise, you combine the remaining numbers and the obtained new number into a new set for the subsequent reasoning step. Therefore, the current number set of this step is the new set of the previous step."

    analysis_format: str = " Step <idx>, Current set: , Selected two numbers: , Operation: , Computed new number: , Remaining numbers: , New set: "

    question_prompt_head: str = f"""In the game of 24, you are given four numbers, and each number can be used only once. The goal is to use basic arithmetic operations (+, -, *, /) to combine these numbers and obtain a result of 24.\n  Rule: {instruction}.\nAnalysis format of each step: {analysis_format}.The given numbers are: """
    question_prompt_tail: str = ""

    answer_prompt_head: str = "Answer: Let's think step by step."

    notice: str = "When no remaining numbers, the result of the reasoning chain will be the number in the Computed new number of the analysis. Summarizing the reasoning steps into one equation with parentheses and place it after the sentence '{}' at the end of the answer for readability. "
